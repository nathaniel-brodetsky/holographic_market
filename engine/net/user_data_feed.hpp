#pragma once
//
// user_data_feed.hpp
// holo::net — Binance Futures User Data Stream consumer.
//
// Maintains a persistent wss connection to wss://<host>/ws/<listenKey>,
// parses ORDER_TRADE_UPDATE events, and pushes state transitions into
// OMSCore. Assumes the listenKey has already been obtained (and is kept
// alive via a periodic PUT /fapi/v1/listenKey) via the REST layer
// elsewhere in your codebase — this module only consumes the stream.
//
// JSON parsing uses simdjson::dom::parser. dom::parser owns and reuses its
// internal padded buffer across parse() calls, so steady-state message
// handling does not allocate once the buffer has grown to its high-water
// mark. dom (vs. ondemand) is used deliberately here: ORDER_TRADE_UPDATE
// fields are accessed out of declaration order below, which ondemand's
// forward-only cursor does not support without extra bookkeeping.
//
// RECONNECTION: start() runs an outer backoff loop around run(). Binance
// will also proactively close the stream (or send `listenKeyExpired`) if
// the listenKey isn't refreshed — that path logs and returns from run(),
// which the backoff loop will then retry. A stale listenKey after
// reconnect will fail the handshake with an HTTP error, which propagates
// as an exception out of run() and is likewise handled by the backoff
// loop. If your listenKey-refresh lives in a different component, make
// sure it's refreshing on its own timer independent of this reconnect
// loop (do not couple listenKey refresh to WS reconnect attempts).
//
#include "oms_core.hpp"

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <simdjson.h>

#include <algorithm>
#include <charconv>
#include <iostream>
#include <string>

namespace holo::net {

namespace asio      = boost::asio;
namespace beast      = boost::beast;
namespace websocket  = beast::websocket;
using tcp            = asio::ip::tcp;

class UserDataFeed {
public:
    UserDataFeed(asio::any_io_executor exec,
                 asio::ssl::context& ssl_ctx,
                 std::string host,
                 std::string listen_key,
                 OMSCore& oms)
        : exec_(exec)
        , ssl_ctx_(ssl_ctx)
        , host_(std::move(host))
        , listen_key_(std::move(listen_key))
        , oms_(oms) {}

    // Fire-and-forget launch with exponential backoff reconnect. Runs
    // until the process exits; there is no clean stop() because in
    // practice you want this feed alive for the process lifetime (an OMS
    // with a dead user-data feed is silently blind to fills — better to
    // crash-and-restart the whole process via your supervisor than to
    // limp along without it).
    void start() {
        asio::co_spawn(exec_, run_with_backoff(), [](std::exception_ptr e) {
            if (!e) return;
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                std::cerr << "[UserDataFeed] fatal, backoff loop itself threw: " << ex.what() << "\n";
            }
        });
    }

    // Update the listenKey used on the *next* reconnect. Does not affect
    // an already-established connection (Binance streams are bound to the
    // listenKey at handshake time; to rotate mid-flight you must
    // reconnect, which the backoff loop will do naturally on the next
    // disconnect — or call force_reconnect() below).
    void set_listen_key(std::string key) { listen_key_ = std::move(key); }

private:
    asio::awaitable<void> run_with_backoff() {
        auto backoff = std::chrono::milliseconds(200);
        constexpr auto kMaxBackoff = std::chrono::milliseconds(10'000);

        for (;;) {
            try {
                co_await run();
                // run() only returns normally on listenKeyExpired or a
                // clean server close; treat both as "reconnect promptly."
                backoff = std::chrono::milliseconds(200);
            } catch (const std::exception& ex) {
                std::cerr << "[UserDataFeed] connection error: " << ex.what()
                          << " — reconnecting in " << backoff.count() << "ms\n";
            }

            asio::steady_timer t(exec_);
            t.expires_after(backoff);
            co_await t.async_wait(asio::use_awaitable);
            backoff = std::min(backoff * 2, kMaxBackoff);
        }
    }

    asio::awaitable<void> run() {
        // A fresh stream per connection attempt — beast websocket streams
        // are not meant to be reused/re-handshaken after a failed or
        // closed connection.
        websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(exec_, ssl_ctx_);
        tcp::resolver resolver(exec_);

        auto const results = co_await resolver.async_resolve(host_, "443", asio::use_awaitable);

        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        co_await beast::get_lowest_layer(ws).async_connect(results, asio::use_awaitable);

        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }

        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        co_await ws.next_layer().async_handshake(asio::ssl::stream_base::client,
                                                   asio::use_awaitable);

        beast::get_lowest_layer(ws).expires_never();
        ws.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws.set_option(websocket::stream_base::decorator([](websocket::request_type& req) {
            req.set(beast::http::field::user_agent, "holo-oms/1.0");
        }));

        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(host_, target, asio::use_awaitable);
        std::cerr << "[UserDataFeed] connected: " << target << "\n";

        beast::flat_buffer buffer;
        for (;;) {
            buffer.clear();
            co_await ws.async_read(buffer, asio::use_awaitable);
            if (handle_message(static_cast<const char*>(buffer.data().data()), buffer.size())) {
                co_return;  // listenKeyExpired — caller's backoff loop will reconnect
                            // once a fresh listenKey has been set via set_listen_key()
            }
        }
    }

    // Returns true if the caller should tear down and reconnect (i.e. the
    // listenKey just expired).
    bool handle_message(const char* data, size_t len) {
        simdjson::dom::element doc;
        if (auto err = parser_.parse(data, len).get(doc)) {
            std::cerr << "[UserDataFeed] parse error: " << simdjson::error_message(err) << "\n";
            return false;
        }

        std::string_view event_type;
        if (doc["e"].get(event_type)) return false;  // no event type -> not an event we handle

        if (event_type == "ORDER_TRADE_UPDATE") {
            simdjson::dom::element o;
            if (!doc["o"].get(o)) handle_order_trade_update(o);
        } else if (event_type == "listenKeyExpired") {
            std::cerr << "[UserDataFeed] listenKey expired — reconnecting; make sure your "
                         "REST layer has already pushed a fresh key via set_listen_key().\n";
            return true;
        }
        // ACCOUNT_UPDATE, MARGIN_CALL, etc. can be dispatched here as needed.
        return false;
    }

    void handle_order_trade_update(simdjson::dom::element o) {
        std::string_view client_id, status_str, z_str;
        int64_t exch_order_id = 0;

        // Required fields — malformed/incomplete event is dropped rather
        // than applied with partial data.
        if (o["c"].get(client_id) || o["X"].get(status_str) || o["z"].get(z_str) ||
            o["i"].get(exch_order_id)) {
            std::cerr << "[UserDataFeed] malformed ORDER_TRADE_UPDATE, dropping\n";
            return;
        }

        std::string_view l_price_str;  // last fill price, absent/"0" if no fill on this event
        auto _l_err = o["L"].get(l_price_str); (void)_l_err;

        // "n" = commission charged for *this* individual trade execution
        // (already a per-event delta, not a running cumulative total —
        // Binance does not report a cumulative commission field). Absent
        // on non-fill events (e.g. a plain NEW ack), in which case there's
        // nothing to charge for this update.
        std::string_view commission_str;
        auto _n_err = o["n"].get(commission_str); (void)_n_err;

        int64_t t_ms = 0;
        auto _t_err = o["T"].get(t_ms); (void)_t_err;
        const int64_t event_ns = t_ms * 1'000'000;

        const double cum_filled = svtod(z_str);
        const double last_px    = l_price_str.empty() ? 0.0 : svtod(l_price_str);
        const double commission = commission_str.empty() ? 0.0 : svtod(commission_str);
        const OrderStatus status = map_status(status_str);
        const uint64_t key = fnv1a64(client_id);

        if (!oms_.apply_update(key, status, cum_filled, last_px, exch_order_id, event_ns, commission)) {
            // Not necessarily a bug: this fires for any client_order_id
            // this OMS instance didn't itself register (e.g. an order
            // placed manually on the exchange UI, or from a previous
            // process run against the same API key). Downgrade to a
            // debug-level trace if that's expected in your deployment.
            std::cerr << "[UserDataFeed] update for unknown client_order_id=" << client_id
                      << " (order not registered in this OMS instance)\n";
        }
    }

    static double svtod(std::string_view sv) noexcept {
        double v = 0.0;
        std::from_chars(sv.data(), sv.data() + sv.size(), v);
        return v;
    }

    static OrderStatus map_status(std::string_view s) noexcept {
        if (s == "NEW") return OrderStatus::Open;
        if (s == "PARTIALLY_FILLED") return OrderStatus::Partial;
        if (s == "FILLED") return OrderStatus::Filled;
        if (s == "CANCELED" || s == "EXPIRED") return OrderStatus::Canceled;
        if (s == "REJECTED") return OrderStatus::Rejected;
        return OrderStatus::Open;
    }

    asio::any_io_executor exec_;
    asio::ssl::context& ssl_ctx_;
    std::string host_;
    std::string listen_key_;
    OMSCore& oms_;
    simdjson::dom::parser parser_;
};

}  // namespace holo::net