#pragma once
//
// user_data_feed.hpp
// holo::net — Binance Futures User Data Stream consumer.
//
// Maintains a persistent wss connection to wss://<host>/ws/<listenKey>,
// parses ORDER_TRADE_UPDATE events, and pushes state transitions into
// OMSCore. Assumes the listenKey has already been obtained (and is kept
// alive) via the REST layer elsewhere in your codebase — this module only
// consumes the stream.
//
// JSON parsing uses simdjson::dom::parser. dom::parser owns and reuses its
// internal padded buffer across parse() calls, so steady-state message
// handling does not allocate once the buffer has grown to its high-water
// mark. dom (vs. ondemand) is used deliberately here: ORDER_TRADE_UPDATE
// fields are accessed out of declaration order below, which ondemand's
// forward-only cursor does not support without extra bookkeeping.
//
#include "oms_core.hpp"

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <simdjson.h>

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
        : resolver_(exec)
        , ws_(exec, ssl_ctx)
        , host_(std::move(host))
        , listen_key_(std::move(listen_key))
        , oms_(oms) {}

    // Fire-and-forget launch. On unrecoverable error the coroutine
    // completes and the exception is logged here; wire your own
    // reconnect/backoff loop around start() if you want auto-reconnect
    // (recommended: also refresh the listenKey via REST on reconnect).
    void start(asio::any_io_executor exec) {
        asio::co_spawn(exec, run(), [](std::exception_ptr e) {
            if (!e) return;
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                std::cerr << "[UserDataFeed] fatal: " << ex.what() << "\n";
            }
        });
    }

    asio::awaitable<void> run() {
        auto const results = co_await resolver_.async_resolve(host_, "443", asio::use_awaitable);

        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));
        co_await beast::get_lowest_layer(ws_).async_connect(results, asio::use_awaitable);

        if (!SSL_set_tlsext_host_name(ws_.next_layer().native_handle(), host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }

        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));
        co_await ws_.next_layer().async_handshake(asio::ssl::stream_base::client,
                                                    asio::use_awaitable);

        beast::get_lowest_layer(ws_).expires_never();
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws_.set_option(websocket::stream_base::decorator([](websocket::request_type& req) {
            req.set(beast::http::field::user_agent, "holo-oms/1.0");
        }));

        const std::string target = "/ws/" + listen_key_;
        co_await ws_.async_handshake(host_, target, asio::use_awaitable);
        std::cerr << "[UserDataFeed] connected: " << target << "\n";

        beast::flat_buffer buffer;
        for (;;) {
            buffer.clear();
            co_await ws_.async_read(buffer, asio::use_awaitable);
            handle_message(static_cast<const char*>(buffer.data().data()), buffer.size());
        }
    }

private:
    void handle_message(const char* data, size_t len) {
        simdjson::dom::element doc;
        if (auto err = parser_.parse(data, len).get(doc)) {
            std::cerr << "[UserDataFeed] parse error: " << simdjson::error_message(err) << "\n";
            return;
        }

        std::string_view event_type;
        if (doc["e"].get(event_type)) return;  // no event type -> not an event we handle

        if (event_type == "ORDER_TRADE_UPDATE") {
            simdjson::dom::element o;
            if (!doc["o"].get(o)) handle_order_trade_update(o);
        } else if (event_type == "listenKeyExpired") {
            std::cerr << "[UserDataFeed] listenKey expired — caller must fetch a fresh "
                         "key and reconnect.\n";
        }
        // ACCOUNT_UPDATE, MARGIN_CALL, etc. can be dispatched here as needed.
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
        o["L"].get(l_price_str);

        int64_t t_ms = 0;
        o["T"].get(t_ms);
        const int64_t event_ns = t_ms * 1'000'000;

        const double cum_filled = svtod(z_str);
        const double last_px    = l_price_str.empty() ? 0.0 : svtod(l_price_str);
        const OrderStatus status = map_status(status_str);
        const uint64_t key = fnv1a64(client_id);

        if (!oms_.apply_update(key, status, cum_filled, last_px, exch_order_id, event_ns)) {
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

    tcp::resolver resolver_;
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws_;
    std::string host_;
    std::string listen_key_;
    OMSCore& oms_;
    simdjson::dom::parser parser_;
};

}  // namespace holo::net
