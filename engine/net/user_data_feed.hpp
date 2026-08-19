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

#include <boost/beast/http.hpp>

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace holo::net {

namespace asio      = boost::asio;
namespace beast      = boost::beast;
namespace websocket  = beast::websocket;
using tcp            = asio::ip::tcp;

class UserDataFeed {
public:
    // `host` is the REST host (testnet.binancefuture.com) -- used for
    // listenKey POST/PUT/DELETE and order-reconciliation GETs. `ws_host`
    // is the WebSocket host (stream.binancefuture.com on testnet) -- a
    // DIFFERENT subdomain on Binance Futures Testnet, used only for the
    // actual WS connection. Conflating these two is what caused every
    // single WS handshake attempt to time out -- see the file-level
    // comment in the patch that introduced this split for the full story.
    UserDataFeed(asio::any_io_executor exec,
                 asio::ssl::context& ssl_ctx,
                 std::string host,
                 std::string ws_host,
                 std::string listen_key,
                 std::string api_key,
                 std::string api_secret,
                 OMSCore& oms)
        : exec_(exec)
        , ssl_ctx_(ssl_ctx)
        , host_(std::move(host))
        , ws_host_(std::move(ws_host))
        , listen_key_(std::move(listen_key))
        , api_key_(std::move(api_key))
        , api_secret_(std::move(api_secret))
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

    // Called on a MARGIN_CALL account-stream event. This is intentionally
    // decoupled from OMSCore's order-update callback: a margin call is an
    // account-level event, not an order-level one, and the caller almost
    // certainly wants it wired to something like
    // ExecutionEngine::force_halt() rather than routed through OMS order
    // tracking. Not set by default — MARGIN_CALL is logged loudly either
    // way (see handle_message), this is purely for the caller to also
    // *act* on it.
    void set_margin_call_callback(std::function<void(std::string_view)> cb) {
        on_margin_call_ = std::move(cb);
    }

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
        // Always fetch a brand-new listenKey before every connection
        // attempt, including reconnects, rather than reusing whatever
        // key was last set. Diagnostic testing (stage-by-stage logging)
        // showed the WS handshake step reliably timing out -- never
        // receiving a 101 response -- specifically on reconnects reusing
        // the SAME key that worked fine for the initial connection.
        // A fresh key sidesteps the ambiguity of genuine expiry vs. some
        // other same-key-reconnect quirk against this venue.
        // Explicitly close/invalidate whatever session Binance currently
        // associates with this account BEFORE requesting a new key --
        // POST alone does not guarantee a genuinely new key (it returns
        // the existing active one if there is one), and a stuck
        // server-side session from an earlier connection that never
        // cleanly closed is the leading suspect for why even a brand-new
        // process's first connection attempt has been failing.
        delete_listen_key();

        try {
            listen_key_ = fetch_fresh_listen_key();
            std::cerr << "[UserDataFeed][diag] fetched fresh listenKey (len="
                      << listen_key_.size() << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed] failed to fetch a fresh listenKey: " << e.what()
                      << "\n";
            throw;  // let the backoff loop retry
        }

        // A fresh stream per connection attempt — beast websocket streams
        // are not meant to be reused/re-handshaken after a failed or
        // closed connection.
        websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(exec_, ssl_ctx_);
        tcp::resolver resolver(exec_);

        std::cerr << "[UserDataFeed][diag] resolving " << ws_host_ << "...\n";
        auto const results = co_await resolver.async_resolve(ws_host_, "443", asio::use_awaitable);
        std::cerr << "[UserDataFeed][diag] resolved, connecting...\n";

        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        co_await beast::get_lowest_layer(ws).async_connect(results, asio::use_awaitable);
        std::cerr << "[UserDataFeed][diag] TCP connected, starting TLS handshake...\n";

        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), ws_host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }

        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        co_await ws.next_layer().async_handshake(asio::ssl::stream_base::client,
                                                   asio::use_awaitable);
        std::cerr << "[UserDataFeed][diag] TLS handshake done, starting WS handshake "
                     "(listen_key_ len=" << listen_key_.size() << ")...\n";

        beast::get_lowest_layer(ws).expires_never();
        ws.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws.set_option(websocket::stream_base::decorator([](websocket::request_type& req) {
            req.set(beast::http::field::user_agent, "holo-oms/1.0");
        }));

        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(ws_host_, target, asio::use_awaitable);
        std::cerr << "[UserDataFeed] connected: " << target << "\n";

        // Binance does not replay missed events on reconnect -- anything
        // that happened during the gap is gone unless we go fetch it
        // ourselves. See the file-level comment above reconcile_open_orders().
        reconcile_open_orders();

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
        } else if (event_type == "MARGIN_CALL") {
            // Binance sends this when one or more positions are at risk of
            // liquidation. This is not something to merely log and move
            // on from — surface it loudly and let the caller decide how to
            // react (typically: halt new signal processing immediately).
            std::cerr << "[UserDataFeed] *** MARGIN_CALL received *** — one or more "
                         "positions are at liquidation risk.\n";
            if (on_margin_call_) on_margin_call_(std::string_view(data, len));
        } else if (event_type == "ACCOUNT_UPDATE") {
            // Routine (balance/position changes from fills, funding,
            // etc.) — traced rather than silently dropped so an
            // unexpected pattern (e.g. a position change with no
            // corresponding ORDER_TRADE_UPDATE we recognize) is at least
            // visible in logs. Not wired to any action by default.
            std::cerr << "[UserDataFeed] ACCOUNT_UPDATE received (trace only, not acted on)\n";
        }
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

        std::string_view commission_str;  // commission for THIS event only — Binance sends no
                                           // cumulative-commission field, unlike "z" for qty.
        auto _n_err = o["n"].get(commission_str); (void)_n_err;

        int64_t t_ms = 0;
        auto _t_err = o["T"].get(t_ms); (void)_t_err;
        const int64_t event_ns = t_ms * 1'000'000;

        const double cum_filled      = svtod(z_str);
        const double last_px         = l_price_str.empty() ? 0.0 : svtod(l_price_str);
        const double commission_delta = commission_str.empty() ? 0.0 : svtod(commission_str);
        const OrderStatus status = map_status(status_str);
        const uint64_t key = fnv1a64(client_id);

        if (!oms_.apply_update(key, status, cum_filled, last_px, exch_order_id, event_ns,
                                commission_delta)) {
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

    static OrderStatus map_status(std::string_view s) {
        if (s == "NEW") return OrderStatus::Open;
        if (s == "PARTIALLY_FILLED") return OrderStatus::Partial;
        if (s == "FILLED") return OrderStatus::Filled;
        if (s == "CANCELED" || s == "EXPIRED") return OrderStatus::Canceled;
        if (s == "REJECTED") return OrderStatus::Rejected;
        // Binance introducing a status we don't recognize should be loud,
        // not a silent "treat it as still-open forever". Defaulting to Open
        // is still the safest *behavior* (we don't want to prematurely mark
        // a genuinely-live order as terminal), but it must not be silent.
        std::cerr << "[UserDataFeed] unknown order status \"" << s
                  << "\" — defaulting to Open; Binance API may have changed.\n";
        return OrderStatus::Open;
    }

    // ------------------------------------------------------------------
    // Reconciliation: see the file-level comment near the top of this
    // patch for why this exists. Runs once per successful (re)connect.
    // ------------------------------------------------------------------
    void reconcile_open_orders() {
        std::vector<OrderRecord> local_live;
        oms_.get_stale_orders(0, local_live);  // age threshold 0 -> every
                                                // currently-live order,
                                                // reusing the existing
                                                // liveness filter.
        if (local_live.empty()) return;

        std::string body;
        try {
            body = rest_get_signed("/fapi/v1/openOrders", "");
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed] reconcile: openOrders REST call failed: "
                      << e.what()
                      << " -- skipping reconciliation this cycle, will retry on next reconnect\n";
            return;
        }

        simdjson::dom::parser p;
        simdjson::dom::element doc;
        if (auto err = p.parse(body).get(doc)) {
            std::cerr << "[UserDataFeed] reconcile: openOrders parse error: "
                      << simdjson::error_message(err) << " -- raw body: " << body << "\n";
            return;
        }

        std::unordered_set<uint64_t> still_open_keys;
        simdjson::dom::array arr;
        if (!doc.get(arr)) {
            for (auto order : arr) {
                std::string_view coid;
                if (!order["clientOrderId"].get(coid)) {
                    still_open_keys.insert(fnv1a64(coid));
                }
            }
        }

        for (const auto& rec : local_live) {
            const std::string_view coid = view(rec.client_order_id);
            const uint64_t key = fnv1a64(coid);
            if (still_open_keys.count(key)) continue;  // still open per exchange -- fine

            // Locally live, but the exchange no longer lists it as open
            // -- it reached a terminal state while we were disconnected
            // and we missed the ORDER_TRADE_UPDATE for it.
            std::cerr << "[UserDataFeed] reconcile: " << coid
                      << " is live locally but not in exchange openOrders -- fetching its "
                         "authoritative final state\n";
            reconcile_single_order(view(rec.symbol), coid, key);
        }
    }

    void reconcile_single_order(std::string_view symbol, std::string_view client_order_id,
                                 uint64_t key) {
        std::string query = "symbol=" + std::string(symbol) +
                             "&origClientOrderId=" + std::string(client_order_id);
        std::string body;
        try {
            body = rest_get_signed("/fapi/v1/order", query);
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed] reconcile: order lookup failed for "
                      << client_order_id << ": " << e.what() << "\n";
            return;
        }

        simdjson::dom::parser p;
        simdjson::dom::element doc;
        if (auto err = p.parse(body).get(doc)) {
            std::cerr << "[UserDataFeed] reconcile: order lookup parse error for "
                      << client_order_id << ": " << simdjson::error_message(err) << "\n";
            return;
        }

        std::string_view status_str, z_str, avg_px_str;
        int64_t exch_order_id = 0;
        if (doc["status"].get(status_str) || doc["executedQty"].get(z_str) ||
            doc["orderId"].get(exch_order_id)) {
            std::cerr << "[UserDataFeed] reconcile: malformed order lookup response for "
                      << client_order_id << ": " << body << "\n";
            return;
        }
        auto _avg_err = doc["avgPrice"].get(avg_px_str); (void)_avg_err;

        const OrderStatus status = map_status(status_str);
        const double cum_filled = svtod(z_str);
        const double avg_px = avg_px_str.empty() ? 0.0 : svtod(avg_px_str);
        const int64_t event_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // No per-event commission figure from this endpoint the way
        // ORDER_TRADE_UPDATE gives one -- pass 0.0. This path only runs
        // for a genuinely rare gap (disconnect exactly spanning a fill),
        // not the primary path, so an approximate commission here is an
        // acceptable tradeoff against leaving the OMS's status
        // permanently wrong for that order.
        if (!oms_.apply_update(key, status, cum_filled, avg_px, exch_order_id, event_ns, 0.0)) {
            std::cerr << "[UserDataFeed] reconcile: apply_update found no matching order for "
                      << client_order_id << " (already released?)\n";
        } else {
            std::cerr << "[UserDataFeed] reconcile: recovered " << client_order_id
                      << " -> status=" << static_cast<int>(status)
                      << " filled=" << cum_filled << "\n";
        }
    }

    // Blocking REST POST for a brand-new listenKey -- mirrors
    // main_live.cpp's fetch_listen_key() exactly (same endpoint, same
    // blocking-socket style), duplicated here rather than shared because
    // main_live.cpp's copy is a free function with no access to this
    // class's members, and introducing a shared REST-utility header for
    // one function isn't worth it yet.
    // Best-effort: closes/invalidates whatever listenKey session Binance
    // currently associates with this account. -1125 "does not exist" (no
    // active key to delete) is an entirely expected, harmless outcome
    // here, not something to treat as fatal -- this call exists purely
    // to clear any possibly-stuck server-side session before creating a
    // fresh one via fetch_fresh_listen_key().
    void delete_listen_key() noexcept {
        namespace http = beast::http;
        namespace ssl  = asio::ssl;
        try {
            asio::io_context ioc;
            ssl::context ctx(ssl::context::tlsv12_client);
            ctx.set_default_verify_paths();
            asio::ip::tcp::resolver resolver(ioc);
            beast::ssl_stream<beast::tcp_stream> stream(ioc, ctx);

            auto const results = resolver.resolve(host_, "443");
            beast::get_lowest_layer(stream).connect(results);
            if (!SSL_set_tlsext_host_name(stream.native_handle(), host_.c_str())) {
                throw std::runtime_error("SNI failed (delete_listen_key)");
            }
            stream.handshake(ssl::stream_base::client);

            http::request<http::empty_body> req{http::verb::delete_, "/fapi/v1/listenKey", 11};
            req.set(http::field::host, host_);
            req.set("X-MBX-APIKEY", api_key_);

            http::write(stream, req);
            beast::flat_buffer buffer;
            http::response<http::string_body> res;
            http::read(stream, buffer, res);
            beast::error_code ec;
            stream.shutdown(ec);
            std::cerr << "[UserDataFeed][diag] DELETE listenKey response: " << res.body()
                      << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed][diag] DELETE listenKey failed (likely harmless "
                         "if no key was active): " << e.what() << "\n";
        }
    }

    std::string fetch_fresh_listen_key() {
        namespace http = beast::http;
        namespace ssl  = asio::ssl;

        asio::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        ctx.set_default_verify_paths();
        asio::ip::tcp::resolver resolver(ioc);
        beast::ssl_stream<beast::tcp_stream> stream(ioc, ctx);

        auto const results = resolver.resolve(host_, "443");
        beast::get_lowest_layer(stream).connect(results);
        if (!SSL_set_tlsext_host_name(stream.native_handle(), host_.c_str())) {
            throw std::runtime_error("SNI failed (fetch_fresh_listen_key)");
        }
        stream.handshake(ssl::stream_base::client);

        http::request<http::empty_body> req{http::verb::post, "/fapi/v1/listenKey", 11};
        req.set(http::field::host, host_);
        req.set("X-MBX-APIKEY", api_key_);

        http::write(stream, req);
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);
        beast::error_code ec;
        stream.shutdown(ec);

        const std::string& body = res.body();
        size_t pos = body.find(R"("listenKey":")");
        if (pos != std::string::npos) {
            pos += 13;
            return body.substr(pos, body.find("\"", pos) - pos);
        }
        throw std::runtime_error("fetch_fresh_listen_key: no listenKey in response: " + body);
    }

    // Blocking signed REST GET -- mirrors main_live.cpp's
    // fetch_listen_key()/renew_listen_key() style (fresh io_context per
    // call, blocking sockets) rather than introducing a separate async
    // REST client just for this. `query` should NOT include timestamp or
    // signature -- both are appended here.
    std::string rest_get_signed(const std::string& path, std::string query) {
        namespace http = beast::http;
        namespace ssl  = asio::ssl;

        const int64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        query += (query.empty() ? "" : "&") + std::string("timestamp=") + std::to_string(ts);
        query += "&signature=" + hmac_sha256_hex(api_secret_, query);

        asio::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        ctx.set_default_verify_paths();
        asio::ip::tcp::resolver resolver(ioc);
        beast::ssl_stream<beast::tcp_stream> stream(ioc, ctx);

        auto const results = resolver.resolve(host_, "443");
        beast::get_lowest_layer(stream).connect(results);
        if (!SSL_set_tlsext_host_name(stream.native_handle(), host_.c_str())) {
            throw std::runtime_error("SNI failed (reconcile REST call)");
        }
        stream.handshake(ssl::stream_base::client);

        http::request<http::empty_body> req{http::verb::get, path + "?" + query, 11};
        req.set(http::field::host, host_);
        req.set("X-MBX-APIKEY", api_key_);

        http::write(stream, req);
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);
        beast::error_code ec;
        stream.shutdown(ec);
        return res.body();
    }

    static std::string hmac_sha256_hex(const std::string& key, const std::string& msg) {
        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int len = 0;
        HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
             reinterpret_cast<const unsigned char*>(msg.data()), msg.size(), digest, &len);
        static const char* hex = "0123456789abcdef";
        std::string out;
        out.reserve(len * 2U);
        for (unsigned int i = 0; i < len; ++i) {
            out.push_back(hex[(digest[i] >> 4) & 0xF]);
            out.push_back(hex[digest[i] & 0xF]);
        }
        return out;
    }

    asio::any_io_executor exec_;
    asio::ssl::context& ssl_ctx_;
    std::string host_;      // REST host (testnet.binancefuture.com)
    std::string ws_host_;   // WebSocket host (stream.binancefuture.com) -- see constructor comment
    std::string listen_key_;
    std::string api_key_;
    std::string api_secret_;
    OMSCore& oms_;
    simdjson::dom::parser parser_;
    std::function<void(std::string_view)> on_margin_call_;
};

}  // namespace holo::net