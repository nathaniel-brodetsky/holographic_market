#pragma once
//
// binance_gateway.hpp
// holo::net — IOrderGateway implementation over the Binance Futures
// WebSocket API (wss://<host>/ws-fapi/v1).
//
// Design notes:
//  - Order acks on this socket are used only to catch *request-level*
//    rejects (bad signature, bad precision, insufficient margin, etc.)
//    early and short-circuit the execution engine's wait — actual fill
//    state always flows through UserDataFeed's ORDER_TRADE_UPDATE stream,
//    which remains the single source of truth for OMSCore. This gateway
//    never calls oms_.apply_update() with a Filled/Partial status, only
//    Rejected, and only for requests *this* gateway instance sent.
//  - Without ack correlation, a WS-API-level reject (which never produces
//    an ORDER_TRADE_UPDATE, because the order was never accepted by the
//    matching engine) would leave the execution engine's monitor
//    coroutine blocked on its channel until the leg-timeout fires — i.e.
//    every rejected order would look identical to "still resting" for
//    50ms. Correlating "id" in the ack back to the client_order_id we
//    sent closes that gap.
//
#include "execution_engine.hpp"

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <openssl/err.h>
#include <openssl/hmac.h>

#include <simdjson.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace holo::net {

namespace asio      = boost::asio;
namespace beast      = boost::beast;
namespace websocket  = beast::websocket;
using tcp            = asio::ip::tcp;

namespace detail {

// HMAC-SHA256, hex-encoded — Binance's required signature scheme for
// SIGNED endpoints (query-string / JSON body form, per the "payload"
// construction below).
inline std::string hmac_sha256_hex(std::string_view secret, std::string_view msg) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    const unsigned char* result =
        HMAC(EVP_sha256(),
             secret.data(), static_cast<int>(secret.size()),
             reinterpret_cast<const unsigned char*>(msg.data()), msg.size(),
             digest, &len);
    if (!result) {
        throw std::runtime_error("HMAC-SHA256 signing failed: " +
                                  std::to_string(ERR_get_error()));
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(len * 2, '\0');
    for (unsigned int i = 0; i < len; ++i) {
        out[2 * i]     = kHex[(digest[i] >> 4) & 0xF];
        out[2 * i + 1] = kHex[digest[i] & 0xF];
    }
    return out;
}

struct SymbolPrecision {
    int qty_decimals;
    int price_decimals;
};

// Hardcoded starting point ONLY. A real deployment must fetch
// stepSize/tickSize per symbol from GET /fapi/v1/exchangeInfo at startup
// (and refresh on any symbol-list change) — Binance revises these over
// time, and a stale hardcoded table produces silent -1111 "Precision is
// over the maximum defined for this asset" rejects that look identical
// to a normal GTX post-only reject unless you're watching order-reject
// reasons closely.
inline const std::unordered_map<std::string_view, SymbolPrecision>& symbol_precision_table() {
    static const std::unordered_map<std::string_view, SymbolPrecision> t = {
        {"BTCUSDT", {3, 1}},
        {"ETHUSDT", {3, 2}},
        {"BNBUSDT", {2, 2}},
        {"SOLUSDT", {0, 3}},
    };
    return t;
}

inline SymbolPrecision precision_for(std::string_view symbol) {
    const auto& t = symbol_precision_table();
    auto it = t.find(symbol);
    if (it != t.end()) return it->second;
    std::cerr << "[BinanceGateway] WARNING: no precision entry for symbol '" << symbol
              << "' — defaulting to (qty=2dp, price=2dp). VERIFY against exchangeInfo "
                 "before trading this symbol.\n";
    return {2, 2};
}

inline std::string format_decimal(double v, int decimals) {
    char fmt[16];
    std::snprintf(fmt, sizeof(fmt), "%%.%df", decimals);
    char buf[48];
    std::snprintf(buf, sizeof(buf), fmt, v);
    return std::string(buf);
}

inline int64_t now_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace detail

class BinanceGateway final : public IOrderGateway {
public:
    // `oms` is used ONLY to short-circuit WS-API-level rejects (see the
    // header comment) — this class never reports fills into it.
    BinanceGateway(asio::any_io_executor exec,
                   asio::ssl::context& ssl_ctx,
                   std::string api_key,
                   std::string api_secret,
                   OMSCore& oms,
                   std::string host = "testnet.binancefuture.com")
        : exec_(exec)
        , ssl_ctx_(ssl_ctx)
        , api_key_(std::move(api_key))
        , api_secret_(std::move(api_secret))
        , oms_(oms)
        , host_(std::move(host)) {}

    void start() {
        asio::co_spawn(exec_, run_with_backoff(), asio::detached);
    }

    asio::awaitable<void> send_limit_post_only(std::string_view cid, std::string_view sym,
                                                Side side, double price, double qty) override {
        const uint64_t id = next_msg_id();
        track_pending(id, cid);
        send_msg(build_order_msg(id, cid, sym, side == Side::Buy ? "BUY" : "SELL",
                                  "LIMIT", qty, price, "GTX"));
        co_return;
    }

    asio::awaitable<void> send_market(std::string_view cid, std::string_view sym, Side side,
                                       double qty) override {
        const uint64_t id = next_msg_id();
        track_pending(id, cid);
        send_msg(build_order_msg(id, cid, sym, side == Side::Buy ? "BUY" : "SELL",
                                  "MARKET", qty, 0.0, ""));
        co_return;
    }

    asio::awaitable<void> cancel_order(std::string_view sym, std::string_view cid) override {
        const uint64_t id = next_msg_id();
        const int64_t ts = detail::now_ms();
        std::string payload = "apiKey=" + api_key_ + "&origClientOrderId=" + std::string(cid) +
                               "&symbol=" + std::string(sym) + "&timestamp=" + std::to_string(ts);
        const std::string sig = detail::hmac_sha256_hex(api_secret_, payload);

        std::string json = R"({"id":")" + std::to_string(id) + R"(","method":"order.cancel","params":{)";
        json += R"("apiKey":")" + api_key_ + R"(","origClientOrderId":")" + std::string(cid) +
                R"(","symbol":")" + std::string(sym) + R"(","timestamp":)" + std::to_string(ts) +
                R"(,"signature":")" + sig + R"("}})";
        send_msg(std::move(json));
        co_return;
    }

private:
    asio::awaitable<void> run_with_backoff() {
        auto backoff = std::chrono::milliseconds(200);
        constexpr auto kMaxBackoff = std::chrono::milliseconds(10'000);
        for (;;) {
            try {
                co_await run();
            } catch (const std::exception& ex) {
                std::cerr << "[BinanceGateway] connection error: " << ex.what()
                          << " — reconnecting in " << backoff.count() << "ms\n";
            }
            connected_ = false;
            // Ack correlation is meaningless across a reconnect: we do not
            // know whether pending requests were actually processed by
            // Binance before the socket died. Do NOT mark them Rejected
            // (they may have gone through) — surface loudly instead so an
            // operator (or your reconciliation-on-reconnect routine, e.g.
            // GET /fapi/v1/openOrders) can resolve them.
            {
                std::lock_guard lk(pending_mtx_);
                if (!pending_.empty()) {
                    std::cerr << "[BinanceGateway] CRITICAL: " << pending_.size()
                              << " order request(s) had unknown outcome at disconnect: ";
                    for (auto& [id, cid] : pending_) std::cerr << cid << " ";
                    std::cerr << "— reconcile via REST before trusting OMS state for these.\n";
                    pending_.clear();
                }
            }
            asio::steady_timer t(exec_);
            t.expires_after(backoff);
            co_await t.async_wait(asio::use_awaitable);
            backoff = std::min(backoff * 2, kMaxBackoff);
        }
    }

    // Precondition/invariant this whole class relies on: run_with_backoff(),
    // run(), send_msg()'s posted continuation, and do_write()'s completion
    // handler all execute on the *same* executor (single-threaded
    // io_context, or all posted through one strand). That's what lets
    // ws_ (a resettable optional, rebuilt fresh each reconnect — beast
    // websocket streams are not meant to be re-handshaken after a failed
    // connection) be touched without its own mutex: everything touching
    // it is already serialized by construction. If you ever run this
    // gateway's executor across multiple threads without a strand, ws_
    // needs to move behind a strand (asio::make_strand(exec_)) instead.
    asio::awaitable<void> run() {
        ws_.emplace(exec_, ssl_ctx_);
        tcp::resolver resolver(exec_);

        auto const results = co_await resolver.async_resolve(host_, "443", asio::use_awaitable);

        beast::get_lowest_layer(*ws_).expires_after(std::chrono::seconds(30));
        co_await beast::get_lowest_layer(*ws_).async_connect(results, asio::use_awaitable);

        if (!SSL_set_tlsext_host_name(ws_->next_layer().native_handle(), host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }

        beast::get_lowest_layer(*ws_).expires_after(std::chrono::seconds(30));
        co_await ws_->next_layer().async_handshake(asio::ssl::stream_base::client,
                                                     asio::use_awaitable);

        beast::get_lowest_layer(*ws_).expires_never();
        ws_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));

        co_await ws_->async_handshake(host_, "/ws-fapi/v1", asio::use_awaitable);
        std::cerr << "[BinanceGateway] connected: /ws-fapi/v1\n";

        connected_ = true;
        if (!queue_.empty()) do_write();

        beast::flat_buffer buffer;
        for (;;) {
            buffer.clear();
            co_await ws_->async_read(buffer, asio::use_awaitable);
            handle_ack(std::string_view(static_cast<const char*>(buffer.data().data()),
                                         buffer.size()));
        }
    }

    void handle_ack(std::string_view json) {
        simdjson::dom::element doc;
        if (auto err = parser_.parse(json).get(doc)) {
            std::cerr << "[BinanceGateway] ack parse error: " << simdjson::error_message(err)
                      << "\n";
            return;
        }

        std::string_view id_str;
        int64_t status = 0;
        // ws-fapi responses echo the request "id" back verbatim (as the
        // same JSON type we sent it: a string, since we sent it quoted).
        if (doc["id"].get(id_str) || doc["status"].get(status)) return;

        uint64_t id = 0;
        auto conv = std::from_chars(id_str.data(), id_str.data() + id_str.size(), id);
        if (conv.ec != std::errc{}) return;

        std::string cid;
        {
            std::lock_guard lk(pending_mtx_);
            auto it = pending_.find(id);
            if (it == pending_.end()) return;  // cancel ack or already handled
            cid = std::move(it->second);
            pending_.erase(it);
        }

        if (status != 200) {
            std::string_view msg_field;
            simdjson::dom::element err_obj;
            if (!doc["error"].get(err_obj)) {
                auto _m_err = err_obj["msg"].get(msg_field);
                (void)_m_err;
            }
            const std::string msg_suffix = msg_field.empty() ? std::string() : ", msg=" + std::string(msg_field);
            std::cerr << "[BinanceGateway] order request " << cid << " rejected (status="
                      << status << msg_suffix << ")\n";
            oms_.apply_update(fnv1a64(cid), OrderStatus::Rejected,
                               /*cumulative_filled_qty=*/0.0, /*last_fill_price=*/0.0,
                               /*exchange_order_id=*/0, now_ns());
        }
        // status == 200: request accepted by the matching engine. We take
        // no further action here — UserDataFeed's ORDER_TRADE_UPDATE is
        // authoritative for everything from here on (Open/Partial/Filled).
    }

    void track_pending(uint64_t id, std::string_view cid) {
        std::lock_guard lk(pending_mtx_);
        pending_.emplace(id, std::string(cid));
    }

    void send_msg(std::string msg) {
        asio::post(exec_, [this, m = std::move(msg)]() mutable {
            queue_.push_back(std::move(m));
            if (connected_ && !writing_) do_write();
        });
    }

    void do_write() {
        if (queue_.empty() || !connected_ || !ws_.has_value()) { writing_ = false; return; }
        writing_ = true;
        ws_->async_write(asio::buffer(queue_.front()),
                          [this](beast::error_code ec, std::size_t) {
                              if (!ec) {
                                  queue_.pop_front();
                                  do_write();
                              } else {
                                  writing_ = false;
                                  connected_ = false;
                                  // The concurrent async_read() in run() will also fail
                                  // shortly (same underlying socket) and unwind run(),
                                  // which run_with_backoff() will catch and retry. We
                                  // flip connected_ here too so no further sends attempt
                                  // do_write() against a socket already known to be dead.
                              }
                          });
    }

    std::string build_order_msg(uint64_t id, std::string_view cid, std::string_view sym,
                                 std::string_view side, std::string_view type, double qty,
                                 double price, std::string_view tif) {
        const auto prec = detail::precision_for(sym);
        const std::string q_str = detail::format_decimal(qty, prec.qty_decimals);
        const int64_t ts = detail::now_ms();

        std::string payload = "apiKey=" + api_key_ + "&newClientOrderId=" + std::string(cid);
        std::string p_str;
        if (price > 0.0) {
            p_str = detail::format_decimal(price, prec.price_decimals);
            payload += "&price=" + p_str;
        }
        payload += "&quantity=" + q_str + "&side=" + std::string(side) + "&symbol=" + std::string(sym);
        if (!tif.empty()) payload += "&timeInForce=" + std::string(tif);
        payload += "&timestamp=" + std::to_string(ts) + "&type=" + std::string(type);
        const std::string sig = detail::hmac_sha256_hex(api_secret_, payload);

        std::string json = R"({"id":")" + std::to_string(id) + R"(","method":"order.place","params":{)";
        json += R"("apiKey":")" + api_key_ + R"(","newClientOrderId":")" + std::string(cid) + R"(",)";
        if (price > 0.0) json += R"("price":")" + p_str + R"(",)";
        json += R"("quantity":")" + q_str + R"(","side":")" + std::string(side) + R"(","symbol":")" +
                std::string(sym) + R"(",)";
        if (!tif.empty()) json += R"("timeInForce":")" + std::string(tif) + R"(",)";
        json += R"("timestamp":)" + std::to_string(ts) + R"(,"type":")" + std::string(type) +
                R"(","signature":")" + sig + R"("}})";
        return json;
    }

    uint64_t next_msg_id() { return msg_id_.fetch_add(1, std::memory_order_relaxed) + 1; }

    asio::any_io_executor exec_;
    asio::ssl::context& ssl_ctx_;
    std::string api_key_, api_secret_;
    OMSCore& oms_;
    std::string host_;

    // The current connection's stream, rebuilt fresh on every reconnect.
    // See the invariant documented on run() above.
    std::optional<websocket::stream<beast::ssl_stream<beast::tcp_stream>>> ws_;
    std::deque<std::string> queue_;
    bool connected_{false}, writing_{false};
    std::atomic<uint64_t> msg_id_{0};

    std::mutex pending_mtx_;
    std::unordered_map<uint64_t, std::string> pending_;  // msg id -> client_order_id
    simdjson::dom::parser parser_;
};

}  // namespace holo::net