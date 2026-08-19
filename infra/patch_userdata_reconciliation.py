#!/usr/bin/env python3
"""
Adds REST-based order-state reconciliation to UserDataFeed.

Why: Binance's user-data WebSocket does not replay/queue missed events on
reconnect -- an ORDER_TRADE_UPDATE that would have arrived exactly during
a disconnect window is lost permanently. Until now, UserDataFeed::run()
reconnected and just kept listening for NEW events, silently trusting
that OMSCore's locally-tracked state was still accurate. If an order
filled on Binance's side during a disconnect, OMSCore would keep
believing it was still open -- the same class of bug as the earlier
unhedged-BTCUSDT incident, just via a different mechanism (a missed WS
message instead of a missing hedge_remaining() call).

Fix: after every successful (re)connect, before entering the read loop,
reconcile against the exchange's authoritative state:
  1. GET /fapi/v1/openOrders -- what Binance currently considers open.
  2. Diff against oms_.get_stale_orders(0, ...) -- every order OMSCore
     currently considers live, regardless of age (reusing the existing
     liveness filter with a zero age threshold rather than duplicating
     it).
  3. For anything locally live but absent from Binance's open-orders
     list, it reached a terminal state while disconnected -- fetch its
     authoritative final state via GET /fapi/v1/order and apply it via
     the same oms_.apply_update() path a real WS message would have
     used.

This requires api_key/api_secret in UserDataFeed's constructor (it only
had the listenKey before, since it never needed to sign a REST request).
The call site in main_live.cpp is updated to pass them.

Run this on the instance, not in the sandbox.
Usage: python3 patch_userdata_reconciliation.py
"""
import sys
from pathlib import Path

ENGINE = Path.home() / "holographic_market" / "engine"
UDF = ENGINE / "net" / "user_data_feed.hpp"
MAIN_LIVE = ENGINE / "app" / "main_live.cpp"

EDITS = [
    # 1. Includes needed for HMAC signing and blocking REST calls.
    (
        UDF,
        '''#include <simdjson.h>

#include <algorithm>
#include <charconv>
#include <functional>
#include <iostream>
#include <string>''',
        '''#include <simdjson.h>

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
#include <vector>'''
    ),
    # 2. Constructor: add api_key/api_secret (needed to sign REST calls --
    #    the WS stream itself never needed them, only the listenKey did).
    (
        UDF,
        '''    UserDataFeed(asio::any_io_executor exec,
                 asio::ssl::context& ssl_ctx,
                 std::string host,
                 std::string listen_key,
                 OMSCore& oms)
        : exec_(exec)
        , ssl_ctx_(ssl_ctx)
        , host_(std::move(host))
        , listen_key_(std::move(listen_key))
        , oms_(oms) {}''',
        '''    UserDataFeed(asio::any_io_executor exec,
                 asio::ssl::context& ssl_ctx,
                 std::string host,
                 std::string listen_key,
                 std::string api_key,
                 std::string api_secret,
                 OMSCore& oms)
        : exec_(exec)
        , ssl_ctx_(ssl_ctx)
        , host_(std::move(host))
        , listen_key_(std::move(listen_key))
        , api_key_(std::move(api_key))
        , api_secret_(std::move(api_secret))
        , oms_(oms) {}'''
    ),
    # 3. Call reconcile_open_orders() right after a successful handshake,
    #    before entering the read loop -- covers both the very first
    #    connect (harmless extra safety check) and every reconnect
    #    (where it actually matters).
    (
        UDF,
        '''        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(host_, target, asio::use_awaitable);
        std::cerr << "[UserDataFeed] connected: " << target << "\\n";

        beast::flat_buffer buffer;''',
        '''        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(host_, target, asio::use_awaitable);
        std::cerr << "[UserDataFeed] connected: " << target << "\\n";

        // Binance does not replay missed events on reconnect -- anything
        // that happened during the gap is gone unless we go fetch it
        // ourselves. See the file-level comment above reconcile_open_orders().
        reconcile_open_orders();

        beast::flat_buffer buffer;'''
    ),
    # 4. Add the reconciliation methods + REST/HMAC helpers, right before
    #    the private member variables at the end of the class.
    (
        UDF,
        '''    asio::any_io_executor exec_;
    asio::ssl::context& ssl_ctx_;
    std::string host_;
    std::string listen_key_;
    OMSCore& oms_;
    simdjson::dom::parser parser_;
    std::function<void(std::string_view)> on_margin_call_;
};''',
        '''    // ------------------------------------------------------------------
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
                      << " -- skipping reconciliation this cycle, will retry on next reconnect\\n";
            return;
        }

        simdjson::dom::parser p;
        simdjson::dom::element doc;
        if (auto err = p.parse(body).get(doc)) {
            std::cerr << "[UserDataFeed] reconcile: openOrders parse error: "
                      << simdjson::error_message(err) << " -- raw body: " << body << "\\n";
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
                         "authoritative final state\\n";
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
                      << client_order_id << ": " << e.what() << "\\n";
            return;
        }

        simdjson::dom::parser p;
        simdjson::dom::element doc;
        if (auto err = p.parse(body).get(doc)) {
            std::cerr << "[UserDataFeed] reconcile: order lookup parse error for "
                      << client_order_id << ": " << simdjson::error_message(err) << "\\n";
            return;
        }

        std::string_view status_str, z_str, avg_px_str;
        int64_t exch_order_id = 0;
        if (doc["status"].get(status_str) || doc["executedQty"].get(z_str) ||
            doc["orderId"].get(exch_order_id)) {
            std::cerr << "[UserDataFeed] reconcile: malformed order lookup response for "
                      << client_order_id << ": " << body << "\\n";
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
                      << client_order_id << " (already released?)\\n";
        } else {
            std::cerr << "[UserDataFeed] reconcile: recovered " << client_order_id
                      << " -> status=" << static_cast<int>(status)
                      << " filled=" << cum_filled << "\\n";
        }
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
    std::string host_;
    std::string listen_key_;
    std::string api_key_;
    std::string api_secret_;
    OMSCore& oms_;
    simdjson::dom::parser parser_;
    std::function<void(std::string_view)> on_margin_call_;
};'''
    ),
    # 5. Update the call site in main_live.cpp to pass api_key/api_secret.
    (
        MAIN_LIVE,
        '''    UserDataFeed ud_feed(ioc.get_executor(), ssl_ctx, "testnet.binancefuture.com", listen_key, oms);''',
        '''    UserDataFeed ud_feed(ioc.get_executor(), ssl_ctx, "testnet.binancefuture.com", listen_key,
                         api_key, api_secret, oms);'''
    ),
]


def main():
    for path, _, _ in EDITS:
        if not path.exists():
            sys.exit(f"FATAL: {path} not found.")

    changed = {}
    for i, (path, old, new) in enumerate(EDITS, 1):
        src = changed.get(path, path.read_text())
        count = src.count(old)
        if count == 0:
            sys.exit(f"FATAL: edit {i}/{len(EDITS)} on {path.name} -- exact old_str not "
                      f"found. File has drifted -- paste the current relevant section "
                      f"instead of proceeding blindly.")
        if count > 1:
            sys.exit(f"FATAL: edit {i}/{len(EDITS)} on {path.name} -- old_str matched "
                      f"{count} times, expected exactly 1. Aborting with no changes made.")
        changed[path] = src.replace(old, new)
        print(f"edit {i}/{len(EDITS)} applied OK ({path.name})")

    for path, src in changed.items():
        backup = path.with_suffix(path.suffix + ".bak5")
        backup.write_text(path.read_text())
        path.write_text(src)
        print(f"patched {path} (backup: {backup})")


if __name__ == "__main__":
    main()
