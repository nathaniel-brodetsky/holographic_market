#!/usr/bin/env python3
"""
Fixes UserDataFeed to use the correct WebSocket host for Binance Futures
Testnet, separate from the REST host.

Binance Futures Testnet serves REST (order placement, listenKey
management, etc.) on testnet.binancefuture.com, but WebSocket streams
(market data AND user-data streams) on a DIFFERENT subdomain:
stream.binancefuture.com. UserDataFeed was constructed with a single
`host_` used for BOTH purposes -- REST calls (correct) and the actual
WS connection itself (wrong). testnet.binancefuture.com happily accepts
TCP+TLS connections (it's a normal HTTPS server), which is why those
steps always succeeded in every diagnostic test, but it doesn't serve a
WS upgrade at /ws/<listenKey> -- so the WS handshake step's own timeout
(handshake_timeout=30s) fired every single time, consistent from the
very first connection attempt, completely unaffected by listenKey
freshness, an explicit DELETE, or backoff timing (all of which were
tested and ruled out first).

Fix: add a distinct ws_host_ member, used only for the WS connection
(resolve/connect/TLS SNI/WS handshake Host); host_ remains used for all
REST calls (listenKey POST/PUT/DELETE, order reconciliation GETs) since
that host is correct as-is.

Run this on the instance, not in the sandbox.
Usage: python3 patch_ws_host_split.py
"""
import sys
from pathlib import Path

ENGINE = Path.home() / "holographic_market" / "engine"
UDF = ENGINE / "net" / "user_data_feed.hpp"
MAIN_LIVE = ENGINE / "app" / "main_live.cpp"

EDITS = [
    (
        UDF,
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
        , oms_(oms) {}''',
        '''    // `host` is the REST host (testnet.binancefuture.com) -- used for
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
        , oms_(oms) {}'''
    ),
    (
        UDF,
        '''        std::cerr << "[UserDataFeed][diag] resolving " << host_ << "...\\n";
        auto const results = co_await resolver.async_resolve(host_, "443", asio::use_awaitable);''',
        '''        std::cerr << "[UserDataFeed][diag] resolving " << ws_host_ << "...\\n";
        auto const results = co_await resolver.async_resolve(ws_host_, "443", asio::use_awaitable);'''
    ),
    (
        UDF,
        '''        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }''',
        '''        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), ws_host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }'''
    ),
    (
        UDF,
        '''        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(host_, target, asio::use_awaitable);''',
        '''        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(ws_host_, target, asio::use_awaitable);'''
    ),
    (
        UDF,
        '''    asio::any_io_executor exec_;
    asio::ssl::context& ssl_ctx_;
    std::string host_;
    std::string listen_key_;''',
        '''    asio::any_io_executor exec_;
    asio::ssl::context& ssl_ctx_;
    std::string host_;      // REST host (testnet.binancefuture.com)
    std::string ws_host_;   // WebSocket host (stream.binancefuture.com) -- see constructor comment
    std::string listen_key_;'''
    ),
    (
        MAIN_LIVE,
        '''    UserDataFeed ud_feed(ioc.get_executor(), ssl_ctx, "testnet.binancefuture.com", listen_key,
                         api_key, api_secret, oms);''',
        '''    UserDataFeed ud_feed(ioc.get_executor(), ssl_ctx, "testnet.binancefuture.com",
                         "stream.binancefuture.com", listen_key, api_key, api_secret, oms);'''
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
        backup = path.with_suffix(path.suffix + ".bak9")
        backup.write_text(path.read_text())
        path.write_text(src)
        print(f"patched {path} (backup: {backup})")


if __name__ == "__main__":
    main()
