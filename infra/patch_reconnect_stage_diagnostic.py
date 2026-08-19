#!/usr/bin/env python3
"""
Diagnostic-only patch: adds a log line before/after each connection stage
in UserDataFeed::run() (resolve, TCP connect, TLS handshake, WS handshake)
so the NEXT reconnect failure shows exactly which co_await is timing out,
instead of only the generic "closed due to a timeout" message that could
originate from any of several stages.

This is meant to be temporary -- once the real stage is identified, revert
via the .bak6 backup this patch creates, and apply a targeted fix instead
of leaving this verbose logging in permanently.

Run this on the instance, not in the sandbox.
Usage: python3 patch_reconnect_stage_diagnostic.py
"""
import sys
from pathlib import Path

TARGET = Path.home() / "holographic_market" / "engine" / "net" / "user_data_feed.hpp"

OLD = '''        auto const results = co_await resolver.async_resolve(host_, "443", asio::use_awaitable);

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
        std::cerr << "[UserDataFeed] connected: " << target << "\\n";'''

NEW = '''        std::cerr << "[UserDataFeed][diag] resolving " << host_ << "...\\n";
        auto const results = co_await resolver.async_resolve(host_, "443", asio::use_awaitable);
        std::cerr << "[UserDataFeed][diag] resolved, connecting...\\n";

        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        co_await beast::get_lowest_layer(ws).async_connect(results, asio::use_awaitable);
        std::cerr << "[UserDataFeed][diag] TCP connected, starting TLS handshake...\\n";

        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host_.c_str())) {
            throw beast::system_error(beast::error_code(
                static_cast<int>(::ERR_get_error()), asio::error::get_ssl_category()));
        }

        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        co_await ws.next_layer().async_handshake(asio::ssl::stream_base::client,
                                                   asio::use_awaitable);
        std::cerr << "[UserDataFeed][diag] TLS handshake done, starting WS handshake "
                     "(listen_key_ len=" << listen_key_.size() << ")...\\n";

        beast::get_lowest_layer(ws).expires_never();
        ws.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws.set_option(websocket::stream_base::decorator([](websocket::request_type& req) {
            req.set(beast::http::field::user_agent, "holo-oms/1.0");
        }));

        const std::string target = "/ws/" + listen_key_;
        co_await ws.async_handshake(host_, target, asio::use_awaitable);
        std::cerr << "[UserDataFeed] connected: " << target << "\\n";'''


def main():
    if not TARGET.exists():
        sys.exit(f"FATAL: {TARGET} not found.")

    src = TARGET.read_text()
    count = src.count(OLD)
    if count == 0:
        sys.exit("FATAL: exact old_str not found. File has drifted from what this patch "
                  "expects -- paste the current relevant section instead of proceeding blindly.")
    if count > 1:
        sys.exit(f"FATAL: old_str matched {count} times, expected exactly 1. "
                  f"Aborting with no changes made.")

    backup = TARGET.with_suffix(TARGET.suffix + ".bak6")
    backup.write_text(src)
    TARGET.write_text(src.replace(OLD, NEW))
    print("edit applied OK")
    print(f"Backup of pre-patch file saved to {backup}")
    print(f"Patched: {TARGET}")


if __name__ == "__main__":
    main()
