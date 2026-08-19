#!/usr/bin/env python3
"""
Fetches a brand-new listenKey at the start of every UserDataFeed::run()
attempt (initial connect AND every reconnect), instead of reusing
whatever key was last set.

Diagnostic evidence (stage-by-stage logging added in a prior patch on
this branch): resolve, TCP connect, and TLS handshake all succeed
quickly and consistently on every reconnect attempt. The failure is
specifically the WS protocol upgrade handshake (ws.async_handshake) --
it never gets a 101 response, timing out after handshake_timeout (30s)
-- but only on RECONNECTS. The very first connection at process startup,
using a key fetched fresh moments earlier, succeeds immediately. This
points at unreliable behavior reusing the same listenKey across
reconnects against this venue (at least on testnet), rather than a bug
in the connection-setup code itself (which was already verified correct
against Boost.Beast's documented idle_timeout=none() semantics for the
client role).

Fix: fetch a fresh listenKey via the same blocking-socket REST pattern
main_live.cpp already uses for this exact call, right before every
connection attempt -- sidesteps the ambiguity of whether this is genuine
expiry or a same-key-reconnect quirk, at the cost of one extra REST
round trip per (re)connect (connections are infrequent -- this is not a
hot path).

Run this on the instance, not in the sandbox.
Usage: python3 patch_fresh_listen_key_per_attempt.py
"""
import sys
from pathlib import Path

TARGET = Path.home() / "holographic_market" / "engine" / "net" / "user_data_feed.hpp"

EDITS = [
    (
        '''    asio::awaitable<void> run() {
        // A fresh stream per connection attempt — beast websocket streams
        // are not meant to be reused/re-handshaken after a failed or
        // closed connection.
        websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(exec_, ssl_ctx_);
        tcp::resolver resolver(exec_);''',
        '''    asio::awaitable<void> run() {
        // Always fetch a brand-new listenKey before every connection
        // attempt, including reconnects, rather than reusing whatever
        // key was last set. Diagnostic testing (stage-by-stage logging)
        // showed the WS handshake step reliably timing out -- never
        // receiving a 101 response -- specifically on reconnects reusing
        // the SAME key that worked fine for the initial connection.
        // A fresh key sidesteps the ambiguity of genuine expiry vs. some
        // other same-key-reconnect quirk against this venue.
        try {
            listen_key_ = fetch_fresh_listen_key();
            std::cerr << "[UserDataFeed][diag] fetched fresh listenKey (len="
                      << listen_key_.size() << ")\\n";
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed] failed to fetch a fresh listenKey: " << e.what()
                      << "\\n";
            throw;  // let the backoff loop retry
        }

        // A fresh stream per connection attempt — beast websocket streams
        // are not meant to be reused/re-handshaken after a failed or
        // closed connection.
        websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(exec_, ssl_ctx_);
        tcp::resolver resolver(exec_);'''
    ),
    (
        '''    // Blocking signed REST GET -- mirrors main_live.cpp's
    // fetch_listen_key()/renew_listen_key() style''',
        '''    // Blocking REST POST for a brand-new listenKey -- mirrors
    // main_live.cpp's fetch_listen_key() exactly (same endpoint, same
    // blocking-socket style), duplicated here rather than shared because
    // main_live.cpp's copy is a free function with no access to this
    // class's members, and introducing a shared REST-utility header for
    // one function isn't worth it yet.
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
            return body.substr(pos, body.find("\\"", pos) - pos);
        }
        throw std::runtime_error("fetch_fresh_listen_key: no listenKey in response: " + body);
    }

    // Blocking signed REST GET -- mirrors main_live.cpp's
    // fetch_listen_key()/renew_listen_key() style'''
    ),
]


def main():
    if not TARGET.exists():
        sys.exit(f"FATAL: {TARGET} not found.")

    src = TARGET.read_text()
    original = src
    for i, (old, new) in enumerate(EDITS, 1):
        count = src.count(old)
        if count == 0:
            sys.exit(f"FATAL: edit {i}/{len(EDITS)} -- exact old_str not found. File has "
                      f"drifted -- paste the current relevant section instead of proceeding "
                      f"blindly.")
        if count > 1:
            sys.exit(f"FATAL: edit {i}/{len(EDITS)} -- old_str matched {count} times, "
                      f"expected exactly 1. Aborting with no changes made.")
        src = src.replace(old, new)
        print(f"edit {i}/{len(EDITS)} applied OK")

    backup = TARGET.with_suffix(TARGET.suffix + ".bak7")
    backup.write_text(original)
    TARGET.write_text(src)
    print(f"\\nBackup of pre-patch file saved to {backup}")
    print(f"Patched: {TARGET}")


if __name__ == "__main__":
    main()
