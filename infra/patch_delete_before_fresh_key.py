#!/usr/bin/env python3
"""
Adds an explicit DELETE /fapi/v1/listenKey call before fetching a fresh
key on every UserDataFeed connection attempt.

Why: POST /fapi/v1/listenKey does NOT always create a new key -- per
Binance's own docs, "Doing a POST on an account with an active listenKey
will return the currently active listenKey and extend its validity" --
so the previous "fetch fresh key" patch likely kept getting the SAME key
back every time (the account already had an active one from process
start), which is why it didn't change the observed failure at all.

Even a brand new PROCESS's very first connection attempt has now been
observed failing identically -- pointing at server-side state (a
listenKey/session Binance still considers "active" from an earlier
connection that never cleanly closed) rather than anything client-side.
DELETE /fapi/v1/listenKey explicitly closes and invalidates whatever
session is currently associated with the account, before a fresh POST
creates a genuinely new one.

Run this on the instance, not in the sandbox.
Usage: python3 patch_delete_before_fresh_key.py
"""
import sys
from pathlib import Path

TARGET = Path.home() / "holographic_market" / "engine" / "net" / "user_data_feed.hpp"

EDITS = [
    (
        '''        try {
            listen_key_ = fetch_fresh_listen_key();
            std::cerr << "[UserDataFeed][diag] fetched fresh listenKey (len="
                      << listen_key_.size() << ")\\n";
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed] failed to fetch a fresh listenKey: " << e.what()
                      << "\\n";
            throw;  // let the backoff loop retry
        }''',
        '''        // Explicitly close/invalidate whatever session Binance currently
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
                      << listen_key_.size() << ")\\n";
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed] failed to fetch a fresh listenKey: " << e.what()
                      << "\\n";
            throw;  // let the backoff loop retry
        }'''
    ),
    (
        '''    std::string fetch_fresh_listen_key() {''',
        '''    // Best-effort: closes/invalidates whatever listenKey session Binance
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
                      << "\\n";
        } catch (const std::exception& e) {
            std::cerr << "[UserDataFeed][diag] DELETE listenKey failed (likely harmless "
                         "if no key was active): " << e.what() << "\\n";
        }
    }

    std::string fetch_fresh_listen_key() {'''
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

    backup = TARGET.with_suffix(TARGET.suffix + ".bak8")
    backup.write_text(original)
    TARGET.write_text(src)
    print(f"\\nBackup of pre-patch file saved to {backup}")
    print(f"Patched: {TARGET}")


if __name__ == "__main__":
    main()
