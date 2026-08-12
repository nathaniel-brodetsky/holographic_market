#pragma once
//
// symbols.hpp
// holo::net — the traded instrument list, as a single source of truth.
//
// Deliberately dependency-free (no Boost.Asio/Beast/OpenSSL) so it can be
// included from offline/CPU-only code (e.g. main_backtest.cpp) without
// pulling in the live networking stack. Previously k_feed_n_instruments
// and k_symbols lived only in binance_feed.hpp, and main_backtest.cpp
// re-hardcoded the count as a bare "4U" (and "16" for a 4x4 matrix) in
// several unrelated places (LobSoA/SignalRouter construction, PnlTracker's
// EWMA baseline matrix, a bounds check) — adding a 5th instrument meant
// finding and updating every one of those literals by hand, and missing
// one would silently drop or out-of-bounds signals for the new instrument
// rather than fail loudly. See holographic_market_AUDIT.md §10.5.
//
#include <array>
#include <cstddef>
#include <string_view>

namespace holo::net {

    static constexpr std::size_t k_feed_n_instruments = 4U;

    static constexpr std::array<std::string_view, k_feed_n_instruments> k_symbols = {
        "btcusdt", "ethusdt", "solusdt", "bnbusdt"
    };

    static constexpr std::array<std::string_view, k_feed_n_instruments> k_symbols_upper = {
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"
    };

}  // namespace holo::net