#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace holo
{

// ── Cache / alignment constants ──────────────────────────────────────────────
static constexpr std::size_t k_cache_line = 64U;
static constexpr std::size_t k_page_size  = 4096U;

// ── Instrument catalogue ──────────────────────────────────────────────────────
static constexpr std::size_t k_n_instruments = 4U;

static constexpr std::array<std::string_view, k_n_instruments> k_symbols = {
    "btcusdt", "ethusdt", "solusdt", "bnbusdt"
};
static constexpr std::array<std::string_view, k_n_instruments> k_symbols_upper = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"
};

// ── LOB primitives ────────────────────────────────────────────────────────────
static constexpr std::size_t   k_max_instruments = 512U;
static constexpr std::size_t   k_max_depth       = 16U;
static constexpr std::size_t   k_sides           = 2U;
static constexpr std::uint32_t k_invalid_seq =
    std::numeric_limits<std::uint32_t>::max();

enum class Side : std::uint8_t
{
    Bid = 0U,
    Ask = 1U
};

struct alignas(8) LobUpdate
{
    std::uint64_t timestamp_ns;
    float         price;
    float         quantity;
    std::uint32_t instrument_id;
    std::uint8_t  depth_level;
    Side          side;
    std::uint8_t  _pad[2];
};
static_assert(sizeof(LobUpdate)  == 24U);
static_assert(alignof(LobUpdate) == 8U);
static_assert(std::is_trivially_copyable_v<LobUpdate>);

// ── Routing primitives (produced by trading layer, consumed by execution) ─────
struct alignas(32) RoutedEdge
{
    std::uint32_t src_instrument{0U};
    std::uint32_t dst_instrument{0U};
    float         harmonic_flow{0.0F};
    float         yang_mills_action{0.0F};
    std::uint64_t signal_ts_ns{0U};
    std::uint32_t edge_index{0U};
    std::uint32_t _pad{0U};
};
static_assert(sizeof(RoutedEdge) == 32U);

} // namespace holo
