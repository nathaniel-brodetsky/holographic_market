#pragma once

// LobSoA — Structure-of-Arrays limit order book.
// Moved from engine/core/ to engine/data_feed/ because it is an ingestion
// artefact: the feed handler writes here; the compute layer reads it.

#include <common/memory_arena.hpp>
#include <common/types.hpp>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace holo
{

struct alignas(k_cache_line) LobStats
{
    std::atomic<std::uint64_t> total_updates{0U};
    std::atomic<std::uint64_t> bid_updates{0U};
    std::atomic<std::uint64_t> ask_updates{0U};
    std::atomic<std::uint64_t> dropped_updates{0U};
    std::byte _pad[k_cache_line - 4U * sizeof(std::atomic<std::uint64_t>)]{};
};
static_assert(sizeof(LobStats) == k_cache_line);

class alignas(k_cache_line) LobSoA final
{
public:
    LobSoA(MemoryArena& arena, std::size_t n_instruments, std::size_t depth)
        : n_instruments_{n_instruments}
        , depth_{depth}
        , total_slots_{n_instruments * depth * k_sides}
        , bid_prices_{arena.alloc_span<float>(n_instruments * depth)}
        , ask_prices_{arena.alloc_span<float>(n_instruments * depth)}
        , bid_qtys_  {arena.alloc_span<float>(n_instruments * depth)}
        , ask_qtys_  {arena.alloc_span<float>(n_instruments * depth)}
        , seq_nos_   {arena.alloc_span<std::uint32_t>(n_instruments * depth * k_sides)}
        , last_ts_ns_{arena.alloc_span<std::uint64_t>(n_instruments)}
        , stats_     {arena.emplace<LobStats>()}
    {
        for (std::size_t i = 0U; i < bid_prices_.size(); ++i)
        {
            bid_prices_[i] = 100.0F;
            ask_prices_[i] = 100.0F;
            bid_qtys_[i]   = 0.0F;
            ask_qtys_[i]   = 0.0F;
        }
        for (auto& s : seq_nos_)    s = k_invalid_seq;
        for (auto& t : last_ts_ns_) t = 0U;
    }

    LobSoA(const LobSoA&)            = delete;
    LobSoA& operator=(const LobSoA&) = delete;
    LobSoA(LobSoA&&)                 = delete;
    LobSoA& operator=(LobSoA&&)      = delete;

    void apply(const LobUpdate& u) noexcept
    {
        if (u.instrument_id >= n_instruments_ ||
            u.depth_level   >= depth_) [[unlikely]] { return; }

        const std::size_t idx = u.instrument_id * depth_ + u.depth_level;
        if (u.side == Side::Bid) { bid_prices_[idx] = u.price; bid_qtys_[idx] = u.quantity; }
        else                     { ask_prices_[idx] = u.price; ask_qtys_[idx] = u.quantity; }

        last_ts_ns_[u.instrument_id] = u.timestamp_ns;
        stats_->total_updates.fetch_add(1U, std::memory_order_relaxed);
    }

    [[nodiscard]] float mid_price(std::size_t instr) const noexcept
    {
        if (instr >= n_instruments_) return 0.0F;
        const std::size_t idx = instr * depth_;
        return (bid_prices_[idx] + ask_prices_[idx]) * 0.5F;
    }

    [[nodiscard]] float spread(std::size_t instr) const noexcept
    {
        if (instr >= n_instruments_) return 0.0F;
        const std::size_t idx = instr * depth_;
        return ask_prices_[idx] - bid_prices_[idx];
    }

    // Raw span accessors for GPU mirror
    [[nodiscard]] std::span<const float> bid_prices() const noexcept { return bid_prices_; }
    [[nodiscard]] std::span<const float> ask_prices() const noexcept { return ask_prices_; }
    [[nodiscard]] std::span<const float> bid_qtys()   const noexcept { return bid_qtys_;   }
    [[nodiscard]] std::span<const float> ask_qtys()   const noexcept { return ask_qtys_;   }

    [[nodiscard]] std::size_t n_instruments() const noexcept { return n_instruments_; }
    [[nodiscard]] std::size_t depth()         const noexcept { return depth_; }
    [[nodiscard]] const LobStats& stats()     const noexcept { return *stats_; }

private:
    const std::size_t n_instruments_;
    const std::size_t depth_;
    const std::size_t total_slots_;

    std::span<float>          bid_prices_;
    std::span<float>          ask_prices_;
    std::span<float>          bid_qtys_;
    std::span<float>          ask_qtys_;
    std::span<std::uint32_t>  seq_nos_;
    std::span<std::uint64_t>  last_ts_ns_;
    LobStats*                 stats_;
};

} // namespace holo
