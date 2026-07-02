#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <concepts>
#include <limits>
#include <type_traits>
#include <core/memory_arena.hpp>

namespace holo::core
{
    static constexpr std::size_t k_max_instruments = 512U;
    static constexpr std::size_t k_max_depth       = 16U;
    static constexpr std::size_t k_sides           = 2U;
    static constexpr std::uint32_t k_invalid_seq   =
        std::numeric_limits<std::uint32_t>::max();

    enum class Side : std::uint8_t
    {
        Bid = 0U,
        Ask = 1U
    };

    struct alignas(8) LobUpdate
    {
        std::uint64_t  timestamp_ns;
        float          price;
        float          quantity;
        std::uint32_t  instrument_id;
        std::uint8_t   depth_level;
        Side           side;
        std::uint8_t   _pad[2];
    };

    static_assert(sizeof(LobUpdate)  == 24U);
    static_assert(alignof(LobUpdate) == 8U);
    static_assert(std::is_trivially_copyable_v<LobUpdate>);

    struct alignas(k_cache_line) LobStats
    {
        std::atomic<std::uint64_t> total_updates{0U};
        std::atomic<std::uint64_t> bid_updates{0U};
        std::atomic<std::uint64_t> ask_updates{0U};
        std::atomic<std::uint64_t> dropped_updates{0U};
        std::byte _pad[k_cache_line - 4U * sizeof(std::atomic<std::uint64_t>)];
    };

    static_assert(sizeof(LobStats) == k_cache_line);

    class alignas(k_cache_line) LobSoA final
    {
    public:
        LobSoA(
            MemoryArena  &arena,
            std::size_t   n_instruments,
            std::size_t   depth)
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
            for (std::size_t i = 0U; i < seq_nos_.size();   ++i) seq_nos_[i]    = k_invalid_seq;
            for (std::size_t i = 0U; i < last_ts_ns_.size();++i) last_ts_ns_[i] = 0U;
        }

        LobSoA(const LobSoA &) = delete;
        LobSoA &operator=(const LobSoA &) = delete;
        LobSoA(LobSoA &&) = delete;
        LobSoA &operator=(LobSoA &&) = delete;

        void apply(const LobUpdate &u) noexcept
        {
            const std::size_t instr = static_cast<std::size_t>(u.instrument_id);
            const std::size_t depth = static_cast<std::size_t>(u.depth_level);

            if (instr >= n_instruments_) [[unlikely]]
            { stats_->dropped_updates.fetch_add(1U, std::memory_order_relaxed); return; }
            if (depth >= depth_) [[unlikely]]
            { stats_->dropped_updates.fetch_add(1U, std::memory_order_relaxed); return; }

            const std::size_t base = instr * depth_ + depth;

            if (u.side == Side::Bid)
            {
                bid_prices_[base] = u.price;
                bid_qtys_[base]   = u.quantity;
                stats_->bid_updates.fetch_add(1U, std::memory_order_relaxed);
            }
            else
            {
                ask_prices_[base] = u.price;
                ask_qtys_[base]   = u.quantity;
                stats_->ask_updates.fetch_add(1U, std::memory_order_relaxed);
            }

            const std::size_t seq_base =
                instr * depth_ * k_sides
                + depth * k_sides
                + static_cast<std::size_t>(u.side);

            seq_nos_[seq_base] = static_cast<std::uint32_t>(
                stats_->total_updates.load(std::memory_order_relaxed));
            last_ts_ns_[instr] = u.timestamp_ns;
            stats_->total_updates.fetch_add(1U, std::memory_order_relaxed);
        }

        // ── Level-0 accessors (best bid / best ask) ──────────────────────

        // Best bid price for instrument (level 0).
        [[nodiscard]] float best_bid(std::size_t instrument_id) const noexcept
        {
            return bid_prices_[instrument_id * depth_];
        }

        // Best ask price for instrument (level 0).
        [[nodiscard]] float best_ask(std::size_t instrument_id) const noexcept
        {
            return ask_prices_[instrument_id * depth_];
        }

        // Mid-price = (best_bid + best_ask) / 2.
        [[nodiscard]] float mid_price(std::size_t instrument_id) const noexcept
        {
            const std::size_t base = instrument_id * depth_;
            return (bid_prices_[base] + ask_prices_[base]) * 0.5F;
        }

        // Raw bid-ask spread in price units (ask_L0 - bid_L0).
        // Returns a small positive fallback if LOB not yet populated.
        [[nodiscard]] float spread(std::size_t instrument_id) const noexcept
        {
            const std::size_t base = instrument_id * depth_;
            const float s = ask_prices_[base] - bid_prices_[base];
            // Guard: if LOB still at default (bid==ask==100), spread == 0.
            // Return a 2 bps fallback so PnL model stays well-defined.
            return (s > 0.0F) ? s : mid_price(instrument_id) * 2e-4F;
        }

        // ── Full depth spans ──────────────────────────────────────────────

        [[nodiscard]] std::span<const float>
        bid_prices(std::size_t instrument_id) const noexcept
        { return bid_prices_.subspan(instrument_id * depth_, depth_); }

        [[nodiscard]] std::span<const float>
        ask_prices(std::size_t instrument_id) const noexcept
        { return ask_prices_.subspan(instrument_id * depth_, depth_); }

        [[nodiscard]] std::span<const float>
        bid_qtys(std::size_t instrument_id) const noexcept
        { return bid_qtys_.subspan(instrument_id * depth_, depth_); }

        [[nodiscard]] std::span<const float>
        ask_qtys(std::size_t instrument_id) const noexcept
        { return ask_qtys_.subspan(instrument_id * depth_, depth_); }

        // ── Raw array views (for GPU upload) ─────────────────────────────

        [[nodiscard]] std::span<const float> raw_bid_price_array() const noexcept { return bid_prices_; }
        [[nodiscard]] std::span<const float> raw_ask_price_array() const noexcept { return ask_prices_; }
        [[nodiscard]] std::span<const float> raw_bid_qty_array()   const noexcept { return bid_qtys_;   }
        [[nodiscard]] std::span<const float> raw_ask_qty_array()   const noexcept { return ask_qtys_;   }

        // ── Metadata ─────────────────────────────────────────────────────

        [[nodiscard]] const LobStats  &stats()        const noexcept { return *stats_;        }
        [[nodiscard]] std::size_t      n_instruments() const noexcept { return n_instruments_; }
        [[nodiscard]] std::size_t      depth()         const noexcept { return depth_;         }
        [[nodiscard]] std::size_t      total_slots()   const noexcept { return total_slots_;   }

    private:
        const std::size_t n_instruments_;
        const std::size_t depth_;
        const std::size_t total_slots_;

        alignas(k_cache_line) std::span<float>          bid_prices_;
        alignas(k_cache_line) std::span<float>          ask_prices_;
        alignas(k_cache_line) std::span<float>          bid_qtys_;
        alignas(k_cache_line) std::span<float>          ask_qtys_;
        alignas(k_cache_line) std::span<std::uint32_t>  seq_nos_;
        alignas(k_cache_line) std::span<std::uint64_t>  last_ts_ns_;

        LobStats *stats_;
    };

} // namespace holo::core

namespace holo { using namespace holo::core; }