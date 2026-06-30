#pragma once

// FeedHandler — owns BinanceWssClient, parses JSON depth frames, pushes
// LobUpdate records into a lock-free ring buffer.
// Single responsibility: ingest → parse → enqueue.

#include <common/lockfree_ring_buffer.hpp>
#include <common/thread_utils.hpp>
#include <common/types.hpp>
#include <data_feed/binance_wss.hpp>
#include <data_feed/lob_core.hpp>

#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <boost/json.hpp>

namespace holo
{

struct alignas(k_cache_line) FeedMetrics
{
    std::atomic<std::uint64_t> msgs_received{0U};
    std::atomic<std::uint64_t> msgs_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<std::uint64_t> reconnects{0U};
    std::array<std::atomic<std::uint64_t>, k_n_instruments> per_instrument{};

    FeedMetrics() noexcept
    { for (auto& a : per_instrument) a.store(0U, std::memory_order_relaxed); }
};

class FeedHandler final
{
public:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    explicit FeedHandler(RingT& ring) noexcept
        : ring_{ring}
        , wss_{[this](std::string_view f) { parse_and_enqueue(f); }}
    {}

    ~FeedHandler() noexcept { stop(); }

    FeedHandler(const FeedHandler&)            = delete;
    FeedHandler& operator=(const FeedHandler&) = delete;
    FeedHandler(FeedHandler&&)                 = delete;
    FeedHandler& operator=(FeedHandler&&)      = delete;

    void start()             { wss_.start(); }
    void stop()  noexcept    { wss_.stop();  }

    [[nodiscard]] const FeedMetrics& metrics() const noexcept { return metrics_; }

private:
    // constexpr char-compare — no hash, no collision risk
    [[nodiscard]] static constexpr std::uint32_t
    instrument_id(std::string_view s) noexcept
    {
        if (s.size() >= 6)
        {
            if (s[0]=='b' && s[1]=='t' && s[2]=='c') return 0U;
            if (s[0]=='e' && s[1]=='t' && s[2]=='h') return 1U;
            if (s[0]=='s' && s[1]=='o' && s[2]=='l') return 2U;
            if (s[0]=='b' && s[1]=='n' && s[2]=='b') return 3U;
        }
        return std::numeric_limits<std::uint32_t>::max();
    }

    // Locale-free price parser — faster than std::stod, zero alloc
    [[nodiscard]] static float parse_price(std::string_view sv) noexcept
    {
        float result = 0.0F;
        bool  dot    = false;
        float frac   = 0.1F;
        for (char c : sv)
        {
            if (c >= '0' && c <= '9')
            {
                if (!dot) result = result * 10.0F + static_cast<float>(c - '0');
                else      { result += static_cast<float>(c - '0') * frac; frac *= 0.1F; }
            }
            else if (c == '.') dot = true;
            else break;
        }
        return result;
    }

    void parse_and_enqueue(std::string_view raw) noexcept
    {
        try
        {
            boost::json::monotonic_resource mr;
            const boost::json::value jv = boost::json::parse(raw, &mr);
            const auto& obj = jv.as_object();

            const auto* sv = obj.if_contains("stream");
            const auto* dv = obj.if_contains("data");
            if (!sv || !dv) return;

            const std::uint32_t id = instrument_id(sv->as_string());
            if (id >= k_n_instruments) return;

            const auto& data = dv->as_object();
            const auto* bv   = data.if_contains("b");
            const auto* av   = data.if_contains("a");
            if (!bv || !av) return;

            metrics_.msgs_received.fetch_add(1U, std::memory_order_relaxed);
            metrics_.per_instrument[id].fetch_add(1U, std::memory_order_relaxed);

            const std::uint64_t ts = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());

            const auto push_levels = [&](const boost::json::array& levels, Side side)
            {
                const std::size_t n = std::min(levels.size(),
                    static_cast<std::size_t>(k_max_depth));
                for (std::size_t lvl = 0U; lvl < n; ++lvl)
                {
                    const auto& entry = levels[lvl].as_array();
                    if (entry.size() < 2U) continue;

                    LobUpdate u{};
                    u.timestamp_ns  = ts;
                    u.instrument_id = id;
                    u.depth_level   = static_cast<std::uint8_t>(lvl);
                    u.side          = side;
                    u.price         = parse_price(entry[0].as_string());
                    u.quantity      = parse_price(entry[1].as_string());

                    if (!ring_.try_push(u)) [[unlikely]]
                        metrics_.msgs_dropped.fetch_add(1U, std::memory_order_relaxed);
                }
            };

            push_levels(bv->as_array(), Side::Bid);
            push_levels(av->as_array(), Side::Ask);
        }
        catch (...)
        {
            metrics_.parse_errors.fetch_add(1U, std::memory_order_relaxed);
        }
    }

    RingT&          ring_;
    BinanceWssClient wss_;
    FeedMetrics     metrics_;
};

} // namespace holo
