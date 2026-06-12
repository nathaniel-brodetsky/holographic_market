#pragma once

// net/csv_replay.hpp
// Phase VI: historical LOB CSV replay — drop-in replacement for BinanceFeedHandler.
//
// CSV format (produced by infra/download_binance_data.py):
//   timestamp_ns, instrument_id, bid_price, bid_qty, ask_price, ask_qty
//   (no header, no depth column — bookTicker data, depth_level is always 0)
//
// Usage:
//   DynamicSpscRingBuffer<LobUpdate> ring{65536};
//   CsvReplayHandler replay{ring, "data/test_data.csv"};
//   replay.start();                    // blocks until EOF or stop()
//   auto& m = replay.metrics();

#include <core/lob_core.hpp>
#include <core/lockfree_ring_buffer.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>

namespace holo::net
{

// ── Replay speed control ─────────────────────────────────────────────────────

enum class ReplayMode : std::uint8_t
{
    Fastest,     // no sleep — saturate the ring as fast as possible (backtest throughput)
    Realtime,    // honour original inter-message gaps (latency simulation)
    Throttled,   // emit at a fixed rate (msgs_per_sec configurable)
};

struct ReplayConfig
{
    ReplayMode mode{ReplayMode::Fastest};

    // Throttled mode: target messages per second (ignored for other modes)
    std::uint32_t msgs_per_sec{100'000U};

    // Realtime mode: cap a single sleep to this many ms (avoids multi-minute gaps)
    std::uint32_t max_gap_ms{500U};

    // How many rows to buffer in the reader thread before the consumer catches up.
    // Larger = smoother replay, more RAM. Each LobUpdate is 24 bytes.
    std::size_t read_ahead{65'536U};
};

// ── Metrics ──────────────────────────────────────────────────────────────────

struct alignas(64) ReplayMetrics
{
    std::atomic<std::uint64_t> rows_read{0U};
    std::atomic<std::uint64_t> updates_pushed{0U};  // bid + ask per row = 2
    std::atomic<std::uint64_t> updates_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<bool>          finished{false};
};

// ── Fast locale-free CSV field parsers ───────────────────────────────────────

namespace detail
{

// Parse uint64 from [p, end). Returns pointer past last digit.
[[nodiscard]] inline const char* parse_u64(const char* p, const char* end,
                                            std::uint64_t& out) noexcept
{
    out = 0U;
    while (p < end && *p >= '0' && *p <= '9')
        out = out * 10ULL + static_cast<std::uint64_t>(*p++ - '0');
    return p;
}

// Parse uint32 from [p, end).
[[nodiscard]] inline const char* parse_u32(const char* p, const char* end,
                                            std::uint32_t& out) noexcept
{
    out = 0U;
    while (p < end && *p >= '0' && *p <= '9')
        out = out * 10U + static_cast<std::uint32_t>(*p++ - '0');
    return p;
}

// Parse float (price / qty) from [p, end).
[[nodiscard]] inline const char* parse_float(const char* p, const char* end,
                                              float& out) noexcept
{
    out = 0.0F;
    float frac = 0.1F;
    bool  dot  = false;
    while (p < end)
    {
        const char c = *p;
        if (c >= '0' && c <= '9')
        {
            if (!dot) out = out * 10.0F + static_cast<float>(c - '0');
            else      { out += static_cast<float>(c - '0') * frac; frac *= 0.1F; }
            ++p;
        }
        else if (c == '.' && !dot) { dot = true; ++p; }
        else break;
    }
    return p;
}

// Advance past the next comma (or to end).
[[nodiscard]] inline const char* skip_comma(const char* p, const char* end) noexcept
{
    while (p < end && *p != ',') ++p;
    if (p < end) ++p;  // consume ','
    return p;
}

// Returns true if line is blank or a header (starts with non-digit).
[[nodiscard]] inline bool is_header_or_blank(const char* p, const char* end) noexcept
{
    while (p < end && (*p == ' ' || *p == '\r')) ++p;
    return (p == end || *p == '\n' || (*p != '-' && (*p < '0' || *p > '9')));
}

} // namespace detail

// ── CsvReplayHandler ─────────────────────────────────────────────────────────

class CsvReplayHandler final
{
public:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    explicit CsvReplayHandler(RingT& ring,
                              std::string csv_path,
                              ReplayConfig cfg = {}) noexcept
        : ring_{ring}
        , path_{std::move(csv_path)}
        , cfg_{cfg}
        , shutdown_{false}
    {}

    ~CsvReplayHandler() noexcept { stop(); }

    CsvReplayHandler(const CsvReplayHandler&)            = delete;
    CsvReplayHandler& operator=(const CsvReplayHandler&) = delete;
    CsvReplayHandler(CsvReplayHandler&&)                 = delete;
    CsvReplayHandler& operator=(CsvReplayHandler&&)      = delete;

    // Starts the replay worker thread (non-blocking).
    void start()
    {
        shutdown_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { run(); });
    }

    // Blocks until the worker thread finishes (either EOF or stop()).
    void join()
    {
        if (worker_.joinable()) worker_.join();
    }

    // Signal the worker to stop and wait.
    void stop() noexcept
    {
        shutdown_.store(true, std::memory_order_release);
        if (worker_.joinable()) worker_.join();
    }

    [[nodiscard]] bool finished() const noexcept
    { return metrics_.finished.load(std::memory_order_acquire); }

    [[nodiscard]] const ReplayMetrics& metrics() const noexcept { return metrics_; }

private:
    static constexpr std::size_t k_line_buf = 256U;

    // ── main worker ──────────────────────────────────────────────────────────

    void run() noexcept
    {
        std::FILE* f = std::fopen(path_.c_str(), "r");
        if (!f)
        {
            std::fprintf(stderr, "[CsvReplayHandler] cannot open: %s\n", path_.c_str());
            metrics_.finished.store(true, std::memory_order_release);
            return;
        }

        char line[k_line_buf];
        std::uint64_t prev_ts = 0U;

        // Throttled mode: pre-compute inter-message gap in nanoseconds.
        const std::uint64_t throttle_gap_ns = (cfg_.mode == ReplayMode::Throttled)
            ? (1'000'000'000ULL / std::max(cfg_.msgs_per_sec, 1U))
            : 0ULL;

        while (!shutdown_.load(std::memory_order_acquire)
               && std::fgets(line, k_line_buf, f))
        {
            const char* p   = line;
            const char* end = line + std::strlen(line);

            if (detail::is_header_or_blank(p, end)) continue;

            // ── Parse row ────────────────────────────────────────────────────
            // Fields: timestamp_ns, instrument_id, bid_price, bid_qty, ask_price, ask_qty

            std::uint64_t ts_ns      = 0U;
            std::uint32_t instr_id   = 0U;
            float         bid_price  = 0.0F;
            float         bid_qty    = 0.0F;
            float         ask_price  = 0.0F;
            float         ask_qty    = 0.0F;

            p = detail::parse_u64  (p, end, ts_ns);    p = detail::skip_comma(p, end);
            p = detail::parse_u32  (p, end, instr_id); p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, bid_price); p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, bid_qty);   p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, ask_price); p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, ask_qty);

            if (bid_price == 0.0F || ask_price == 0.0F || instr_id >= k_max_instruments)
            {
                metrics_.parse_errors.fetch_add(1U, std::memory_order_relaxed);
                continue;
            }

            metrics_.rows_read.fetch_add(1U, std::memory_order_relaxed);

            // ── Timing ───────────────────────────────────────────────────────

            if (cfg_.mode == ReplayMode::Realtime && prev_ts != 0U && ts_ns > prev_ts)
            {
                const std::uint64_t gap_ns = ts_ns - prev_ts;
                const std::uint64_t cap_ns =
                    static_cast<std::uint64_t>(cfg_.max_gap_ms) * 1'000'000ULL;
                const std::uint64_t sleep_ns = (gap_ns < cap_ns) ? gap_ns : cap_ns;
                if (sleep_ns > 100'000ULL)  // skip tiny sleeps < 0.1 ms
                {
                    struct timespec req{
                        static_cast<time_t>(sleep_ns / 1'000'000'000ULL),
                        static_cast<long>  (sleep_ns % 1'000'000'000ULL)
                    };
                    nanosleep(&req, nullptr);
                }
            }
            else if (cfg_.mode == ReplayMode::Throttled && throttle_gap_ns > 0U)
            {
                struct timespec req{
                    static_cast<time_t>(throttle_gap_ns / 1'000'000'000ULL),
                    static_cast<long>  (throttle_gap_ns % 1'000'000'000ULL)
                };
                nanosleep(&req, nullptr);
            }

            prev_ts = ts_ns;

            // ── Enqueue Bid update ────────────────────────────────────────────

            {
                LobUpdate bid{};
                bid.timestamp_ns  = ts_ns;
                bid.instrument_id = instr_id;
                bid.depth_level   = 0U;
                bid.side          = Side::Bid;
                bid.price         = bid_price;
                bid.quantity      = bid_qty;

                if (ring_.try_push(bid)) [[likely]]
                    metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
                else
                    metrics_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);
            }

            // ── Enqueue Ask update ────────────────────────────────────────────

            {
                LobUpdate ask{};
                ask.timestamp_ns  = ts_ns;
                ask.instrument_id = instr_id;
                ask.depth_level   = 0U;
                ask.side          = Side::Ask;
                ask.price         = ask_price;
                ask.quantity      = ask_qty;

                if (ring_.try_push(ask)) [[likely]]
                    metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
                else
                    metrics_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);
            }
        }

        std::fclose(f);
        metrics_.finished.store(true, std::memory_order_release);
    }

    // ── Members ──────────────────────────────────────────────────────────────

    RingT&        ring_;
    std::string   path_;
    ReplayConfig  cfg_;

    std::atomic<bool> shutdown_;
    std::thread       worker_;
    ReplayMetrics     metrics_;
};

} // namespace holo::net

namespace holo { using namespace holo::net; }
