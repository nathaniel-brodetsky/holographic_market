#pragma once

#include <data_feed/lob_core.hpp>
#include <common/lockfree_ring_buffer.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace holo
{

enum class ReplayMode : std::uint8_t { Fastest, Realtime, Throttled };

struct ReplayConfig
{
    ReplayMode    mode{ReplayMode::Fastest};
    std::uint32_t msgs_per_sec{100'000U};
    std::uint32_t max_gap_ms{500U};
};

struct alignas(64) ReplayMetrics
{
    std::atomic<std::uint64_t> rows_read{0U};
    std::atomic<std::uint64_t> updates_pushed{0U};
    std::atomic<std::uint64_t> updates_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<bool>          finished{false};
};

namespace detail
{

[[nodiscard]] inline const char* parse_u64(const char* p, const char* end,
                                            std::uint64_t& out) noexcept
{
    out = 0U;
    while (p < end && static_cast<unsigned>(*p - '0') < 10U)
        out = out * 10ULL + static_cast<std::uint64_t>(*p++ - '0');
    return p;
}

[[nodiscard]] inline const char* parse_u32(const char* p, const char* end,
                                            std::uint32_t& out) noexcept
{
    out = 0U;
    while (p < end && static_cast<unsigned>(*p - '0') < 10U)
        out = out * 10U + static_cast<std::uint32_t>(*p++ - '0');
    return p;
}

[[nodiscard]] inline const char* parse_float(const char* p, const char* end,
                                              float& out) noexcept
{
    out = 0.0F;
    while (p < end && static_cast<unsigned>(*p - '0') < 10U)
        out = out * 10.0F + static_cast<float>(*p++ - '0');
    if (p < end && *p == '.')
    {
        ++p;
        float frac = 0.1F;
        while (p < end && static_cast<unsigned>(*p - '0') < 10U)
        { out += static_cast<float>(*p++ - '0') * frac; frac *= 0.1F; }
    }
    return p;
}

[[nodiscard]] inline const char* skip_comma(const char* p, const char* end) noexcept
{
    while (p < end && *p != ',') ++p;
    return (p < end) ? p + 1 : p;
}

[[nodiscard]] inline const char* skip_line(const char* p, const char* end) noexcept
{
    while (p < end && *p != '\n') ++p;
    return (p < end) ? p + 1 : p;
}

[[nodiscard]] inline bool is_header(const char* p, const char* end) noexcept
{
    while (p < end && (*p == ' ' || *p == '\r')) ++p;
    return p == end || *p == '\n' || (*p != '-' && (static_cast<unsigned>(*p - '0') >= 10U));
}

} // namespace detail

class CsvReplayHandler final
{
public:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    explicit CsvReplayHandler(RingT& ring,
                              std::string csv_path,
                              ReplayConfig cfg = {}) noexcept
        : ring_{ring}, path_{std::move(csv_path)}, cfg_{cfg}
    {}

    ~CsvReplayHandler() noexcept { stop(); }

    CsvReplayHandler(const CsvReplayHandler&)            = delete;
    CsvReplayHandler& operator=(const CsvReplayHandler&) = delete;
    CsvReplayHandler(CsvReplayHandler&&)                 = delete;
    CsvReplayHandler& operator=(CsvReplayHandler&&)      = delete;

    void start()
    {
        shutdown_.store(false, std::memory_order_release);
        worker_ = std::thread([this]{ run(); });
    }

    void join()  { if (worker_.joinable()) worker_.join(); }

    void stop() noexcept
    {
        shutdown_.store(true, std::memory_order_release);
        if (worker_.joinable()) worker_.join();
    }

    [[nodiscard]] bool finished() const noexcept
    { return metrics_.finished.load(std::memory_order_acquire); }

    [[nodiscard]] const ReplayMetrics& metrics() const noexcept { return metrics_; }

private:
    void run() noexcept
    {
        const int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::fprintf(stderr, "[CsvReplay] cannot open: %s\n", path_.c_str());
            metrics_.finished.store(true, std::memory_order_release);
            return;
        }

        struct stat sb{};
        if (::fstat(fd, &sb) < 0 || sb.st_size == 0)
        {
            ::close(fd);
            metrics_.finished.store(true, std::memory_order_release);
            return;
        }

        const std::size_t file_size = static_cast<std::size_t>(sb.st_size);
        void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);

        if (mapped == MAP_FAILED)
        {
            std::fprintf(stderr, "[CsvReplay] mmap failed\n");
            metrics_.finished.store(true, std::memory_order_release);
            return;
        }

        ::madvise(mapped, file_size, MADV_SEQUENTIAL);

        const char* p   = static_cast<const char*>(mapped);
        const char* end = p + file_size;

        std::uint64_t prev_ts = 0U;
        const std::uint64_t throttle_gap_ns = (cfg_.mode == ReplayMode::Throttled)
            ? (1'000'000'000ULL / std::max(cfg_.msgs_per_sec, 1U)) : 0ULL;

        while (p < end && !shutdown_.load(std::memory_order_acquire))
        {
            if (detail::is_header(p, end)) { p = detail::skip_line(p, end); continue; }

            std::uint64_t ts_ns    = 0U;
            std::uint32_t instr_id = 0U;
            float bid_price = 0.0F, bid_qty = 0.0F;
            float ask_price = 0.0F, ask_qty = 0.0F;

            p = detail::parse_u64  (p, end, ts_ns);     p = detail::skip_comma(p, end);
            p = detail::parse_u32  (p, end, instr_id);  p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, bid_price);  p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, bid_qty);    p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, ask_price);  p = detail::skip_comma(p, end);
            p = detail::parse_float(p, end, ask_qty);
            p = detail::skip_line(p, end);

            if (bid_price == 0.0F || ask_price == 0.0F || instr_id >= k_max_instruments)
            {
                metrics_.parse_errors.fetch_add(1U, std::memory_order_relaxed);
                continue;
            }

            metrics_.rows_read.fetch_add(1U, std::memory_order_relaxed);

            if (cfg_.mode == ReplayMode::Realtime && prev_ts != 0U && ts_ns > prev_ts)
            {
                const std::uint64_t gap = ts_ns - prev_ts;
                const std::uint64_t cap = static_cast<std::uint64_t>(cfg_.max_gap_ms) * 1'000'000ULL;
                const std::uint64_t sl  = (gap < cap) ? gap : cap;
                if (sl > 100'000ULL)
                {
                    struct timespec req{ static_cast<time_t>(sl / 1'000'000'000ULL),
                                        static_cast<long>  (sl % 1'000'000'000ULL) };
                    ::nanosleep(&req, nullptr);
                }
            }
            else if (cfg_.mode == ReplayMode::Throttled && throttle_gap_ns > 0U)
            {
                struct timespec req{ static_cast<time_t>(throttle_gap_ns / 1'000'000'000ULL),
                                     static_cast<long>  (throttle_gap_ns % 1'000'000'000ULL) };
                ::nanosleep(&req, nullptr);
            }

            prev_ts = ts_ns;

            LobUpdate bid{};
            bid.timestamp_ns  = ts_ns;
            bid.instrument_id = instr_id;
            bid.depth_level   = 0U;
            bid.side          = Side::Bid;
            bid.price         = bid_price;
            bid.quantity      = bid_qty;

            LobUpdate ask{};
            ask.timestamp_ns  = ts_ns;
            ask.instrument_id = instr_id;
            ask.depth_level   = 0U;
            ask.side          = Side::Ask;
            ask.price         = ask_price;
            ask.quantity      = ask_qty;

            if (ring_.try_push(bid)) [[likely]]
                metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
            else
                metrics_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);

            if (ring_.try_push(ask)) [[likely]]
                metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
            else
                metrics_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);
        }

        ::munmap(mapped, file_size);
        metrics_.finished.store(true, std::memory_order_release);
    }

    RingT&            ring_;
    std::string       path_;
    ReplayConfig      cfg_;
    std::atomic<bool> shutdown_{false};
    std::thread       worker_;
    ReplayMetrics     metrics_;
};

} // namespace holo

namespace holo { using namespace holo; }