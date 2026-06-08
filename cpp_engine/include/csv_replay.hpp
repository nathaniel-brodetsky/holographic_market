#pragma once

#include <lob_core.hpp>
#include <lockfree_ring_buffer.hpp>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace holo
{

struct alignas(64) ReplayMetrics
{
    std::atomic<std::uint64_t> lines_parsed{0U};
    std::atomic<std::uint64_t> lines_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<std::uint64_t> ring_drops{0U};
    std::byte _pad[64U - 4U * sizeof(std::atomic<std::uint64_t>)]{};
};

namespace detail
{

[[nodiscard]] __attribute__((always_inline)) inline
const char* skip_whitespace(const char* p, const char* end) noexcept
{
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
    return p;
}

[[nodiscard]] __attribute__((always_inline)) inline
const char* parse_u64(const char* p, const char* end, std::uint64_t& out) noexcept
{
    std::uint64_t v = 0U;
    bool any = false;
    while (p < end)
    {
        const unsigned c = static_cast<unsigned char>(*p) - '0';
        if (c > 9U) break;
        v = v * 10U + c;
        any = true;
        ++p;
    }
    out = v;
    return any ? p : nullptr;
}

[[nodiscard]] __attribute__((always_inline)) inline
const char* parse_u32(const char* p, const char* end, std::uint32_t& out) noexcept
{
    std::uint32_t v = 0U;
    bool any = false;
    while (p < end)
    {
        const unsigned c = static_cast<unsigned char>(*p) - '0';
        if (c > 9U) break;
        v = v * 10U + c;
        any = true;
        ++p;
    }
    out = v;
    return any ? p : nullptr;
}

// Branchless decimal float parser. No heap, no locale, no libc.
// Handles: [+-]?[0-9]*\.?[0-9]*([eE][+-]?[0-9]+)?
[[nodiscard]] __attribute__((always_inline)) inline
const char* parse_float(const char* p, const char* end, float& out) noexcept
{
    static constexpr double k_pow10[16] = {
        1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
        1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15
    };

    if (p >= end) return nullptr;

    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    else if (*p == '+') { ++p; }

    std::uint64_t mantissa = 0U;
    int frac_digits = 0;
    bool any = false;

    while (p < end)
    {
        const unsigned c = static_cast<unsigned char>(*p) - '0';
        if (c > 9U) break;
        mantissa = mantissa * 10U + c;
        any = true;
        ++p;
    }

    if (p < end && *p == '.')
    {
        ++p;
        while (p < end)
        {
            const unsigned c = static_cast<unsigned char>(*p) - '0';
            if (c > 9U) break;
            mantissa = mantissa * 10U + c;
            ++frac_digits;
            any = true;
            ++p;
        }
    }

    if (!any) return nullptr;

    int exp_adj = 0;
    if (p < end && (*p == 'e' || *p == 'E'))
    {
        ++p;
        bool eneg = false;
        if (p < end && *p == '-') { eneg = true; ++p; }
        else if (p < end && *p == '+') { ++p; }
        int ev = 0;
        while (p < end)
        {
            const unsigned c = static_cast<unsigned char>(*p) - '0';
            if (c > 9U) break;
            ev = ev * 10 + static_cast<int>(c);
            ++p;
        }
        exp_adj = eneg ? -ev : ev;
    }

    const int total_exp = exp_adj - frac_digits;
    double v = static_cast<double>(mantissa);

    if (total_exp >= 0 && total_exp < 16)
        v *= k_pow10[total_exp];
    else if (total_exp < 0 && (-total_exp) < 16)
        v /= k_pow10[-total_exp];
    else
        v *= __builtin_exp(total_exp * 2.302585092994046); // ln(10)

    out = static_cast<float>(neg ? -v : v);
    return p;
}

[[nodiscard]] __attribute__((always_inline)) inline
const char* skip_field(const char* p, const char* end) noexcept
{
    while (p < end && *p != ',' && *p != '\n') ++p;
    return p;
}

[[nodiscard]] __attribute__((always_inline)) inline
const char* expect_comma(const char* p, const char* end) noexcept
{
    if (p < end && *p == ',') return p + 1;
    return nullptr;
}

} // namespace detail

class CsvReplayHandler final
{
public:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    explicit CsvReplayHandler(RingT& ring) noexcept
        : ring_{ring}
        , shutdown_{false}
        , done_{false}
    {}

    ~CsvReplayHandler() noexcept { stop(); unmap(); }

    CsvReplayHandler(const CsvReplayHandler&) = delete;
    CsvReplayHandler& operator=(const CsvReplayHandler&) = delete;
    CsvReplayHandler(CsvReplayHandler&&) = delete;
    CsvReplayHandler& operator=(CsvReplayHandler&&) = delete;

    [[nodiscard]] bool open(const char* path) noexcept
    {
        fd_ = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) return false;

        struct stat st{};
        if (::fstat(fd_, &st) < 0) { ::close(fd_); fd_ = -1; return false; }

        file_size_ = static_cast<std::size_t>(st.st_size);
        if (file_size_ == 0U) { ::close(fd_); fd_ = -1; return false; }

        map_ptr_ = static_cast<const char*>(
            ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_, 0));

        if (map_ptr_ == MAP_FAILED)
        {
            map_ptr_ = nullptr;
            ::close(fd_); fd_ = -1;
            return false;
        }

        ::madvise(const_cast<char*>(map_ptr_), file_size_, MADV_SEQUENTIAL | MADV_WILLNEED);
        return true;
    }

    void start()
    {
        assert(map_ptr_ != nullptr);
        shutdown_.store(false, std::memory_order_release);
        done_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { replay_loop(); });
    }

    void stop() noexcept
    {
        shutdown_.store(true, std::memory_order_release);
        if (worker_.joinable()) worker_.join();
    }

    [[nodiscard]] bool done() const noexcept
    {
        return done_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const ReplayMetrics& metrics() const noexcept { return metrics_; }

private:
    void unmap() noexcept
    {
        if (map_ptr_) { ::munmap(const_cast<char*>(map_ptr_), file_size_); map_ptr_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    void replay_loop() noexcept
    {
        const char* p   = map_ptr_;
        const char* end = map_ptr_ + file_size_;

        // Skip header line if present (non-digit first byte).
        if (p < end && (*p < '0' || *p > '9'))
        {
            while (p < end && *p != '\n') ++p;
            if (p < end) ++p;
        }

        while (p < end && !shutdown_.load(std::memory_order_relaxed))
        {
            p = detail::skip_whitespace(p, end);
            if (p >= end) break;
            if (*p == '\n') { ++p; continue; }

            const char* line_start = p;

            // timestamp_ns
            std::uint64_t ts_ns = 0U;
            p = detail::parse_u64(p, end, ts_ns);
            if (!p) goto bad_line;
            p = detail::expect_comma(p, end);
            if (!p) goto bad_line;

            // instrument_id
            std::uint32_t instr_id;
            p = detail::parse_u32(p, end, instr_id);
            if (!p) goto bad_line;
            p = detail::expect_comma(p, end);
            if (!p) goto bad_line;

            // bid_price
            {
                float bid_price, bid_qty, ask_price, ask_qty;

                p = detail::parse_float(p, end, bid_price);
                if (!p) goto bad_line;
                p = detail::expect_comma(p, end);
                if (!p) goto bad_line;

                p = detail::parse_float(p, end, bid_qty);
                if (!p) goto bad_line;
                p = detail::expect_comma(p, end);
                if (!p) goto bad_line;

                p = detail::parse_float(p, end, ask_price);
                if (!p) goto bad_line;
                p = detail::expect_comma(p, end);
                if (!p) goto bad_line;

                p = detail::parse_float(p, end, ask_qty);
                if (!p) goto bad_line;

                // Advance to next line.
                while (p < end && *p != '\n') ++p;
                if (p < end) ++p;

                LobUpdate bid{};
                bid.timestamp_ns  = ts_ns;
                bid.price         = bid_price;
                bid.quantity      = bid_qty;
                bid.instrument_id = instr_id;
                bid.depth_level   = 0U;
                bid.side          = Side::Bid;

                LobUpdate ask{};
                ask.timestamp_ns  = ts_ns;
                ask.price         = ask_price;
                ask.quantity      = ask_qty;
                ask.instrument_id = instr_id;
                ask.depth_level   = 0U;
                ask.side          = Side::Ask;

                metrics_.lines_parsed.fetch_add(1U, std::memory_order_relaxed);

                if (!ring_.try_push(bid)) [[unlikely]]
                    metrics_.ring_drops.fetch_add(1U, std::memory_order_relaxed);
                if (!ring_.try_push(ask)) [[unlikely]]
                    metrics_.ring_drops.fetch_add(1U, std::memory_order_relaxed);

                continue;
            }

bad_line:
            metrics_.parse_errors.fetch_add(1U, std::memory_order_relaxed);
            while (p < end && *p != '\n') ++p;
            if (p < end) ++p;
            (void)line_start;
        }

        done_.store(true, std::memory_order_release);
    }

    RingT&             ring_;
    std::atomic<bool>  shutdown_;
    std::atomic<bool>  done_;
    std::thread        worker_;
    ReplayMetrics      metrics_;

    const char*        map_ptr_{nullptr};
    std::size_t        file_size_{0U};
    int                fd_{-1};
};

} // namespace holo