#pragma once

#include <net/signal_router.hpp>
#include <net/binance_feed.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>

#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/ssl.h>

namespace holo::net
{

static constexpr std::string_view k_binance_futures_host = "testnet.binancefuture.com";
static constexpr std::string_view k_binance_futures_port = "443";

// FIX: correct pad sizes with static_assert
struct alignas(64) PaperPosition
{
    std::atomic<float>         net_qty{0.0F};
    std::atomic<float>         avg_entry_price{0.0F};
    std::atomic<float>         realized_pnl{0.0F};
    std::atomic<std::uint64_t> n_orders{0U};

    static constexpr std::size_t k_used =
        3U * sizeof(std::atomic<float>) + sizeof(std::atomic<std::uint64_t>);
    std::byte _pad[64U - k_used]{};
};
static_assert(sizeof(PaperPosition) == 64U);

struct alignas(64) ExecutionMetrics
{
    std::atomic<std::uint64_t> orders_submitted{0U};
    std::atomic<std::uint64_t> orders_filled{0U};
    std::atomic<std::uint64_t> orders_rejected{0U};
    std::atomic<std::uint64_t> http_errors{0U};
    std::atomic<float>         total_pnl{0.0F};

    static constexpr std::size_t k_used =
        4U * sizeof(std::atomic<std::uint64_t>) + sizeof(std::atomic<float>);
    std::byte _pad[64U - k_used]{};
};
static_assert(sizeof(ExecutionMetrics) == 64U);

namespace detail
{

[[nodiscard]] inline std::string hmac_sha256_hex(
    std::string_view key, std::string_view data) noexcept
{
    unsigned char digest[SHA256_DIGEST_LENGTH];
    unsigned int  len = 0U;
    HMAC(EVP_sha256(),
         key.data(),  static_cast<int>(key.size()),
         reinterpret_cast<const unsigned char*>(data.data()), data.size(),
         digest, &len);
    static constexpr char k_hex[] = "0123456789abcdef";
    std::string out(SHA256_DIGEST_LENGTH * 2U, '\0');
    for (unsigned int i = 0U; i < len; ++i)
    {
        out[i*2U]   = k_hex[(digest[i] >> 4U) & 0xFU];
        out[i*2U+1U]= k_hex[digest[i] & 0xFU];
    }
    return out;
}

// FIX: build params for POST body (not URL query string)
[[nodiscard]] inline std::string build_order_body(
    std::string_view symbol,
    std::string_view side,
    float            qty,
    std::string_view api_secret,
    std::uint64_t    ts_ms)
{
    char qty_buf[32];
    std::snprintf(qty_buf, sizeof(qty_buf), "%.6f", static_cast<double>(qty));

    std::string params;
    params.reserve(256U);
    params += "symbol=";    params += symbol;
    params += "&side=";     params += side;
    params += "&type=MARKET&quantity="; params += qty_buf;
    params += "&timestamp="; params += std::to_string(ts_ms);

    params += "&signature=";
    params += hmac_sha256_hex(api_secret, params.substr(0, params.size()));
    return params;
}

} // namespace holo::net::detail

class ExecutionEngine final
{
public:
    ExecutionEngine(
        std::string_view api_key,
        std::string_view api_secret,
        float            max_position_usd,
        float            order_size_usd) noexcept
        : api_key_{api_key}
        , api_secret_{api_secret}
        , max_position_usd_{max_position_usd}
        , order_size_usd_{order_size_usd}
        , ioc_{1}
    {}

    ~ExecutionEngine() noexcept { stop(); }

    ExecutionEngine(const ExecutionEngine&)            = delete;
    ExecutionEngine& operator=(const ExecutionEngine&) = delete;
    ExecutionEngine(ExecutionEngine&&)                 = delete;
    ExecutionEngine& operator=(ExecutionEngine&&)      = delete;

    void start()
    {
        shutdown_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { ioc_.run(); });
    }

    void stop() noexcept
    {
        shutdown_.store(true, std::memory_order_release);
        ioc_.stop();
        if (worker_.joinable()) worker_.join();
    }

    void submit(const RoutedEdge& edge, float mid_src, float mid_dst) noexcept
    {
        if (shutdown_.load(std::memory_order_relaxed)) return;
        boost::asio::co_spawn(
            ioc_,
            place_order_coro(edge, mid_src, mid_dst),
            boost::asio::detached);
    }

    [[nodiscard]] const ExecutionMetrics& metrics() const noexcept { return metrics_; }

    void print_risk_summary() const noexcept
    {
        std::printf("  ── Execution Engine Risk Summary ──\n");
        std::printf("  Orders submitted : %llu\n",
            (unsigned long long)metrics_.orders_submitted.load());
        std::printf("  Orders filled    : %llu\n",
            (unsigned long long)metrics_.orders_filled.load());
        std::printf("  Orders rejected  : %llu\n",
            (unsigned long long)metrics_.orders_rejected.load());
        std::printf("  HTTP errors      : %llu\n",
            (unsigned long long)metrics_.http_errors.load());
        std::printf("  Realized PnL     : %.4f USD\n",
            static_cast<double>(metrics_.total_pnl.load()));
        for (std::size_t i = 0U; i < k_feed_n_instruments; ++i)
        {
            std::printf("  [%s] qty=%.4f rPnL=%.4f\n",
                k_symbols_upper[i].data(),
                static_cast<double>(positions_[i].net_qty.load()),
                static_cast<double>(positions_[i].realized_pnl.load()));
        }
    }

private:
    boost::asio::awaitable<void> place_order_coro(
        RoutedEdge edge, float mid_src, float mid_dst)
    {
        if (edge.src_instrument >= k_feed_n_instruments ||
            edge.dst_instrument >= k_feed_n_instruments)
            co_return;

        {
            const float cur  = positions_[edge.src_instrument].net_qty.load(std::memory_order_relaxed);
            if (std::abs(cur) * mid_src > max_position_usd_) co_return;
        }

        const float qty_src = order_size_usd_ / (mid_src > 0.0F ? mid_src : 1.0F);
        const float qty_dst = order_size_usd_ / (mid_dst > 0.0F ? mid_dst : 1.0F);
        const bool  long_s  = edge.harmonic_flow > 0.0F;

        const std::uint64_t ts_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        // FIX: build_order_body puts params in POST body, not URL
        const std::string body_src = detail::build_order_body(
            k_symbols_upper[edge.src_instrument],
            long_s ? "BUY" : "SELL", qty_src, api_secret_, ts_ms);
        const std::string body_dst = detail::build_order_body(
            k_symbols_upper[edge.dst_instrument],
            long_s ? "SELL" : "BUY", qty_dst, api_secret_, ts_ms + 1U);

        metrics_.orders_submitted.fetch_add(2U, std::memory_order_relaxed);

        co_await send_request(body_src, edge.src_instrument, qty_src, mid_src,  long_s);
        co_await send_request(body_dst, edge.dst_instrument, qty_dst, mid_dst, !long_s);
    }

    boost::asio::awaitable<void> send_request(
        const std::string& body,
        std::uint32_t      instr_id,
        float              qty,
        float              mid,
        bool               is_buy)
    {
        namespace beast = boost::beast;
        namespace http  = beast::http;
        namespace asio  = boost::asio;
        namespace ssl   = asio::ssl;

        auto executor = co_await boost::asio::this_coro::executor;

        try
        {
            // FIX: construct ssl_ctx locally — not shared across coroutines
            ssl::context ssl_ctx{ssl::context::tlsv12_client};
            ssl_ctx.set_default_verify_paths();

            asio::ip::tcp::resolver resolver{executor};
            const auto ep = co_await resolver.async_resolve(
                std::string{k_binance_futures_host},
                std::string{k_binance_futures_port},
                boost::asio::use_awaitable);

            beast::ssl_stream<beast::tcp_stream> stream{executor, ssl_ctx};

            // FIX: set SNI before handshake
            if (!SSL_set_tlsext_host_name(
                    stream.native_handle(),
                    k_binance_futures_host.data()))
                co_return;

            co_await beast::get_lowest_layer(stream).async_connect(
                ep, boost::asio::use_awaitable);
            co_await stream.async_handshake(
                ssl::stream_base::client, boost::asio::use_awaitable);

            // FIX: POST body carries the query, URL is clean /fapi/v1/order
            http::request<http::string_body> req{http::verb::post, "/fapi/v1/order", 11};
            req.set(http::field::host,         k_binance_futures_host);
            req.set(http::field::content_type, "application/x-www-form-urlencoded");
            req.set("X-MBX-APIKEY", api_key_);
            req.body() = body;
            req.prepare_payload();

            co_await http::async_write(stream, req, boost::asio::use_awaitable);

            beast::flat_buffer buf;
            http::response<http::string_body> res;
            co_await http::async_read(stream, buf, res, boost::asio::use_awaitable);

            if (res.result() == http::status::ok)
            {
                update_paper_position(instr_id, qty, mid, is_buy);
                metrics_.orders_filled.fetch_add(1U, std::memory_order_relaxed);
            }
            else { metrics_.orders_rejected.fetch_add(1U, std::memory_order_relaxed); }

            beast::error_code ec; stream.shutdown(ec);
        }
        catch (...) { metrics_.http_errors.fetch_add(1U, std::memory_order_relaxed); }
    }

    void update_paper_position(
        std::uint32_t id, float qty, float price, bool is_buy) noexcept
    {
        if (id >= k_feed_n_instruments) return;
        auto& pos = positions_[id];

        const float sign     = is_buy ? 1.0F : -1.0F;
        const float prev_qty = pos.net_qty.load(std::memory_order_relaxed);
        const float prev_ep  = pos.avg_entry_price.load(std::memory_order_relaxed);
        const float new_qty  = prev_qty + sign * qty;

        if (std::abs(new_qty) < 1e-9F)
        {
            const float rpnl = prev_qty * (price - prev_ep);
            pos.realized_pnl.fetch_add(rpnl, std::memory_order_relaxed);
            metrics_.total_pnl.fetch_add(rpnl, std::memory_order_relaxed);
            pos.net_qty.store(0.0F, std::memory_order_relaxed);
            pos.avg_entry_price.store(0.0F, std::memory_order_relaxed);
        }
        else if (prev_qty == 0.0F || (prev_qty > 0.0F) == is_buy)
        {
            const float ne = (prev_ep * std::abs(prev_qty) + price * qty) /
                             (std::abs(prev_qty) + qty);
            pos.avg_entry_price.store(ne, std::memory_order_relaxed);
            pos.net_qty.store(new_qty, std::memory_order_relaxed);
        }
        else
        {
            const float closed = std::min(std::abs(prev_qty), qty);
            const float rpnl   = (is_buy ? -1.0F : 1.0F) * closed * (price - prev_ep);
            pos.realized_pnl.fetch_add(rpnl, std::memory_order_relaxed);
            metrics_.total_pnl.fetch_add(rpnl, std::memory_order_relaxed);
            pos.net_qty.store(new_qty, std::memory_order_relaxed);
        }
        pos.n_orders.fetch_add(1U, std::memory_order_relaxed);
    }

    const std::string api_key_;
    const std::string api_secret_;
    const float       max_position_usd_;
    const float       order_size_usd_;

    std::atomic<bool>       shutdown_{false};
    boost::asio::io_context ioc_;
    std::thread             worker_;

    std::array<PaperPosition, k_feed_n_instruments> positions_{};
    ExecutionMetrics                                 metrics_;
};

} // namespace holo::net

namespace holo { using namespace holo::net; }
