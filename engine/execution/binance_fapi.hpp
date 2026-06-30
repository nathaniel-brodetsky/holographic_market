#pragma once

#include <execution/execution_gateway.hpp>
#include <trading/risk_manager.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>

#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/ssl.h>

namespace holo
{

static constexpr std::string_view k_fapi_host = "testnet.binancefuture.com";
static constexpr std::string_view k_fapi_port = "443";

struct alignas(64) GatewayMetrics
{
    std::atomic<std::uint64_t> orders_submitted{0U};
    std::atomic<std::uint64_t> orders_filled{0U};
    std::atomic<std::uint64_t> orders_rejected{0U};
    std::atomic<std::uint64_t> http_errors{0U};

    static constexpr std::size_t k_used = 4U * sizeof(std::atomic<std::uint64_t>);
    std::byte _pad[64U - k_used]{};
};
static_assert(sizeof(GatewayMetrics) == 64U);

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
        out[i*2U]    = k_hex[(digest[i] >> 4U) & 0xFU];
        out[i*2U+1U] = k_hex[digest[i] & 0xFU];
    }
    return out;
}

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
    params += hmac_sha256_hex(api_secret, params);
    return params;
}

} // namespace detail

class BinanceFapiGateway final : public ExecutionGateway
{
public:
    BinanceFapiGateway(
        std::string_view api_key,
        std::string_view api_secret,
        float            order_size_usd,
        RiskManager&     risk) noexcept
        : api_key_{api_key}
        , api_secret_{api_secret}
        , order_size_usd_{order_size_usd}
        , risk_{risk}
        , ioc_{1}
        , work_{boost::asio::make_work_guard(ioc_)}
        , ssl_ctx_{boost::asio::ssl::context::tlsv12_client}
        , stream_a_{ioc_, ssl_ctx_}
        , stream_b_{ioc_, ssl_ctx_}
    {
        ssl_ctx_.set_default_verify_paths();
    }

    ~BinanceFapiGateway() noexcept override { stop(); }

    BinanceFapiGateway(const BinanceFapiGateway&)            = delete;
    BinanceFapiGateway& operator=(const BinanceFapiGateway&) = delete;
    BinanceFapiGateway(BinanceFapiGateway&&)                 = delete;
    BinanceFapiGateway& operator=(BinanceFapiGateway&&)      = delete;

    void start() override
    {
        shutdown_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { ioc_.run(); });
        boost::asio::co_spawn(ioc_, ensure_connected(stream_a_, connected_a_), boost::asio::detached);
        boost::asio::co_spawn(ioc_, ensure_connected(stream_b_, connected_b_), boost::asio::detached);
    }

    void stop() noexcept override
    {
        shutdown_.store(true, std::memory_order_release);
        ioc_.stop();
        if (worker_.joinable()) worker_.join();
    }

    void submit(const RoutedEdge& edge,
                float             mid_src,
                float             mid_dst) noexcept override
    {
        if (shutdown_.load(std::memory_order_relaxed)) return;
        if (!risk_.pre_trade_check(edge.src_instrument, mid_src)) return;

        boost::asio::co_spawn(
            ioc_,
            place_order_coro(edge, mid_src, mid_dst),
            boost::asio::detached);
    }

    [[nodiscard]] const GatewayMetrics& metrics() const noexcept { return metrics_; }

    void print_summary() const noexcept
    {
        std::printf("  -- Gateway Metrics --\n");
        std::printf("  Orders submitted : %llu\n",
            (unsigned long long)metrics_.orders_submitted.load());
        std::printf("  Orders filled    : %llu\n",
            (unsigned long long)metrics_.orders_filled.load());
        std::printf("  Orders rejected  : %llu\n",
            (unsigned long long)metrics_.orders_rejected.load());
        std::printf("  HTTP errors      : %llu\n",
            (unsigned long long)metrics_.http_errors.load());
    }

private:
    using Stream = boost::beast::ssl_stream<boost::beast::tcp_stream>;

    boost::asio::awaitable<void> ensure_connected(Stream& stream, std::atomic<bool>& connected)
    {
        namespace beast = boost::beast;
        namespace asio  = boost::asio;
        namespace ssl   = asio::ssl;

        try
        {
            auto executor = co_await asio::this_coro::executor;
            asio::ip::tcp::resolver resolver{executor};
            const auto ep = co_await resolver.async_resolve(
                std::string{k_fapi_host}, std::string{k_fapi_port}, asio::use_awaitable);

            beast::get_lowest_layer(stream).close();

            if (!SSL_set_tlsext_host_name(stream.native_handle(), k_fapi_host.data()))
            {
                connected.store(false, std::memory_order_release);
                co_return;
            }

            co_await beast::get_lowest_layer(stream).async_connect(ep, asio::use_awaitable);
            beast::get_lowest_layer(stream).socket().set_option(asio::ip::tcp::no_delay(true));
            co_await stream.async_handshake(ssl::stream_base::client, asio::use_awaitable);

            connected.store(true, std::memory_order_release);
        }
        catch (...)
        {
            connected.store(false, std::memory_order_release);
        }
    }

    boost::asio::awaitable<bool> send_request(Stream& stream, std::atomic<bool>& connected, const std::string& body)
    {
        namespace beast = boost::beast;
        namespace http  = beast::http;
        namespace asio  = boost::asio;

        if (!connected.load(std::memory_order_acquire))
            co_await ensure_connected(stream, connected);

        if (!connected.load(std::memory_order_acquire))
            co_return false;

        try
        {
            http::request<http::string_body> req{http::verb::post, "/fapi/v1/order", 11};
            req.set(http::field::host,         k_fapi_host);
            req.set(http::field::content_type, "application/x-www-form-urlencoded");
            req.set("X-MBX-APIKEY", api_key_);
            req.keep_alive(true);
            req.body() = body;
            req.prepare_payload();

            co_await http::async_write(stream, req, asio::use_awaitable);

            beast::flat_buffer buf;
            http::response<http::string_body> res;
            co_await http::async_read(stream, buf, res, asio::use_awaitable);

            if (!res.keep_alive())
                connected.store(false, std::memory_order_release);

            if (res.result() == http::status::ok)
            {
                metrics_.orders_filled.fetch_add(1U, std::memory_order_relaxed);
                co_return true;
            }

            metrics_.orders_rejected.fetch_add(1U, std::memory_order_relaxed);
            co_return false;
        }
        catch (...)
        {
            connected.store(false, std::memory_order_release);
            metrics_.http_errors.fetch_add(1U, std::memory_order_relaxed);
            co_return false;
        }
    }

    boost::asio::awaitable<void>
    place_order_coro(RoutedEdge edge, float mid_src, float mid_dst)
    {
        using namespace boost::asio::experimental::awaitable_operators;

        if (edge.src_instrument >= k_n_instruments ||
            edge.dst_instrument >= k_n_instruments)
            co_return;

        const float qty_src = order_size_usd_ / (mid_src > 0.0F ? mid_src : 1.0F);
        const float qty_dst = order_size_usd_ / (mid_dst > 0.0F ? mid_dst : 1.0F);
        const bool  long_s  = edge.harmonic_flow > 0.0F;

        const std::uint64_t ts_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        const std::string body_src = detail::build_order_body(
            k_symbols_upper[edge.src_instrument],
            long_s ? "BUY" : "SELL", qty_src, api_secret_, ts_ms);
        const std::string body_dst = detail::build_order_body(
            k_symbols_upper[edge.dst_instrument],
            long_s ? "SELL" : "BUY", qty_dst, api_secret_, ts_ms + 1U);

        metrics_.orders_submitted.fetch_add(2U, std::memory_order_relaxed);

        auto [src_ok, dst_ok] = co_await (
            send_request(stream_a_, connected_a_, body_src) &&
            send_request(stream_b_, connected_b_, body_dst));

        if (src_ok) risk_.record_fill(edge.src_instrument, qty_src, mid_src,  long_s);
        if (dst_ok) risk_.record_fill(edge.dst_instrument, qty_dst, mid_dst, !long_s);
    }

    const std::string api_key_;
    const std::string api_secret_;
    const float       order_size_usd_;
    RiskManager&      risk_;

    std::atomic<bool>      shutdown_{false};
    boost::asio::io_context ioc_;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_;
    boost::asio::ssl::context ssl_ctx_;
    Stream                 stream_a_;
    Stream                 stream_b_;
    std::atomic<bool>      connected_a_{false};
    std::atomic<bool>      connected_b_{false};
    std::thread            worker_;
    GatewayMetrics         metrics_;
};

} // namespace holo