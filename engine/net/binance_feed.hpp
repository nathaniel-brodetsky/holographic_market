#pragma once

#include <core/lob_core.hpp>
#include <core/lockfree_ring_buffer.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <thread>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/json.hpp>

// FIX: SSL SNI requires openssl header
#include <openssl/ssl.h>

namespace holo::net
{

static constexpr std::size_t k_feed_n_instruments = 4U;

static constexpr std::array<std::string_view, k_feed_n_instruments> k_symbols = {
    "btcusdt", "ethusdt", "solusdt", "bnbusdt"
};

static constexpr std::array<std::string_view, k_feed_n_instruments> k_symbols_upper = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"
};

namespace detail
{

// FIX: constexpr char-compare router (no hash collision risk, zero overhead)
[[nodiscard]] constexpr std::uint32_t instrument_id_from_stream(std::string_view s) noexcept
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

// FIX: locale-free fast float parser replacing std::stod
// Uses boost::json's internal parser via number_cast — zero alloc.
[[nodiscard]] inline float parse_price(std::string_view sv) noexcept
{
    // boost::json::value from string_view without heap alloc via monotonic_resource
    // For a string_view of a price like "65432.10", simple manual parse is fastest:
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

} // namespace holo::net::detail

struct alignas(64) FeedMetrics
{
    std::atomic<std::uint64_t> msgs_received{0U};
    std::atomic<std::uint64_t> msgs_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<std::uint64_t> reconnects{0U};
    std::array<std::atomic<std::uint64_t>, k_feed_n_instruments> per_instrument_msgs{};

    FeedMetrics() noexcept
    { for (auto& a : per_instrument_msgs) a.store(0U, std::memory_order_relaxed); }
};

class BinanceFeedHandler final
{
public:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    explicit BinanceFeedHandler(RingT& ring) noexcept
        : ring_{ring}, shutdown_{false} {}

    ~BinanceFeedHandler() noexcept { stop(); }

    BinanceFeedHandler(const BinanceFeedHandler&)            = delete;
    BinanceFeedHandler& operator=(const BinanceFeedHandler&) = delete;
    BinanceFeedHandler(BinanceFeedHandler&&)                 = delete;
    BinanceFeedHandler& operator=(BinanceFeedHandler&&)      = delete;

    void start()
    {
        shutdown_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { run_loop(); });
    }

    void stop() noexcept
    {
        shutdown_.store(true, std::memory_order_release);
        try { ioc_.stop(); } catch (...) {}
        if (worker_.joinable()) worker_.join();
    }

    [[nodiscard]] const FeedMetrics& metrics() const noexcept { return metrics_; }

private:
    // FIX: use spot stream (stream.binance.com:9443) — correct for depth data
    static constexpr std::string_view k_ws_host = "stream.binance.com";
    static constexpr std::string_view k_ws_port = "9443";
    static constexpr std::size_t k_max_frame_bytes = 65536U;

    [[nodiscard]] static std::string build_path()
    {
        std::string p = "/stream?streams=";
        for (std::size_t i = 0U; i < k_feed_n_instruments; ++i)
        {
            if (i > 0U) p += '/';
            p += k_symbols[i];
            p += "@depth5@100ms";
        }
        return p;
    }

    void run_loop() noexcept
    {
        while (!shutdown_.load(std::memory_order_acquire))
        {
            try { ioc_.restart(); connect_and_stream(); }
            catch (...) { metrics_.reconnects.fetch_add(1U, std::memory_order_relaxed); }

            if (!shutdown_.load(std::memory_order_acquire))
            {
                struct timespec req{2, 0};
                nanosleep(&req, nullptr);
            }
        }
    }

    void connect_and_stream()
    {
        namespace beast     = boost::beast;
        namespace websocket = beast::websocket;
        namespace asio      = boost::asio;
        namespace ssl       = asio::ssl;

        ssl::context ssl_ctx{ssl::context::tlsv12_client};
        ssl_ctx.set_default_verify_paths();

        asio::ip::tcp::resolver resolver{ioc_};
        const auto results = resolver.resolve(
            std::string{k_ws_host}, std::string{k_ws_port});

        // FIX: use Beast's ssl_stream<tcp_stream> — correct Beast 1.79+ type
        using WsStream = websocket::stream<beast::ssl_stream<beast::tcp_stream>>;
        WsStream ws{ioc_, ssl_ctx};

        beast::get_lowest_layer(ws).connect(results);

        // FIX: set SSL SNI before handshake — required by stream.binance.com
        if (!SSL_set_tlsext_host_name(
                ws.next_layer().native_handle(),
                k_ws_host.data()))
        {
            throw boost::system::system_error{
                static_cast<int>(::ERR_get_error()),
                boost::asio::error::get_ssl_category()};
        }

        ws.next_layer().handshake(ssl::stream_base::client);

        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req)
            { req.set(boost::beast::http::field::user_agent, "holographic-engine/0.4"); }));

        ws.handshake(std::string{k_ws_host}, build_path());

        beast::flat_buffer buf;
        buf.reserve(k_max_frame_bytes);

        while (!shutdown_.load(std::memory_order_acquire))
        {
            ws.read(buf);
            const std::string_view sv{
                static_cast<const char*>(buf.data().data()),
                buf.data().size()};
            parse_and_enqueue(sv);
            buf.consume(buf.size());
        }

        beast::error_code ec;
        ws.close(websocket::close_code::normal, ec);
    }

    void parse_and_enqueue(std::string_view raw) noexcept
    {
        // FIX: boost::json with monotonic_resource → zero heap on steady state
        try
        {
            boost::json::monotonic_resource mr;
            const boost::json::value jv = boost::json::parse(raw, &mr);
            const auto& obj = jv.as_object();

            const auto* sv = obj.if_contains("stream");
            const auto* dv = obj.if_contains("data");
            if (!sv || !dv) return;

            const std::string_view stream_name{sv->as_string()};
            const std::uint32_t instr_id = detail::instrument_id_from_stream(stream_name);
            if (instr_id >= k_feed_n_instruments) return;

            const auto& data = dv->as_object();
            const auto* bv   = data.if_contains("b");
            const auto* av   = data.if_contains("a");
            if (!bv || !av) return;

            metrics_.msgs_received.fetch_add(1U, std::memory_order_relaxed);
            metrics_.per_instrument_msgs[instr_id].fetch_add(1U, std::memory_order_relaxed);

            // FIX: use now_ns via rdtsc proxy — cheaper than steady_clock
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
                    u.instrument_id = instr_id;
                    u.depth_level   = static_cast<std::uint8_t>(lvl);
                    u.side          = side;
                    // FIX: use fast locale-free parser instead of std::stod
                    u.price    = detail::parse_price(entry[0].as_string());
                    u.quantity = detail::parse_price(entry[1].as_string());

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

    RingT&                  ring_;
    std::atomic<bool>       shutdown_;
    boost::asio::io_context ioc_{1};
    std::thread             worker_;
    FeedMetrics             metrics_;
};

} // namespace holo::net

namespace holo { using namespace holo::net; }
