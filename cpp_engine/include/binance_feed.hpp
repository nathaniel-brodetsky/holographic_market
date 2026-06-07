#pragma once

#include <lob_core.hpp>
#include <lockfree_ring_buffer.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <thread>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/json.hpp>

namespace holo
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

[[nodiscard]] constexpr std::uint32_t instrument_id_from_stream(std::string_view stream_name) noexcept
{
    if (stream_name.size() >= 6)
    {
        const char c0 = stream_name[0];
        const char c1 = stream_name[1];
        const char c2 = stream_name[2];

        if (c0 == 'b' && c1 == 't' && c2 == 'c') return 0U;
        if (c0 == 'e' && c1 == 't' && c2 == 'h') return 1U;
        if (c0 == 's' && c1 == 'o' && c2 == 'l') return 2U;
        if (c0 == 'b' && c1 == 'n' && c2 == 'b') return 3U;
    }
    return std::numeric_limits<std::uint32_t>::max();
}

} // namespace detail

struct alignas(64) FeedMetrics
{
    std::atomic<std::uint64_t> msgs_received{0U};
    std::atomic<std::uint64_t> msgs_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<std::uint64_t> reconnects{0U};
    std::array<std::atomic<std::uint64_t>, k_feed_n_instruments> per_instrument_msgs{};

    FeedMetrics() noexcept
    {
        for (auto& a : per_instrument_msgs)
            a.store(0U, std::memory_order_relaxed);
    }
};

class BinanceFeedHandler final
{
public:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    explicit BinanceFeedHandler(RingT& ring) noexcept
        : ring_{ring}
        , shutdown_{false}
    {}

    ~BinanceFeedHandler() noexcept { stop(); }

    BinanceFeedHandler(const BinanceFeedHandler&) = delete;
    BinanceFeedHandler& operator=(const BinanceFeedHandler&) = delete;
    BinanceFeedHandler(BinanceFeedHandler&&) = delete;
    BinanceFeedHandler& operator=(BinanceFeedHandler&&) = delete;

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
    static constexpr std::string_view k_ws_host = "stream.binance.com";
    static constexpr std::string_view k_ws_port = "9443";
    static constexpr std::size_t k_max_frame_bytes = 65536U;

    [[nodiscard]] static std::string build_combined_stream_path()
    {
        std::string path = "/stream?streams=";
        for (std::size_t i = 0U; i < k_feed_n_instruments; ++i)
        {
            if (i > 0U) path += '/';
            path += k_symbols[i];
            path += "@depth5@100ms";
        }
        return path;
    }

    void run_loop() noexcept
    {
        while (!shutdown_.load(std::memory_order_acquire))
        {
            try
            {
                ioc_.restart();
                connect_and_stream();
            }
            catch (...)
            {
                metrics_.reconnects.fetch_add(1U, std::memory_order_relaxed);
            }

            if (!shutdown_.load(std::memory_order_acquire))
            {
                struct timespec req{1, 0};
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
        auto const results = resolver.resolve(
            std::string{k_ws_host}, std::string{k_ws_port});

        using StreamT = websocket::stream<ssl::stream<asio::ip::tcp::socket>>;
        StreamT ws{ioc_, ssl_ctx};

        beast::get_lowest_layer(ws).connect(results);
        ws.next_layer().handshake(ssl::stream_base::client);

        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req)
            {
                req.set(boost::beast::http::field::user_agent, "holographic-engine/0.4");
            }));

        const std::string path = build_combined_stream_path();
        ws.handshake(std::string{k_ws_host}, path);

        beast::flat_buffer buf;
        buf.reserve(k_max_frame_bytes);

        while (!shutdown_.load(std::memory_order_acquire))
        {
            ws.read(buf);
            const auto sv = beast::buffers_to_string(buf.data());
            buf.consume(buf.size());
            parse_and_enqueue(sv);
        }

        beast::error_code ec;
        ws.close(websocket::close_code::normal, ec);
    }

    void parse_and_enqueue(std::string_view raw) noexcept
    {
        try
        {
            boost::json::value jv = boost::json::parse(raw);
            const auto& obj = jv.as_object();

            const auto* stream_val = obj.if_contains("stream");
            const auto* data_val   = obj.if_contains("data");

            if (!stream_val || !data_val) return;

            const std::string_view stream_name = stream_val->as_string();
            const std::uint32_t instr_id = detail::instrument_id_from_stream(stream_name);
            if (instr_id >= k_feed_n_instruments) return;

            const auto& data = data_val->as_object();

            const auto* bids_val = data.if_contains("b");
            const auto* asks_val = data.if_contains("a");
            if (!bids_val || !asks_val) return;

            metrics_.msgs_received.fetch_add(1U, std::memory_order_relaxed);
            metrics_.per_instrument_msgs[instr_id].fetch_add(1U, std::memory_order_relaxed);

            const auto& bids = bids_val->as_array();
            const auto& asks = asks_val->as_array();

            const std::size_t bid_levels = std::min(bids.size(), static_cast<std::size_t>(k_max_depth));
            const std::size_t ask_levels = std::min(asks.size(), static_cast<std::size_t>(k_max_depth));

            for (std::size_t lvl = 0U; lvl < bid_levels; ++lvl)
            {
                const auto& entry = bids[lvl].as_array();
                LobUpdate u{};
                u.timestamp_ns     = static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count());
                u.instrument_id    = instr_id;
                u.depth_level      = static_cast<std::uint8_t>(lvl);
                u.side             = Side::Bid;
                u.price            = static_cast<float>(std::stod(
                    std::string{entry[0].as_string()}));
                u.quantity         = static_cast<float>(std::stod(
                    std::string{entry[1].as_string()}));

                if (!ring_.try_push(u)) [[unlikely]]
                    metrics_.msgs_dropped.fetch_add(1U, std::memory_order_relaxed);
            }

            for (std::size_t lvl = 0U; lvl < ask_levels; ++lvl)
            {
                const auto& entry = asks[lvl].as_array();
                LobUpdate u{};
                u.timestamp_ns     = static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count());
                u.instrument_id    = instr_id;
                u.depth_level      = static_cast<std::uint8_t>(lvl);
                u.side             = Side::Ask;
                u.price            = static_cast<float>(std::stod(
                    std::string{entry[0].as_string()}));
                u.quantity         = static_cast<float>(std::stod(
                    std::string{entry[1].as_string()}));

                if (!ring_.try_push(u)) [[unlikely]]
                    metrics_.msgs_dropped.fetch_add(1U, std::memory_order_relaxed);
            }
        }
        catch (...)
        {
            metrics_.parse_errors.fetch_add(1U, std::memory_order_relaxed);
        }
    }

    RingT&               ring_;
    std::atomic<bool>    shutdown_;
    boost::asio::io_context ioc_{1};
    std::thread          worker_;
    FeedMetrics          metrics_;
};

} // namespace holo