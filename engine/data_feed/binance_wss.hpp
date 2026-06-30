#pragma once

// BinanceWssClient — raw WebSocket connection to fstream.binance.com.
// Responsibility: open TLS connection, subscribe to streams, deliver raw
// JSON frames via a callback.  Zero knowledge of LOB, signals, or routing.

#include <common/types.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <thread>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/http/field.hpp>

#include <openssl/ssl.h>

namespace holo
{

class BinanceWssClient final
{
public:
    using FrameCallback = std::function<void(std::string_view)>;

    explicit BinanceWssClient(FrameCallback on_frame) noexcept
        : on_frame_{std::move(on_frame)} {}

    ~BinanceWssClient() noexcept { stop(); }

    BinanceWssClient(const BinanceWssClient&)            = delete;
    BinanceWssClient& operator=(const BinanceWssClient&) = delete;
    BinanceWssClient(BinanceWssClient&&)                 = delete;
    BinanceWssClient& operator=(BinanceWssClient&&)      = delete;

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

    [[nodiscard]] std::uint64_t reconnects() const noexcept
    {
        return reconnects_.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::string_view k_host = "fstream.binance.com";
    static constexpr std::string_view k_port = "443";
    static constexpr std::size_t k_max_frame = 65536U;

    [[nodiscard]] static std::string build_stream_path()
    {
        std::string p = "/stream?streams=";
        for (std::size_t i = 0U; i < k_n_instruments; ++i)
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
            try
            {
                ioc_.restart();
                connect_and_stream();
            }
            catch (const std::exception& e)
            {
                std::fprintf(stderr, "\n[WssClient] %s\n", e.what());
                reconnects_.fetch_add(1U, std::memory_order_relaxed);
            }

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
        const auto ep = resolver.resolve(std::string{k_host}, std::string{k_port});

        using WsStream = websocket::stream<beast::ssl_stream<beast::tcp_stream>>;
        WsStream ws{ioc_, ssl_ctx};

        beast::get_lowest_layer(ws).connect(ep);
        beast::get_lowest_layer(ws).socket().set_option(asio::ip::tcp::no_delay(true));

        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), k_host.data()))
            throw boost::system::system_error{
                static_cast<int>(::ERR_get_error()),
                boost::asio::error::get_ssl_category()};

        ws.next_layer().handshake(ssl::stream_base::client);
        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req)
            { req.set(boost::beast::http::field::user_agent, "holographic-engine/0.5"); }));
        ws.handshake(std::string{k_host}, build_stream_path());

        beast::flat_buffer buf;
        buf.reserve(k_max_frame);

        while (!shutdown_.load(std::memory_order_acquire))
        {
            ws.read(buf);
            on_frame_(std::string_view{
                static_cast<const char*>(buf.data().data()), buf.data().size()});
            buf.consume(buf.size());
        }

        beast::error_code ec;
        ws.close(websocket::close_code::normal, ec);
    }

    FrameCallback           on_frame_;
    std::atomic<bool>       shutdown_{false};
    std::atomic<std::uint64_t> reconnects_{0U};
    boost::asio::io_context ioc_{1};
    std::thread             worker_;
};

} // namespace holo