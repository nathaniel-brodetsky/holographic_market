#pragma once

#include <lob_core.hpp>
#include <lockfree_ring_buffer.hpp>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <simdjson.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <thread>

namespace holo {
    namespace net = boost::asio;
    namespace ssl = boost::asio::ssl;
    namespace beast = boost::beast;
    namespace ws = boost::beast::websocket;
    namespace http = boost::beast::http;

    using tcp = net::ip::tcp;
    using WsStream = ws::stream<beast::ssl_stream<beast::tcp_stream> >;

    static constexpr std::string_view k_binance_host = "fstream.binance.com";
    static constexpr std::string_view k_binance_port = "443";
    static constexpr std::string_view k_binance_target = "/stream?streams=btcusdt@depth20@100ms";
    static constexpr std::size_t k_read_buffer_bytes = 1U << 17U;
    static constexpr std::size_t k_simdjson_padding = simdjson::SIMDJSON_PADDING;

    struct FeedStats {
        std::atomic<std::uint64_t> messages_received{0U};
        std::atomic<std::uint64_t> updates_pushed{0U};
        std::atomic<std::uint64_t> updates_dropped{0U};
        std::atomic<std::uint64_t> parse_errors{0U};
        std::atomic<std::uint64_t> reconnects{0U};
    };

    class BinanceFeedHandler final {
    public:
        explicit BinanceFeedHandler(
            DynamicSpscRingBuffer<LobUpdate> &ring,
            std::uint32_t instrument_id = 0U) noexcept
            : ring_{ring}, instrument_id_{instrument_id} {
        }

        ~BinanceFeedHandler() noexcept { stop(); }

        BinanceFeedHandler(const BinanceFeedHandler &) = delete;

        BinanceFeedHandler &operator=(const BinanceFeedHandler &) = delete;

        BinanceFeedHandler(BinanceFeedHandler &&) = delete;

        BinanceFeedHandler &operator=(BinanceFeedHandler &&) = delete;

        void start() {
            shutdown_.store(false, std::memory_order_release);
            thread_ = std::thread([this]() { run_loop(); });
        }

        void stop() noexcept {
            shutdown_.store(true, std::memory_order_release);
            ioc_.stop();
            if (thread_.joinable()) thread_.join();
        }

        [[nodiscard]] const FeedStats &stats() const noexcept { return stats_; }

    private:
        void run_loop() noexcept {
            while (!shutdown_.load(std::memory_order_acquire)) {
                try {
                    ioc_.restart();
                    connect_and_consume();
                } catch (const std::exception &e) {
                    std::fprintf(stderr, "[BinanceFeed] %s — reconnecting\n", e.what());
                    stats_.reconnects.fetch_add(1U, std::memory_order_relaxed);
                    struct timespec req{1, 0};
                    nanosleep(&req, nullptr);
                }
            }
        }

        void connect_and_consume() {
            ssl::context ssl_ctx{ssl::context::tlsv13_client};
            ssl_ctx.set_default_verify_paths();
            ssl_ctx.set_verify_mode(ssl::verify_peer);

            WsStream stream{ioc_, ssl_ctx};

            tcp::resolver resolver{ioc_};
            const auto endpoints = resolver.resolve(
                std::string(k_binance_host), std::string(k_binance_port));

            beast::get_lowest_layer(stream).connect(endpoints);
            beast::get_lowest_layer(stream).expires_never();

            if (!SSL_set_tlsext_host_name(
                stream.next_layer().native_handle(),
                k_binance_host.data()))
                throw boost::system::system_error{
                    static_cast<int>(::ERR_get_error()),
                    boost::asio::error::get_ssl_category()
                };

            stream.next_layer().handshake(ssl::stream_base::client);

            stream.set_option(ws::stream_base::decorator(
                [](ws::request_type &req) {
                    req.set(http::field::user_agent, "holographic-engine/0.3.0");
                }));
            stream.handshake(std::string(k_binance_host), std::string(k_binance_target));

            std::fprintf(stdout, "[BinanceFeed] connected to %s%s\n",
                         k_binance_host.data(), k_binance_target.data());

            beast::flat_buffer buf;
            buf.reserve(k_read_buffer_bytes);

            simdjson::ondemand::parser parser;
            std::string padded;
            padded.reserve(k_read_buffer_bytes + k_simdjson_padding);

            while (!shutdown_.load(std::memory_order_acquire)) {
                buf.clear();
                stream.read(buf);
                stats_.messages_received.fetch_add(1U, std::memory_order_relaxed);
                const auto data = buf.cdata();
                dispatch_message(
                    {static_cast<const char *>(data.data()), data.size()},
                    parser, padded);
            }

            beast::error_code ec;
            stream.close(ws::close_code::normal, ec);
        }

        void dispatch_message(
            std::string_view raw,
            simdjson::ondemand::parser &parser,
            std::string &padded) noexcept {
            padded.assign(raw);
            padded.resize(raw.size() + k_simdjson_padding, '\0');

            simdjson::ondemand::document doc;
            if (parser.iterate(padded.data(), raw.size(), padded.size()).get(doc) != simdjson::SUCCESS) {
                stats_.parse_errors.fetch_add(1U, std::memory_order_relaxed);
                return;
            }

            simdjson::ondemand::object data_obj;
            if (doc["data"].get(data_obj) != simdjson::SUCCESS) return;

            simdjson::ondemand::array bids_arr, asks_arr;
            if (data_obj["b"].get(bids_arr) == simdjson::SUCCESS) push_levels(bids_arr, Side::Bid);
            if (data_obj["a"].get(asks_arr) == simdjson::SUCCESS) push_levels(asks_arr, Side::Ask);
        }

        void push_levels(simdjson::ondemand::array &levels, Side side) noexcept {
            std::uint8_t depth_idx = 0U;
            for (auto entry: levels) {
                simdjson::ondemand::array pair;
                if (entry.get(pair) != simdjson::SUCCESS) break;
                auto it = pair.begin();
                if (it == pair.end()) break;
                std::string_view price_sv;
                if ((*it).get_string().get(price_sv) != simdjson::SUCCESS) break;
                ++it;
                if (it == pair.end()) break;
                std::string_view qty_sv;
                if ((*it).get_string().get(qty_sv) != simdjson::SUCCESS) break;

                LobUpdate u{};
                u.timestamp_ns = static_cast<std::uint64_t>(__builtin_ia32_rdtsc());
                u.instrument_id = instrument_id_;
                u.depth_level = depth_idx;
                u.side = side;
                std::sscanf(price_sv.data(), "%f", &u.price);
                std::sscanf(qty_sv.data(), "%f", &u.quantity);

                if (ring_.try_push(u)) [[likely]]
                        stats_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
                else
                    stats_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);

                if (++depth_idx >= static_cast<std::uint8_t>(k_max_depth)) break;
            }
        }

        DynamicSpscRingBuffer<LobUpdate> &ring_;
        std::uint32_t instrument_id_;
        std::atomic<bool> shutdown_{true};
        net::io_context ioc_{1};
        std::thread thread_;
        FeedStats stats_;
    };
} // namespace holo
