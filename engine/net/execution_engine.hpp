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
#include <deque>
#include <thread>
#include <memory>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/ssl.h>

namespace holo::net
{

// Эндпоинты для WebSocket Binance Futures Testnet
static constexpr std::string_view k_binance_futures_ws_host = "testnet.binancefuture.com";
static constexpr std::string_view k_binance_futures_ws_port = "443";
static constexpr std::string_view k_binance_futures_ws_path = "/ws-fapi/v1";

struct alignas(64) PaperPosition {
    std::atomic<float>         net_qty{0.0F};
    std::atomic<float>         avg_entry_price{0.0F};
    std::atomic<float>         realized_pnl{0.0F};
    std::atomic<std::uint64_t> n_orders{0U};
};

struct alignas(64) ExecutionMetrics {
    std::atomic<std::uint64_t> orders_submitted{0U};
    std::atomic<std::uint64_t> orders_filled{0U};
    std::atomic<std::uint64_t> orders_rejected{0U};
    std::atomic<std::uint64_t> http_errors{0U};
    std::atomic<float>         total_pnl{0.0F};
    static constexpr std::size_t k_used = 4U * sizeof(std::atomic<std::uint64_t>) + sizeof(std::atomic<float>);
    std::byte _pad[64U - k_used]{};
};

namespace detail {
    // Криптография для подписи ордера (HMAC SHA256)
    [[nodiscard]] inline std::string hmac_sha256_hex(std::string_view key, std::string_view data) noexcept {
        unsigned char digest[SHA256_DIGEST_LENGTH];
        unsigned int  len = 0U;
        HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
             reinterpret_cast<const unsigned char*>(data.data()), data.size(), digest, &len);
        static constexpr char k_hex[] = "0123456789abcdef";
        std::string out(SHA256_DIGEST_LENGTH * 2U, '\0');
        for (unsigned int i = 0U; i < len; ++i) {
            out[i*2U]   = k_hex[(digest[i] >> 4U) & 0xFU];
            out[i*2U+1U]= k_hex[digest[i] & 0xFU];
        }
        return out;
    }

    // Формирование JSON для WebSocket API
    [[nodiscard]] inline std::string build_ws_order_msg(
        std::uint64_t msg_id, std::string_view symbol, std::string_view side, float qty,
        std::string_view api_key, std::string_view api_secret, std::uint64_t ts_ms)
    {
        int precision = (symbol == "BTCUSDT" || symbol == "ETHUSDT") ? 3 : (symbol == "BNBUSDT" ? 2 : 0);
        char qty_buf[32]; char fmt[16];
        std::snprintf(fmt, sizeof(fmt), "%%.%df", precision);
        std::snprintf(qty_buf, sizeof(qty_buf), fmt, static_cast<double>(qty));

        std::string payload;
        payload.reserve(256);
        payload += "apiKey="; payload += api_key;
        payload += "&quantity="; payload += qty_buf;
        payload += "&side="; payload += side;
        payload += "&symbol="; payload += symbol;
        payload += "&timestamp="; payload += std::to_string(ts_ms);
        payload += "&type=MARKET";

        std::string sig = hmac_sha256_hex(api_secret, payload);

        // Собираем итоговый JSON
        std::string json = R"({"id":")" + std::to_string(msg_id) + R"(","method":"order.place","params":{)";
        json += R"("apiKey":")" + std::string(api_key) + R"(",)";
        json += R"("quantity":")" + std::string(qty_buf) + R"(",)";
        json += R"("side":")" + std::string(side) + R"(",)";
        json += R"("symbol":")" + std::string(symbol) + R"(",)";
        json += R"("timestamp":)" + std::to_string(ts_ms) + R"(,)";
        json += R"("type":"MARKET",)";
        json += R"("signature":")" + sig + R"("}})";
        return json;
    }
}

class ExecutionEngine final {
public:
    ExecutionEngine(std::string_view api_key, std::string_view api_secret, float max_position_usd, float order_size_usd) noexcept
        : api_key_{api_key}, api_secret_{api_secret}, max_position_usd_{max_position_usd}, order_size_usd_{order_size_usd}, ioc_{1} {}

    ~ExecutionEngine() noexcept { stop(); }

    void start() {
        shutdown_.store(false, std::memory_order_release);
        // Запускаем фоновый поток, который будет крутить корутины Asio
        worker_ = std::thread([this]() {
            boost::asio::co_spawn(ioc_, ws_session(), boost::asio::detached);
            ioc_.run();
        });
    }

    void stop() noexcept {
        shutdown_.store(true, std::memory_order_release);
        ioc_.stop();
        if (worker_.joinable()) worker_.join();
    }

    void submit(const RoutedEdge& edge, float mid_src, float mid_dst) noexcept {
        if (shutdown_.load(std::memory_order_relaxed)) return;

        // Лимиты по позициям
        const float cur = positions_[edge.src_instrument].net_qty.load(std::memory_order_relaxed);
        if (std::abs(cur) * mid_src > max_position_usd_) return;

        const float qty_src = order_size_usd_ / (mid_src > 0.0F ? mid_src : 1.0F);
        const float qty_dst = order_size_usd_ / (mid_dst > 0.0F ? mid_dst : 1.0F);
        const bool  long_s  = edge.harmonic_flow > 0.0F;

        const std::uint64_t ts_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

        // Формируем JSON ордера на покупку
        std::uint64_t id1 = msg_id_.fetch_add(1, std::memory_order_relaxed);
        std::string msg_src = detail::build_ws_order_msg(id1, k_symbols_upper[edge.src_instrument], long_s ? "BUY" : "SELL", qty_src, api_key_, api_secret_, ts_ms);

        // Формируем JSON ордера на продажу
        std::uint64_t id2 = msg_id_.fetch_add(1, std::memory_order_relaxed);
        std::string msg_dst = detail::build_ws_order_msg(id2, k_symbols_upper[edge.dst_instrument], long_s ? "SELL" : "BUY", qty_dst, api_key_, api_secret_, ts_ms + 1U);

        metrics_.orders_submitted.fetch_add(2U, std::memory_order_relaxed);

        // Асинхронно кидаем в очередь (Zero Blocking для GPU!)
        boost::asio::post(ioc_, [this, m1 = std::move(msg_src), m2 = std::move(msg_dst)]() mutable {
            write_queue_.push_back(std::move(m1));
            write_queue_.push_back(std::move(m2));
            if (!is_writing_ && ws_connected_) do_write();
        });

        // Считаем Paper PnL
        update_paper_position(edge.src_instrument, qty_src, mid_src, long_s);
        update_paper_position(edge.dst_instrument, qty_dst, mid_dst, !long_s);
    }

    [[nodiscard]] const ExecutionMetrics& metrics() const noexcept { return metrics_; }

    void print_risk_summary() const noexcept {
        std::printf("  ── Execution Engine Risk Summary ──\n");
        std::printf("  Orders submitted : %llu\n", (unsigned long long)metrics_.orders_submitted.load());
        std::printf("  Orders filled    : %llu\n", (unsigned long long)metrics_.orders_filled.load());
        std::printf("  Orders rejected  : %llu\n", (unsigned long long)metrics_.orders_rejected.load());
        std::printf("  WS disconnects   : %llu\n", (unsigned long long)metrics_.http_errors.load());
        std::printf("  Realized PnL     : %.4f USD\n", static_cast<double>(metrics_.total_pnl.load()));
    }

private:
    // Главная корутина: поддерживает соединение с биржей
    boost::asio::awaitable<void> ws_session() {
        namespace beast = boost::beast; namespace websocket = beast::websocket;
        namespace asio = boost::asio; namespace ssl = asio::ssl;
        auto executor = co_await asio::this_coro::executor;

        while (!shutdown_.load(std::memory_order_relaxed)) {
            try {
                ssl::context ssl_ctx{ssl::context::tlsv12_client};
                ssl_ctx.set_default_verify_paths();
                asio::ip::tcp::resolver resolver{executor};
                auto const results = co_await resolver.async_resolve(std::string{k_binance_futures_ws_host}, std::string{k_binance_futures_ws_port}, asio::use_awaitable);

                ws_ = std::make_unique<websocket::stream<beast::ssl_stream<beast::tcp_stream>>>(executor, ssl_ctx);
                if (!SSL_set_tlsext_host_name(ws_->next_layer().native_handle(), k_binance_futures_ws_host.data())) throw std::runtime_error("SNI Failed");

                co_await beast::get_lowest_layer(*ws_).async_connect(results, asio::use_awaitable);
                co_await ws_->next_layer().async_handshake(ssl::stream_base::client, asio::use_awaitable);
                co_await ws_->async_handshake(std::string{k_binance_futures_ws_host}, std::string{k_binance_futures_ws_path}, asio::use_awaitable);

                ws_connected_ = true;
                if (!write_queue_.empty() && !is_writing_) do_write();

                // Вечный цикл чтения ответов от биржи
                beast::flat_buffer buf;
                while (!shutdown_.load(std::memory_order_relaxed)) {
                    co_await ws_->async_read(buf, asio::use_awaitable);
                    std::string_view response{static_cast<const char*>(buf.data().data()), buf.data().size()};

                    // Парсинг ответа
                    if (response.find(R"("code":)") != std::string_view::npos && response.find(R"("code":200)") == std::string_view::npos) {
                        metrics_.orders_rejected.fetch_add(1U, std::memory_order_relaxed);
                        std::printf("\n[BINANCE WS REJECT] %s\n", std::string(response).c_str());
                    } else if (response.find(R"("id":)") != std::string_view::npos) {
                        metrics_.orders_filled.fetch_add(1U, std::memory_order_relaxed);
                    }
                    buf.consume(buf.size());
                }
            } catch (...) {
                ws_connected_ = false; is_writing_ = false;
                metrics_.http_errors.fetch_add(1U, std::memory_order_relaxed);
                // Ждем 2 секунды перед реконнектом
                asio::steady_timer timer(executor, std::chrono::seconds(2));
                co_await timer.async_wait(asio::use_awaitable);
            }
        }
    }

    // Асинхронная запись в сокет (выгребает очередь)
    void do_write() {
        if (!ws_connected_ || write_queue_.empty()) { is_writing_ = false; return; }
        is_writing_ = true;
        ws_->async_write(boost::asio::buffer(write_queue_.front()), [this](boost::beast::error_code ec, std::size_t) {
            if (!ec) { write_queue_.pop_front(); do_write(); }
            else { is_writing_ = false; ws_connected_ = false; }
        });
    }

    void update_paper_position(std::uint32_t id, float qty, float price, bool is_buy) noexcept {
        if (id >= k_feed_n_instruments) return;
        auto& pos = positions_[id];
        const float sign = is_buy ? 1.0F : -1.0F;
        const float prev_qty = pos.net_qty.load(std::memory_order_relaxed);
        const float prev_ep  = pos.avg_entry_price.load(std::memory_order_relaxed);
        const float new_qty  = prev_qty + sign * qty;

        if (std::abs(new_qty) < 1e-9F) {
            const float rpnl = prev_qty * (price - prev_ep);
            pos.realized_pnl.fetch_add(rpnl, std::memory_order_relaxed);
            metrics_.total_pnl.fetch_add(rpnl, std::memory_order_relaxed);
            pos.net_qty.store(0.0F, std::memory_order_relaxed);
        } else if (prev_qty == 0.0F || (prev_qty > 0.0F) == is_buy) {
            pos.avg_entry_price.store((prev_ep * std::abs(prev_qty) + price * qty) / (std::abs(prev_qty) + qty), std::memory_order_relaxed);
            pos.net_qty.store(new_qty, std::memory_order_relaxed);
        } else {
            const float closed = std::min(std::abs(prev_qty), qty);
            const float rpnl = (is_buy ? -1.0F : 1.0F) * closed * (price - prev_ep);
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
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_{boost::asio::make_work_guard(ioc_)};
    std::thread             worker_;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::beast::tcp_stream>>> ws_;
    bool ws_connected_{false};
    bool is_writing_{false};
    std::deque<std::string> write_queue_;
    std::atomic<std::uint64_t> msg_id_{1};
    std::array<PaperPosition, k_feed_n_instruments> positions_{};
    ExecutionMetrics metrics_;
};

} // namespace holo::net