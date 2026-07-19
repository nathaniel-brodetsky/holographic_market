#include <core/memory_arena.hpp>
#include <core/lockfree_ring_buffer.hpp>
#include <core/lob_core.hpp>
#include <math/cuda_pipeline.cuh>
#include <math/cuda_utils.cuh>
#include <net/binance_feed.hpp>
#include <net/signal_router.hpp>
#include <net/oms_core.hpp>
#include <net/user_data_feed.hpp>
#include <net/binance_gateway.hpp>
#include <net/execution_engine.hpp>

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>

using namespace holo;
using namespace holo::cuda;
using namespace holo::net;

// Функция для получения ListenKey (нужен для User Data Stream)
std::string fetch_listen_key(const std::string& api_key) {
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace asio = boost::asio;
    namespace ssl = asio::ssl;

    asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    ctx.set_default_verify_paths();

    asio::ip::tcp::resolver resolver(ioc);
    beast::ssl_stream<beast::tcp_stream> stream(ioc, ctx);

    auto const results = resolver.resolve("testnet.binancefuture.com", "443");
    beast::get_lowest_layer(stream).connect(results);

    if(!SSL_set_tlsext_host_name(stream.native_handle(), "testnet.binancefuture.com")) {
        throw std::runtime_error("SNI Failed");
    }

    stream.handshake(ssl::stream_base::client);

    http::request<http::empty_body> req{http::verb::post, "/fapi/v1/listenKey", 11};
    req.set(http::field::host, "testnet.binancefuture.com");
    req.set("X-MBX-APIKEY", api_key);

    http::write(stream, req);

    beast::flat_buffer buffer;
    http::response<http::string_body> res;
    http::read(stream, buffer, res);

    beast::error_code ec;
    stream.shutdown(ec);

    std::string body = res.body();
    size_t pos = body.find(R"("listenKey":")");
    if(pos != std::string::npos) {
        pos += 13;
        return body.substr(pos, body.find("\"", pos) - pos);
    }
    throw std::runtime_error("Failed to get listenKey. Response: " + body);
}

int main() {
    // --- 1. CONFIGURATION ---
    // Credentials must never live in source (they end up in git history and
    // in every build artifact). Set these in the environment before
    // starting the process, e.g.:
    //   export HOLO_API_KEY=...
    //   export HOLO_API_SECRET=...
    const char* env_key = std::getenv("HOLO_API_KEY");
    const char* env_secret = std::getenv("HOLO_API_SECRET");
    if (!env_key || !env_secret) {
        std::cerr << "[FATAL] HOLO_API_KEY and HOLO_API_SECRET environment variables must be "
                     "set (do not hardcode credentials in source).\n";
        return 1;
    }
    std::string api_key = env_key;
    std::string api_secret = env_secret;

    const double order_size_usd = 100.0;
    const double max_position_usd = 100'000.0;

    std::cout << "========================================================\n";
    std::cout << "  HOLOGRAPHIC MARKET ARCHITECTURE  v2.0 (Maker OMS)\n";
    std::cout << "========================================================\n";

    // --- 2. MEMORY & DATA STRUCTURES ---
    MemoryArena arena{256U * 1024U * 1024U};
    auto* ring_ptr = arena.emplace<DynamicSpscRingBuffer<holo::core::LobUpdate>>(arena, 1U << 17U);
    auto* lob_ptr = arena.emplace<holo::core::LobSoA>(arena, k_feed_n_instruments, 1U);

    // --- 3. BINANCE MARKET DATA FEED ---
    BinanceFeedHandler feed{*ring_ptr};
    feed.start();
    std::cout << "[INFO] Market data feed started. Warming up (2s)...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));

    holo::core::LobUpdate u{};
    while (ring_ptr->try_pop(u)) {
        lob_ptr->apply(u);
    }

    // --- 4. CUDA PIPELINE & ROUTER ---
    CudaPipeline pipeline{*lob_ptr, arena, 0};
    SignalRouter router{k_feed_n_instruments};

    // --- 5. ASIO & OMS INITIALIZATION ---
    boost::asio::io_context ioc;
    boost::asio::ssl::context ssl_ctx(boost::asio::ssl::context::tlsv12_client);

    std::string listen_key;
    try {
        listen_key = fetch_listen_key(api_key);
        std::cout << "[INFO] ListenKey acquired.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Could not fetch ListenKey: " << e.what() << "\n";
        return 1;
    }

    OMSCore oms;

    // UserDataFeed (Слушает исполнение ордеров по WebSocket)
    UserDataFeed ud_feed(ioc.get_executor(), ssl_ctx, "testnet.binancefuture.com", listen_key, oms);
    ud_feed.start();

    // Шлюз отправки ордеров по WebSocket
    BinanceGateway gateway(ioc.get_executor(), ssl_ctx, api_key, api_secret, oms);
    gateway.start();

    // Мозг: Execution Engine
    ExecutionEngine exec(ioc.get_executor(), oms, gateway);
    exec.start_stale_order_sweeper();

    // --- 6. BACKGROUND THREADS ---
    std::thread asio_thread([&ioc]() {
        auto work_guard = boost::asio::make_work_guard(ioc);
        ioc.run();
    });

    std::atomic<bool> shutdown{false};
    std::thread pipeline_thread{[&]() { pipeline.run_continuous(shutdown); }};
    std::thread drain_thread{[&]() {
        while(!shutdown.load(std::memory_order_relaxed)) {
            while (ring_ptr->try_pop(u)) {
                lob_ptr->apply(u);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }};

    // Фоновый поток для логирования профита раз в 5 секунд
    std::thread pnl_logger_thread{[&]() {
        while(!shutdown.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            double total = exec.pnl().total_pnl([&](std::string_view sym) {
                if (sym == "BTCUSDT") return lob_ptr->mid_price(0);
                if (sym == "ETHUSDT") return lob_ptr->mid_price(1);
                if (sym == "SOLUSDT") return lob_ptr->mid_price(2);
                if (sym == "BNBUSDT") return lob_ptr->mid_price(3);
                return 0.0f;
            });
            std::cout << "\n[PNL] Live MTM PnL (Realized + Unrealized): " << std::fixed << std::setprecision(4) << total << " USD | Live Orders in Book: " << oms.live_order_count() << "\n";
        }
    }};

    std::cout << "[INFO] HFT Maker Engine is LIVE and hunting for arbitrage...\n";
    std::uint64_t last_sig_ts = 0;

    // --- 7. MAIN EVENT LOOP ---
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));

        auto sig = pipeline.last_signal();
        if (sig.timestamp_ns != last_sig_ts && sig.timestamp_ns != 0) {
            last_sig_ts = sig.timestamp_ns;

            auto topo = pipeline.last_topology();
            if (topo.n_edges > 0) {
                SignalRouter::TopKBuffer edges{};
                size_t n_routed = router.route(
                    sig,
                    std::span<const float>{topo.harmonic_flow, (size_t)topo.n_edges},
                    std::span<const int>{topo.edge_src, (size_t)topo.n_edges},
                    std::span<const int>{topo.edge_dst, (size_t)topo.n_edges},
                    edges
                );

                // If the engine is halted (circuit breaker) or already at
                // its concurrent-order ceiling, on_signal() would drop
                // every one of these anyway (and already logs a throttled
                // reason for it). Skip the per-edge work, the mid-price
                // lookups, and — most importantly — the "[SIGNAL] Placing
                // Maker Limits" print entirely in that case: printing
                // "Placing" for something that is guaranteed to be dropped
                // is exactly what made the earlier logs look like orders
                // were going out while the whole engine was blocked.
                if (exec.is_halted() || exec.is_at_capacity()) {
                    continue;
                }

                for (size_t i = 0; i < n_routed; ++i) {
                    const auto& edge = edges[i];
                    bool is_long = edge.harmonic_flow > 0.0f;

                    std::string sym1 = std::string(k_symbols_upper[edge.src_instrument]);
                    std::string sym2 = std::string(k_symbols_upper[edge.dst_instrument]);

                    // Проверяем лимиты риска из PnlBook от Claude
                    double cur_pos1_qty = std::abs(exec.pnl().snapshot(sym1).qty);
                    if (cur_pos1_qty * lob_ptr->mid_price(edge.src_instrument) > max_position_usd) continue;

                    // Нога 1 (Лимитка в спред)
                    ExecutionEngine::ArbLeg leg1{
                        sym1,
                        is_long ? holo::net::Side::Buy : holo::net::Side::Sell,
                        order_size_usd / lob_ptr->mid_price(edge.src_instrument),
                        is_long ? lob_ptr->best_bid(edge.src_instrument) : lob_ptr->best_ask(edge.src_instrument)
                    };

                    // Нога 2 (Зеркальная лимитка)
                    ExecutionEngine::ArbLeg leg2{
                        sym2,
                        is_long ? holo::net::Side::Sell : holo::net::Side::Buy,
                        order_size_usd / lob_ptr->mid_price(edge.dst_instrument),
                        is_long ? lob_ptr->best_ask(edge.dst_instrument) : lob_ptr->best_bid(edge.dst_instrument)
                    };

                    std::cout << "[SIGNAL] S_YM=" << sig.yang_mills_action << " | Placing Maker Limits for " << leg1.symbol << " & " << leg2.symbol << "\n";

                    // Запускаем корутину
                    boost::asio::co_spawn(ioc, exec.on_signal(leg1, leg2), boost::asio::detached);
                }
            }
        }
    }

    shutdown = true;
    pipeline_thread.join();
    drain_thread.join();
    pnl_logger_thread.join();
    ioc.stop();
    asio_thread.join();

    return 0;
}