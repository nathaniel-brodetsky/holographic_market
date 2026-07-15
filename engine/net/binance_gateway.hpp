#pragma once
#include <net/execution_engine.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <deque>
#include <string>
#include <iostream>

namespace holo::net {
class BinanceGateway final : public IOrderGateway {
public:
    BinanceGateway(boost::asio::any_io_executor exec, boost::asio::ssl::context& ctx, std::string key, std::string sec)
        : exec_(exec), ws_(exec, ctx), api_key_(std::move(key)), api_secret_(std::move(sec)) {}

    void start() {
        boost::asio::co_spawn(exec_, run(), boost::asio::detached);
    }

    boost::asio::awaitable<void> send_limit_post_only(std::string_view cid, std::string_view sym, Side side, double price, double qty) override {
        send_msg(build_msg(cid, sym, side == Side::Buy ? "BUY" : "SELL", "LIMIT", qty, price, "GTX"));
        co_return;
    }

    boost::asio::awaitable<void> send_market(std::string_view cid, std::string_view sym, Side side, double qty) override {
        send_msg(build_msg(cid, sym, side == Side::Buy ? "BUY" : "SELL", "MARKET", qty, 0.0, ""));
        co_return;
    }

    boost::asio::awaitable<void> cancel_order(std::string_view sym, std::string_view cid) override {
        const uint64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::string payload = "apiKey=" + api_key_ + "&origClientOrderId=" + std::string(cid) + "&symbol=" + std::string(sym) + "&timestamp=" + std::to_string(ts);
        std::string sig = detail::hmac_sha256_hex(api_secret_, payload);
        std::string json = R"({"id":")" + std::to_string(++msg_id_) + R"(","method":"order.cancel","params":{)";
        json += R"("apiKey":")" + api_key_ + R"(", "origClientOrderId":")" + std::string(cid) + R"(", "symbol":")" + std::string(sym) + R"(", "timestamp":)" + std::to_string(ts) + R"(, "signature":")" + sig + R"("}})";
        send_msg(json);
        co_return;
    }

private:
    boost::asio::awaitable<void> run() {
        namespace asio = boost::asio; namespace beast = boost::beast;
        asio::ip::tcp::resolver resolver(exec_);
        auto const results = co_await resolver.async_resolve("testnet.binancefuture.com", "443", asio::use_awaitable);
        if (!SSL_set_tlsext_host_name(ws_.next_layer().native_handle(), "testnet.binancefuture.com")) co_return;
        co_await beast::get_lowest_layer(ws_).async_connect(results, asio::use_awaitable);
        co_await ws_.next_layer().async_handshake(asio::ssl::stream_base::client, asio::use_awaitable);
        co_await ws_.async_handshake("testnet.binancefuture.com", "/ws-fapi/v1", asio::use_awaitable);
        connected_ = true;
        if (!queue_.empty()) do_write();
        beast::flat_buffer buf;
        while(true) {
            co_await ws_.async_read(buf, asio::use_awaitable);
            buf.consume(buf.size()); // We just ignore acks, UserDataFeed handles fills
        }
    }

    void send_msg(std::string msg) {
        boost::asio::post(exec_, [this, m = std::move(msg)]() mutable {
            queue_.push_back(std::move(m));
            if (connected_ && !writing_) do_write();
        });
    }

    void do_write() {
        if (queue_.empty() || !connected_) { writing_ = false; return; }
        writing_ = true;
        ws_.async_write(boost::asio::buffer(queue_.front()), [this](boost::beast::error_code ec, std::size_t) {
            if(!ec) { queue_.pop_front(); do_write(); } else { writing_ = false; connected_ = false; }
        });
    }

    std::string build_msg(std::string_view cid, std::string_view sym, std::string_view side, std::string_view type, double qty, double price, std::string_view tif) {
        const uint64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        int q_prec = (sym == "BTCUSDT" || sym == "ETHUSDT") ? 3 : (sym == "BNBUSDT" ? 2 : 0);
        char q_buf[32]; char q_fmt[16]; std::snprintf(q_fmt, sizeof(q_fmt), "%%.%df", q_prec); std::snprintf(q_buf, sizeof(q_buf), q_fmt, qty);

        std::string payload = "apiKey=" + api_key_ + "&newClientOrderId=" + std::string(cid);
        if (price > 0) {
            int p_prec = (sym == "BTCUSDT") ? 1 : (sym == "SOLUSDT" ? 4 : 2);
            char p_buf[32]; char p_fmt[16]; std::snprintf(p_fmt, sizeof(p_fmt), "%%.%df", p_prec); std::snprintf(p_buf, sizeof(p_buf), p_fmt, price);
            payload += "&price=" + std::string(p_buf);
        }
        payload += "&quantity=" + std::string(q_buf) + "&side=" + std::string(side) + "&symbol=" + std::string(sym);
        if (!tif.empty()) payload += "&timeInForce=" + std::string(tif);
        payload += "&timestamp=" + std::to_string(ts) + "&type=" + std::string(type);
        std::string sig = detail::hmac_sha256_hex(api_secret_, payload);

        std::string json = R"({"id":")" + std::to_string(++msg_id_) + R"(","method":"order.place","params":{)";
        json += R"("apiKey":")" + api_key_ + R"(","newClientOrderId":")" + std::string(cid) + R"(",)";
        if (price > 0) {
            int p_prec = (sym == "BTCUSDT") ? 1 : (sym == "SOLUSDT" ? 4 : 2);
            char p_buf[32]; char p_fmt[16]; std::snprintf(p_fmt, sizeof(p_fmt), "%%.%df", p_prec); std::snprintf(p_buf, sizeof(p_buf), p_fmt, price);
            json += R"("price":")" + std::string(p_buf) + R"(",)";
        }
        json += R"("quantity":")" + std::string(q_buf) + R"(","side":")" + std::string(side) + R"(","symbol":")" + std::string(sym) + R"(",)";
        if (!tif.empty()) json += R"("timeInForce":")" + std::string(tif) + R"(",)";
        json += R"("timestamp":)" + std::to_string(ts) + R"(,"type":")" + std::string(type) + R"(","signature":")" + sig + R"("}})";
        return json;
    }

    boost::asio::any_io_executor exec_;
    boost::beast::websocket::stream<boost::beast::ssl_stream<boost::asio::ip::tcp::socket>> ws_;
    std::string api_key_, api_secret_;
    std::deque<std::string> queue_;
    bool connected_{false}, writing_{false};
    uint64_t msg_id_{0};
};
}