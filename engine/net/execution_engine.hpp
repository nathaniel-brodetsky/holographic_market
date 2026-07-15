#pragma once
//
// execution_engine.hpp
// holo::net — Maker-side execution engine for pairs/basket arbitrage.
//
// Sends both legs as GTX (post-only) limit orders resting at the passive
// price (best bid to buy, best ask to sell) to capture 0.00% maker fees.
// A per-pair monitor coroutine implements the legging risk manager: once
// one leg fills, the other leg gets kLegTimeout to fill naturally; past
// that, it's canceled and the residual is hedged at market.
//
// NOTE ON boost::asio::experimental::awaitable_operators:
// The exact variant shape returned by operator|| for awaitable<T> vs.
// awaitable<void> has shifted slightly across Boost releases. The pattern
// used below (`.index()` to discriminate, `std::get<0>(...)` for the
// non-void branch) matches the canonical Boost.Asio examples as of the
// 1.82–1.86 range; if your Boost version's shape differs, wait_for_status()
// and monitor_pair() are the only two functions that need adjusting.
//
#include "oms_core.hpp"

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>
#include <boost/asio/experimental/channel.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace holo::net {

namespace asio = boost::asio;

// Passive-side pricing: buy at best bid, sell at best ask — never crosses
// the spread, which is what makes the GTX (post-only) order valid.
inline double post_only_price(Side side, double best_bid, double best_ask) noexcept {
    return side == Side::Buy ? best_bid : best_ask;
}

// ---------------------------------------------------------------------------
// IOrderGateway — thin abstraction over your existing order-send transport
// (REST or the Binance WS order-entry API). Implement this against whatever
// you already use to place/cancel orders; the engine only depends on this
// interface.
// ---------------------------------------------------------------------------
class IOrderGateway {
public:
    virtual ~IOrderGateway() = default;

    // Must send timeInForce=GTX (post-only). If the venue rejects it for
    // crossing the book, that surfaces later as an ORDER_TRADE_UPDATE with
    // X=REJECTED/EXPIRED via the normal OMS update path — this call itself
    // only needs to succeed at *sending* the request.
    virtual asio::awaitable<void> send_limit_post_only(std::string_view client_order_id,
                                                        std::string_view symbol,
                                                        Side side,
                                                        double price,
                                                        double qty) = 0;

    virtual asio::awaitable<void> send_market(std::string_view client_order_id,
                                               std::string_view symbol,
                                               Side side,
                                               double qty) = 0;

    virtual asio::awaitable<void> cancel_order(std::string_view symbol,
                                                std::string_view client_order_id) = 0;
};

// ---------------------------------------------------------------------------
// ExecutionEngine
// ---------------------------------------------------------------------------
class ExecutionEngine {
public:
    struct ArbLeg {
        std::string symbol;
        Side side;
        double qty;
        double price;  // best bid (Buy) or best ask (Sell) at signal time
    };

    ExecutionEngine(asio::any_io_executor exec, OMSCore& oms, IOrderGateway& gateway)
        : exec_(exec), oms_(oms), gateway_(gateway) {
        oms_.set_update_callback([this](const OrderRecord& rec) { dispatch_update(rec); });
    }

    // Entry point called by your signal generator. Registers both legs in
    // the OMS, creates per-order waiter channels *before* sending (a fill
    // can race back on the user-data stream before send_* even returns),
    // fires both GTX orders, then hands off monitoring to a detached
    // coroutine so the signal-processing loop is not blocked on the
    // outcome of this pair.
    asio::awaitable<void> on_signal(ArbLeg leg1, ArbLeg leg2) {
        const std::string id1 = next_client_id();
        const std::string id2 = next_client_id();
        const uint64_t key1 = fnv1a64(id1);
        const uint64_t key2 = fnv1a64(id2);

        oms_.register_new_order(id1, leg1.symbol, leg1.side, leg1.price, leg1.qty);
        oms_.register_new_order(id2, leg2.symbol, leg2.side, leg2.price, leg2.qty);

        auto chan1 = get_or_create_waiter(key1);
        auto chan2 = get_or_create_waiter(key2);

        co_await gateway_.send_limit_post_only(id1, leg1.symbol, leg1.side, leg1.price, leg1.qty);
        co_await gateway_.send_limit_post_only(id2, leg2.symbol, leg2.side, leg2.price, leg2.qty);

        asio::co_spawn(exec_,
                        monitor_pair(id1, leg1, chan1, id2, leg2, chan2),
                        asio::detached);
    }

private:
    static constexpr auto kLegTimeout = std::chrono::milliseconds(50);
    using Chan = asio::experimental::channel<void(boost::system::error_code, OrderRecord)>;

    // -- Legging risk manager -------------------------------------------------
    asio::awaitable<void> monitor_pair(std::string id1, ArbLeg leg1, std::shared_ptr<Chan> chan1,
                                        std::string id2, ArbLeg leg2, std::shared_ptr<Chan> chan2) {
        using namespace boost::asio::experimental::awaitable_operators;

        const uint64_t key1 = fnv1a64(id1);
        const uint64_t key2 = fnv1a64(id2);

        // Wait for either leg to reach a terminal state (Filled, or
        // Canceled/Rejected e.g. a GTX post-only reject).
        auto first = co_await (wait_for_terminal_or_filled(chan1) ||
                                wait_for_terminal_or_filled(chan2));

        OrderRecord first_rec;
        std::shared_ptr<Chan> resting_chan;
        std::string resting_id;
        ArbLeg resting_leg;
        uint64_t resting_key;

        if (first.index() == 0) {
            first_rec = std::get<0>(first);
            remove_waiter(key1);
            resting_chan = chan2; resting_id = id2; resting_leg = leg2; resting_key = key2;
        } else {
            first_rec = std::get<1>(first);
            remove_waiter(key2);
            resting_chan = chan1; resting_id = id1; resting_leg = leg1; resting_key = key1;
        }

        if (first_rec.status != OrderStatus::Filled) {
            // First leg to settle was Canceled/Rejected, not Filled — no
            // exposure was ever taken, nothing to hedge. Just clean up.
            remove_waiter(resting_key);
            co_return;
        }

        // One leg is confirmed filled. Give the other leg kLegTimeout to
        // fill naturally before intervening.
        asio::steady_timer timer(exec_);
        timer.expires_after(kLegTimeout);

        auto race = co_await (wait_for_terminal_or_filled(resting_chan) ||
                               timer.async_wait(asio::use_awaitable));
        remove_waiter(resting_key);

        if (race.index() == 0) {
            const OrderRecord resting_rec = std::get<0>(race);
            if (resting_rec.status == OrderStatus::Filled) {
                co_return;  // both legs filled — arb complete, maker fees on both sides
            }
            // else: resting leg was Canceled/Rejected on its own (e.g. GTX
            // reject) — fall through and hedge the exposure from leg 1.
        }
        // else: 50ms elapsed with the resting leg still live — intervene.

        auto rec = oms_.get(resting_key);
        if (!rec) co_return;  // shouldn't happen, but nothing to act on

        const double remaining = rec->remaining_qty();

        if (is_live(rec->status)) {
            co_await gateway_.cancel_order(resting_leg.symbol, resting_id);
        }

        if (remaining > 1e-12) {
            const std::string hedge_id = next_client_id();
            oms_.register_new_order(hedge_id, resting_leg.symbol, resting_leg.side, 0.0, remaining);
            co_await gateway_.send_market(hedge_id, resting_leg.symbol, resting_leg.side, remaining);
        }
    }

    // Suspends until the given order's channel delivers a terminal update
    // (Filled, Canceled, or Rejected).
    asio::awaitable<OrderRecord> wait_for_terminal_or_filled(std::shared_ptr<Chan> chan) {
        for (;;) {
            auto rec = co_await chan->async_receive(asio::use_awaitable);
            if (is_terminal(rec.status)) co_return rec;
        }
    }

    // -- Waiter registry --------------------------------------------------
    // One-shot pub/sub: each in-flight order gets its own channel so that
    // multiple concurrently-monitored pairs never race over a shared
    // stream. dispatch_update() is invoked from OMSCore's update callback,
    // which — since updates originate from UserDataFeed's read loop — may
    // run on a different strand/thread than the monitor coroutines; the
    // registry is therefore mutex-protected.
    std::shared_ptr<Chan> get_or_create_waiter(uint64_t key) {
        std::lock_guard lk(waiters_mtx_);
        auto [it, inserted] = waiters_.try_emplace(key, nullptr);
        if (inserted) it->second = std::make_shared<Chan>(exec_, /*max_buffer=*/4);
        return it->second;
    }

    void remove_waiter(uint64_t key) {
        std::lock_guard lk(waiters_mtx_);
        waiters_.erase(key);
    }

    void dispatch_update(const OrderRecord& rec) {
        const uint64_t key = fnv1a64(view(rec.client_order_id));
        std::shared_ptr<Chan> chan;
        {
            std::lock_guard lk(waiters_mtx_);
            auto it = waiters_.find(key);
            if (it != waiters_.end()) chan = it->second;
        }
        // Non-blocking: a full channel silently drops (max_buffer=4 is far
        // more than one order should ever need); this callback must never
        // block the caller (the UserDataFeed read loop).
        if (chan) chan->try_send(boost::system::error_code{}, rec);
    }

    std::string next_client_id() {
        return "holo-" + std::to_string(seq_.fetch_add(1, std::memory_order_relaxed));
    }

    asio::any_io_executor exec_;
    OMSCore& oms_;
    IOrderGateway& gateway_;
    std::atomic<uint64_t> seq_{0};

    std::mutex waiters_mtx_;
    std::unordered_map<uint64_t, std::shared_ptr<Chan>> waiters_;
};

}  // namespace holo::net
