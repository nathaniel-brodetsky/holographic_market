#pragma once
//
// execution_engine.hpp
// holo::net — Maker-side execution engine for pairs/basket arbitrage.
//
// Sends both legs as GTX (post-only) limit orders resting at the passive
// price (best bid to buy, best ask to sell) to capture 0.00% maker fees.
// A per-pair monitor coroutine implements the legging risk manager: once
// one leg fills, the other leg gets kLegTimeout to fill naturally; past
// that, it's canceled and the residual is hedged at market. Paper PnL is
// tracked per-symbol with correct weighted-average-cost + position-flip
// (reversal) accounting.
//
// ---------------------------------------------------------------------------
// ARCHITECTURAL NOTES — race conditions this file explicitly defends against
// ---------------------------------------------------------------------------
// (1) Fill-before-send race: the per-order waiter channel is created and
//     registered in `waiters_` *before* the order is sent. Binance can
//     deliver an ORDER_TRADE_UPDATE on the user-data stream before
//     send_limit_post_only()'s awaitable even resumes; if the channel
//     didn't exist yet, that update would be silently dropped.
//
// (2) Cross-thread producer/consumer on the channel: dispatch_update() is
//     invoked from OMSCore's update callback, which fires on whatever
//     thread called OMSCore::apply_update() — i.e. UserDataFeed's read
//     loop / io_context thread — while wait_for_terminal_or_filled()
//     co_awaits async_receive() on the ExecutionEngine's own executor.
//     boost::asio::experimental::channel (the non-concurrent variant) is
//     documented as NOT safe for concurrent send/receive from different
//     threads/strands. We therefore use
//     `boost::asio::experimental::concurrent_channel`, which internally
//     synchronizes its op queue and is the correct primitive here. (If
//     you later guarantee UserDataFeed and ExecutionEngine always share
//     one io_context thread, the plain `channel` would also be safe and
//     slightly cheaper — but that's a fragile invariant to lean on.)
//
// (3) try_send() on a full channel silently drops the message by design
//     (non-blocking). A dropped *Filled* event is dangerous here: it means
//     the risk manager never learns a leg filled and can double-hedge or
//     mis-time the leg timeout. We check try_send()'s return and log loudly
//     on drop; max_buffer is sized generously (16) since a single order's
//     realistic transition sequence (PendingNew ->) Open -> [Partial]* ->
//     terminal is short, but this is still a "notice and fix your buffer
//     size," not a silent-failure design.
//
// (4) Cancel/fill race on the resting leg: after the 50ms leg timeout
//     fires, we send cancel_order() — but the exchange may fill the
//     resting leg in the same instant the cancel arrives (a cancel can
//     race a fill, or simply be rejected because the order already
//     filled). We do NOT trust the OMS snapshot taken before the cancel
//     was sent to compute hedge quantity. Instead we re-await the
//     channel for the *authoritative* post-cancel terminal state (bounded
//     by a short ack timeout) and compute `remaining` from that.
//
// (5) Hedge-order fills previously went untracked: nothing awaited the
//     market hedge order's terminal state, so it never fed PnL and its
//     OMSCore slot leaked forever. monitor_pair now spawns a detached
//     watcher for the hedge order exactly like a normal leg.
//
// (6) OMSCore slot lifetime: OMSCore::release() is called for every order
//     (leg or hedge) once its terminal state has been observed and PnL
//     applied, so long-running engines don't leak `index_` entries.
//
#include "oms_core.hpp"

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>
#include <boost/asio/experimental/channel.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>

#include <atomic>
#include <cmath>
#include <iostream>
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
// PnlBook — per-symbol weighted-average-cost paper PnL with correct
// position-reversal accounting. Thread-safe (fills can arrive from the
// OMS update callback on a different thread than PnL is read from).
// ---------------------------------------------------------------------------
class PnlBook {
public:
    struct Position {
        double qty{0.0};           // signed: +long, -short
        double avg_price{0.0};     // cost basis of the open position
        double realized_pnl{0.0};  // cumulative realized, this symbol
    };

    // Apply an incremental fill (delta_qty > 0, always; direction comes
    // from `side`) at `fill_price` to the named symbol's position.
    void apply_fill(std::string_view symbol, Side side, double delta_qty, double fill_price) {
        if (delta_qty <= 0.0 || fill_price <= 0.0) return;
        const double signed_delta = signed_dir(side) * delta_qty;

        std::lock_guard lk(mtx_);
        Position& pos = book_[std::string(symbol)];

        const bool opening_or_same_dir =
            pos.qty == 0.0 || (pos.qty > 0.0) == (signed_delta > 0.0);

        if (opening_or_same_dir) {
            // Weighted-average into the (possibly new) position.
            const double new_qty = pos.qty + signed_delta;
            pos.avg_price = (pos.qty == 0.0)
                                 ? fill_price
                                 : (pos.avg_price * std::abs(pos.qty) +
                                    fill_price * std::abs(signed_delta)) /
                                       std::abs(new_qty);
            pos.qty = new_qty;
            return;
        }

        // Reducing and/or reversing. First realize PnL on the portion that
        // closes the existing position.
        const double closing_qty = std::min(std::abs(signed_delta), std::abs(pos.qty));
        const double dir_sign    = pos.qty > 0.0 ? 1.0 : -1.0;
        pos.realized_pnl += dir_sign * (fill_price - pos.avg_price) * closing_qty;

        const double new_qty = pos.qty + signed_delta;
        if (new_qty == 0.0) {
            pos.avg_price = 0.0;
        } else if ((new_qty > 0.0) != (pos.qty > 0.0)) {
            // Sign flipped: the residual beyond what closed the old
            // position opens a brand-new position at the fill price.
            pos.avg_price = fill_price;
        }
        // else: partial reduce, same sign, avg_price of the remaining
        // position is unchanged (weighted-average cost is invariant under
        // partial closes).
        pos.qty = new_qty;
    }

    [[nodiscard]] Position snapshot(std::string_view symbol) const {
        std::lock_guard lk(mtx_);
        auto it = book_.find(std::string(symbol));
        return it == book_.end() ? Position{} : it->second;
    }

    // Realized + mark-to-market unrealized PnL across all symbols, given a
    // mark-price lookup (e.g. best bid/ask midpoint from your LOB book).
    template <typename MarkPriceFn>
    [[nodiscard]] double total_pnl(MarkPriceFn&& mark_price) const {
        std::lock_guard lk(mtx_);
        double total = 0.0;
        for (const auto& [sym, pos] : book_) {
            total += pos.realized_pnl;
            if (pos.qty != 0.0) {
                const double mark = mark_price(sym);
                if (mark > 0.0) total += (mark - pos.avg_price) * pos.qty;
            }
        }
        return total;
    }

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, Position> book_;
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
    // the OMS, creates per-order waiter channels *before* sending (see
    // race-condition note (1) at the top of this file), fires both GTX
    // orders, then hands off monitoring to a detached coroutine so the
    // signal-processing loop is not blocked on the outcome of this pair.
    asio::awaitable<void> on_signal(ArbLeg leg1, ArbLeg leg2) {
        const std::string id1 = next_client_id();
        const std::string id2 = next_client_id();

        oms_.register_new_order(id1, leg1.symbol, leg1.side, leg1.price, leg1.qty);
        oms_.register_new_order(id2, leg2.symbol, leg2.side, leg2.price, leg2.qty);

        auto chan1 = get_or_create_waiter(fnv1a64(id1));
        auto chan2 = get_or_create_waiter(fnv1a64(id2));

        co_await gateway_.send_limit_post_only(id1, leg1.symbol, leg1.side, leg1.price, leg1.qty);
        co_await gateway_.send_limit_post_only(id2, leg2.symbol, leg2.side, leg2.price, leg2.qty);

        asio::co_spawn(exec_,
                        monitor_pair(id1, leg1, chan1, id2, leg2, chan2),
                        [this](std::exception_ptr e) { log_exception("monitor_pair", e); });
    }

    [[nodiscard]] PnlBook& pnl() noexcept { return pnl_; }

private:
    static constexpr auto kLegTimeout   = std::chrono::milliseconds(50);
    // Bound on how long we wait for the *authoritative* post-cancel
    // terminal state before falling back to the last known OMS snapshot
    // (see race-condition note (4)). Generous relative to typical Binance
    // Futures Testnet WS-API round-trip (~10-50ms).
    static constexpr auto kCancelAckTimeout = std::chrono::milliseconds(750);

    // NOTE ON THE CHANNEL TYPE: explicitly parameterized on
    // `asio::any_io_executor` rather than relying on the single-argument
    // `channel<Signature>` alias (which *also* resolves to any_io_executor
    // via SFINAE dispatch as of Boost 1.78+ — but being explicit here
    // removes any ambiguity for readers and matches the executor type
    // ExecutionEngine itself is constructed with). We use the *concurrent*
    // variant — see race-condition note (2).
    using Chan = asio::experimental::concurrent_channel<
        asio::any_io_executor, void(boost::system::error_code, OrderRecord)>;
    static constexpr std::size_t kChanBuffer = 16;

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
        uint64_t resting_key, filled_key;

        if (first.index() == 0) {
            first_rec = std::get<0>(first);
            resting_chan = chan2; resting_id = id2; resting_leg = leg2; resting_key = key2;
            filled_key = key1;
        } else {
            first_rec = std::get<1>(first);
            resting_chan = chan1; resting_id = id1; resting_leg = leg1; resting_key = key1;
            filled_key = key2;
        }

        if (first_rec.status != OrderStatus::Filled) {
            // First leg to settle was Canceled/Rejected, not Filled — no
            // exposure was ever taken, nothing to hedge. Clean up both.
            release_terminal(filled_key);
            release_terminal(resting_key);
            co_return;
        }
        apply_fill_pnl(first_rec, first.index() == 0 ? leg1 : leg2);
        release_terminal(filled_key);

        // One leg is confirmed filled. Give the other leg kLegTimeout to
        // fill naturally before intervening.
        asio::steady_timer timer(exec_);
        timer.expires_after(kLegTimeout);

        // NOTE: `asio::cancel_after` would be the idiomatic way to express
        // this timeout directly on async_receive, but it requires Boost
        // >= 1.86; this project pins Boost 1.83, so we use the portable
        // awaitable_operators race instead:
        //   auto race = co_await (wait_for_terminal_or_filled(resting_chan) ||
        //                          timer.async_wait(asio::use_awaitable));
        auto race = co_await (wait_for_terminal_or_filled(resting_chan) ||
                               timer.async_wait(asio::use_awaitable));

        if (race.index() == 0) {
            const OrderRecord resting_rec = std::get<0>(race);
            if (resting_rec.status == OrderStatus::Filled) {
                apply_fill_pnl(resting_rec, resting_leg);
                release_terminal(resting_key);
                co_return;  // both legs filled — arb complete, maker fees on both sides
            }
            // Resting leg was Canceled/Rejected on its own (e.g. GTX
            // reject) — no exposure on that leg; fall through, but there
            // is nothing to cancel and nothing to hedge for *this* leg.
            release_terminal(resting_key);
            co_return;
        }
        // else: 50ms elapsed with the resting leg still live — intervene.

        co_await gateway_.cancel_order(resting_leg.symbol, resting_id);

        // Race-condition note (4): don't trust a pre-cancel snapshot for
        // hedge sizing. Re-await the channel for the authoritative
        // terminal state the cancel (or a concurrent fill) produces,
        // bounded by kCancelAckTimeout in case the cancel ack itself is
        // lost/delayed.
        asio::steady_timer ack_timer(exec_);
        ack_timer.expires_after(kCancelAckTimeout);
        auto post_cancel = co_await (wait_for_terminal_or_filled(resting_chan) ||
                                      ack_timer.async_wait(asio::use_awaitable));

        OrderRecord final_rec;
        if (post_cancel.index() == 0) {
            final_rec = std::get<0>(post_cancel);
        } else {
            // Cancel ack never arrived within kCancelAckTimeout — fall
            // back to the last known OMS snapshot rather than hang
            // forever. Flag loudly: this is an operational anomaly (lost
            // WS message / venue latency spike) that deserves paging, not
            // silent handling.
            std::cerr << "[ExecutionEngine] WARNING: no cancel ack for " << resting_id
                      << " within " << kCancelAckTimeout.count()
                      << "ms; using last known OMS state for hedge sizing.\n";
            auto snap = oms_.get(resting_key);
            if (!snap) { release_terminal(resting_key); co_return; }
            final_rec = *snap;
        }

        if (final_rec.status == OrderStatus::Filled) {
            apply_fill_pnl(final_rec, resting_leg);
            release_terminal(resting_key);
            co_return;  // filled before/during the cancel race — nothing to hedge
        }
        if (final_rec.filled_qty > 0.0) {
            // Partial fill before it was actually canceled/rejected.
            apply_fill_pnl(final_rec, resting_leg);
        }
        release_terminal(resting_key);

        const double remaining = final_rec.qty - final_rec.filled_qty;
        if (remaining > 1e-12) {
            const std::string hedge_id = next_client_id();
            const Side hedge_side = resting_leg.side;  // hedge in the SAME direction
                                                          // the resting leg would have
                                                          // filled, to realize the exposure
                                                          // the completed leg opened.
            oms_.register_new_order(hedge_id, resting_leg.symbol, hedge_side, 0.0, remaining);
            auto hedge_chan = get_or_create_waiter(fnv1a64(hedge_id));
            co_await gateway_.send_market(hedge_id, resting_leg.symbol, hedge_side, remaining);

            // Race-condition note (5): track the hedge order to
            // completion too, so its fill feeds PnL and its OMS slot is
            // released. Detached: on_signal()/monitor_pair() must not
            // block on this, and there is nothing further to coordinate
            // once the hedge is away.
            asio::co_spawn(
                exec_, watch_hedge(hedge_id, resting_leg.symbol, hedge_side, hedge_chan),
                [this](std::exception_ptr e) { log_exception("watch_hedge", e); });
        }
    }

    asio::awaitable<void> watch_hedge(std::string hedge_id, std::string symbol, Side side,
                                       std::shared_ptr<Chan> chan) {
        const uint64_t key = fnv1a64(hedge_id);
        auto rec = co_await wait_for_terminal_or_filled(chan);
        if (rec.filled_qty > 0.0) {
            apply_fill_pnl(rec, ArbLeg{std::move(symbol), side, rec.filled_qty, rec.avg_fill_price});
        } else if (rec.status != OrderStatus::Filled) {
            // A MARKET order that didn't fill at all (e.g. rejected for
            // insufficient margin) leaves real, unhedged exposure from
            // the leg that *did* fill. This is a critical operational
            // condition — surface it loudly rather than swallowing it.
            std::cerr << "[ExecutionEngine] CRITICAL: hedge order " << hedge_id << " for "
                      << rec.qty << " " << symbol << " did not fill (status="
                      << static_cast<int>(rec.status) << "). Manual intervention required.\n";
        }
        release_terminal(key);
    }

    // Suspends until the given order's channel delivers a terminal update
    // (Filled, Canceled, or Rejected).
    asio::awaitable<OrderRecord> wait_for_terminal_or_filled(std::shared_ptr<Chan> chan) {
        for (;;) {
            auto rec = co_await chan->async_receive(asio::use_awaitable);
            if (is_terminal(rec.status)) co_return rec;
        }
    }

    // -- PnL plumbing -------------------------------------------------------
    // Applies the *incremental* fill represented by `rec` (a fresh
    // cumulative-fill snapshot) to the PnL book. Tracks last-seen
    // cumulative filled_qty per key so repeated Partial updates only
    // charge the delta once.
    void apply_fill_pnl(const OrderRecord& rec, const ArbLeg& leg) {
        // Keyed by exchange_order_id (stable once assigned) rather than
        // client_order_id, purely so this map's key type matches what the
        // OMS update carries without an extra string hash on the fill path.
        std::lock_guard lk(last_filled_mtx_);
        double& prev = last_filled_[rec.exchange_order_id];
        const double delta = rec.filled_qty - prev;
        if (delta > 1e-12 && rec.avg_fill_price > 0.0) {
            pnl_.apply_fill(leg.symbol, leg.side, delta, rec.avg_fill_price);
        }
        prev = rec.filled_qty;
    }

    void release_terminal(uint64_t key) {
        oms_.release(key);
        std::lock_guard lk(waiters_mtx_);
        waiters_.erase(key);
    }

    // -- Waiter registry --------------------------------------------------
    // One-shot pub/sub: each in-flight order gets its own channel so that
    // multiple concurrently-monitored pairs never race over a shared
    // stream. dispatch_update() is invoked from OMSCore's update callback,
    // which — since updates originate from UserDataFeed's read loop — may
    // run on a different strand/thread than the monitor coroutines; the
    // registry is therefore mutex-protected, and the channel itself is
    // the *concurrent* variant (see race-condition note (2)).
    std::shared_ptr<Chan> get_or_create_waiter(uint64_t key) {
        std::lock_guard lk(waiters_mtx_);
        auto [it, inserted] = waiters_.try_emplace(key, nullptr);
        if (inserted) it->second = std::make_shared<Chan>(exec_, kChanBuffer);
        return it->second;
    }

    void dispatch_update(const OrderRecord& rec) {
        const uint64_t key = fnv1a64(view(rec.client_order_id));
        std::shared_ptr<Chan> chan;
        {
            std::lock_guard lk(waiters_mtx_);
            auto it = waiters_.find(key);
            if (it != waiters_.end()) chan = it->second;
        }
        if (!chan) return;  // no monitor is (or is no longer) waiting on this order

        // Non-blocking: a full channel drops the message rather than
        // blocking the OMS-callback thread — see race-condition note (3).
        if (!chan->try_send(boost::system::error_code{}, rec)) {
            std::cerr << "[ExecutionEngine] WARNING: dropped update for order "
                      << view(rec.client_order_id) << " (status="
                      << static_cast<int>(rec.status)
                      << ") — channel buffer full. Increase kChanBuffer.\n";
        }
    }

    void log_exception(const char* where, std::exception_ptr e) {
        if (!e) return;
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& ex) {
            std::cerr << "[ExecutionEngine] " << where << " threw: " << ex.what() << "\n";
        }
    }

    std::string next_client_id() {
        return "holo-" + std::to_string(seq_.fetch_add(1, std::memory_order_relaxed));
    }

    asio::any_io_executor exec_;
    OMSCore& oms_;
    IOrderGateway& gateway_;
    PnlBook pnl_;
    std::atomic<uint64_t> seq_{0};

    std::mutex waiters_mtx_;
    std::unordered_map<uint64_t, std::shared_ptr<Chan>> waiters_;

    std::mutex last_filled_mtx_;
    std::unordered_map<int64_t, double> last_filled_;  // keyed by exchange_order_id
};

}  // namespace holo::net