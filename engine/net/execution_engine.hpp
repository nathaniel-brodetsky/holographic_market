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
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

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

    // Fees are a straight debit against realized PnL for that symbol —
    // there is no "position" concept for a fee, just a cost.
    void apply_commission(std::string_view symbol, double commission) {
        if (commission <= 0.0) return;
        std::lock_guard lk(mtx_);
        book_[std::string(symbol)].realized_pnl -= commission;
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

    ExecutionEngine(asio::any_io_executor exec, OMSCore& oms, IOrderGateway& gateway,
                     std::size_t max_open_orders = 40)
        : exec_(exec), oms_(oms), gateway_(gateway), max_open_orders_(max_open_orders) {
        oms_.set_update_callback([this](const OrderRecord& rec) { dispatch_update(rec); });
    }

    // -- Circuit breaker ----------------------------------------------------
    // Trips (halts all new signal processing) after kMaxConsecutiveRejects
    // Rejected orders in a row with no successful Open/Filled in between —
    // e.g. a stuck margin/precision/rate-limit condition that would
    // otherwise cause on_signal() to keep firing orders that all bounce.
    // Does NOT auto-clear: a human must look at *why* before re-arming.
    [[nodiscard]] bool is_halted() const noexcept { return halted_.load(std::memory_order_acquire); }
    void reset_halt() {
        halted_.store(false, std::memory_order_release);
        consecutive_rejects_.store(0, std::memory_order_relaxed);
        std::remove("holo_halt.flag");
        std::cerr << "[ExecutionEngine] circuit breaker manually reset — resuming signal processing.\n";
    }

    // Entry point called by your signal generator. Registers both legs in
    // the OMS, creates per-order waiter channels *before* sending (see
    // race-condition note (1) at the top of this file), fires both GTX
    // orders, then hands off monitoring to a detached coroutine so the
    // signal-processing loop is not blocked on the outcome of this pair.
    //
    // Two independent guards run here before a single order is sent:
    //  - the circuit breaker (halted_) — tripped by a burst of rejects;
    //  - max_open_orders_ — a hard ceiling on concurrently-live orders,
    //    independent of *why* they're live. This is what actually stops
    //    unbounded order-flood scenarios (e.g. UserDataFeed down so
    //    nothing ever reaches a terminal state, or a signal generator
    //    firing faster than legs resolve) rather than relying on the
    //    exchange's rate limiter to push back for you with 429s.
    asio::awaitable<void> on_signal(ArbLeg leg1, ArbLeg leg2) {
        if (halted_.load(std::memory_order_acquire)) {
            throttled_log(last_halt_log_ns_, halt_drops_suppressed_,
                          "circuit breaker is TRIPPED (call reset_halt() after investigating)");
            co_return;
        }
        const std::size_t live = oms_.live_order_count();
        if (live >= max_open_orders_) {
            throttled_log(last_capacity_log_ns_, capacity_drops_suppressed_,
                          std::to_string(live) + " live orders >= max_open_orders_ (" +
                              std::to_string(max_open_orders_) +
                              "). Feed may be stalled (orders never reaching a terminal state) "
                              "or signal rate exceeds fill/cancel rate");
            co_return;
        }

        // Root-cause guard for signal-flood exhaustion: the router can (and
        // in practice does) re-select the *same* edge on back-to-back
        // ticks faster than a previous pair for that same symbol
        // combination has resolved. Without this check, a single
        // persistent edge floods max_open_orders_ and/or the reject
        // counter within milliseconds, well before monitor_pair() for the
        // first instance even gets a chance to finish. One in-flight
        // arb per symbol pair at a time; the next signal for that pair is
        // simply skipped until the current one reaches a terminal state.
        const std::string pair_key = leg1.symbol + "|" + leg2.symbol;
        {
            std::lock_guard lk(inflight_mtx_);
            if (!inflight_pairs_.insert(pair_key).second) {
                throttled_log(last_dup_log_ns_, dup_drops_suppressed_,
                              "duplicate signal for " + pair_key +
                                  " while a previous pair on it is still in flight");
                co_return;
            }
        }
        // Guarantees the pair_key entry is removed if anything between here
        // and the co_spawn below throws (e.g. the gateway send failing) —
        // without this, an exception would leave that symbol pair
        // permanently stuck in inflight_pairs_, silently blocking it from
        // ever trading again for the rest of the process's life. Disarmed
        // right before co_spawn, at which point the spawned coroutine's
        // completion handler takes over cleanup responsibility instead.
        InflightGuard guard{*this, pair_key};

        const std::string id1 = next_client_id();
        const std::string id2 = next_client_id();

        oms_.register_new_order(id1, leg1.symbol, leg1.side, leg1.price, leg1.qty);
        oms_.register_new_order(id2, leg2.symbol, leg2.side, leg2.price, leg2.qty);

        auto chan1 = get_or_create_waiter(fnv1a64(id1));
        auto chan2 = get_or_create_waiter(fnv1a64(id2));

        co_await gateway_.send_limit_post_only(id1, leg1.symbol, leg1.side, leg1.price, leg1.qty);
        co_await gateway_.send_limit_post_only(id2, leg2.symbol, leg2.side, leg2.price, leg2.qty);

        guard.disarm();
        asio::co_spawn(exec_,
                        monitor_pair(id1, leg1, chan1, id2, leg2, chan2),
                        [this, pair_key](std::exception_ptr e) {
                            log_exception("monitor_pair", e);
                            clear_inflight(pair_key);
                        });
    }

    // Lets the signal-generation loop skip printing/spawning altogether for
    // signals it already knows on_signal() would drop — cuts log spam and
    // wasted coroutine spawns at the source, rather than only suppressing
    // the log line after the fact.
    [[nodiscard]] bool is_at_capacity() const noexcept {
        return oms_.live_order_count() >= max_open_orders_;
    }

    [[nodiscard]] PnlBook& pnl() noexcept { return pnl_; }

    // "Уборщик" / stale-order sweeper: periodically cancels resting orders
    // older than max_age that haven't reached a terminal state — the
    // scenario monitor_pair() does NOT cover on its own, because its
    // kLegTimeout logic only engages once one leg of a pair has already
    // filled. A pair where *neither* leg ever fills (price ran away from
    // both GTX prices) would otherwise rest forever, eventually exhausting
    // max_open_orders_ and freezing margin against orders with no chance
    // of filling at their original (now stale) price.
    //
    // Call this once after construction, e.g.:
    //   exec.start_stale_order_sweeper();
    // Safe to leave running for the process lifetime; cancels are cheap
    // and idempotent (Binance just errors a second cancel on an
    // already-terminal order, which we ignore).
    void start_stale_order_sweeper(std::chrono::seconds sweep_interval = std::chrono::seconds(10),
                                    std::chrono::nanoseconds max_age = std::chrono::seconds(10)) {
        asio::co_spawn(exec_, sweeper_loop(sweep_interval, max_age),
                        [this](std::exception_ptr e) { log_exception("sweeper_loop", e); });
    }

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
            // First leg to settle was Canceled/Rejected — but that does
            // NOT guarantee the partner leg has zero exposure: the
            // partner could fill (or be filling) independently at the
            // exact same moment, especially now that the stale-order
            // sweeper (see start_stale_order_sweeper()) can cancel either
            // leg of a pair on its own schedule. The pair no longer has a
            // reason to exist either way, so cancel the partner and
            // check its *authoritative* outcome before deciding whether
            // there's anything to hedge — the same pattern used below for
            // the leg-timeout path.
            release_terminal(filled_key);
            co_await gateway_.cancel_order(resting_leg.symbol, resting_id);
            const OrderRecord partner_final =
                co_await await_authoritative_after_cancel(resting_chan, resting_key, resting_id);
            release_terminal(resting_key);

            if (partner_final.filled_qty > 0.0) {
                apply_fill_pnl(partner_final, resting_leg);
                co_await hedge_remaining(resting_leg, partner_final.qty - partner_final.filled_qty);
            }
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
        // hedge sizing — wait for the authoritative post-cancel outcome.
        const OrderRecord final_rec =
            co_await await_authoritative_after_cancel(resting_chan, resting_key, resting_id);
        release_terminal(resting_key);

        if (final_rec.status == OrderStatus::Filled) {
            apply_fill_pnl(final_rec, resting_leg);
            co_return;  // filled before/during the cancel race — nothing to hedge
        }
        if (final_rec.filled_qty > 0.0) {
            apply_fill_pnl(final_rec, resting_leg);  // partial fill before the cancel landed
        }
        co_await hedge_remaining(resting_leg, final_rec.qty - final_rec.filled_qty);
    }

    // Sends the cancel request already having been issued by the caller,
    // waits up to kCancelAckTimeout for the *authoritative* terminal state
    // it (or a racing fill) produces, and falls back to the last-known OMS
    // snapshot — loudly — if the ack never arrives.
    //
    // BUG FIX (found via testing, pre-existing / independent of the
    // in-flight-pair dedup guard above): when monitor_pair()'s initial
    // race — `wait_for_terminal_or_filled(chan1) || wait_for_terminal_or_
    // filled(chan2)` — has both legs settle to a terminal state at
    // essentially the same instant (e.g. both GTX legs rejected together,
    // or the stale-order sweeper canceling both legs of a pair on the
    // same tick), the *losing* operand's already-buffered terminal
    // message gets silently consumed and discarded as part of that race
    // resolving, even though it never became this function's `first_rec`.
    // The result: this function's own race against resting_chan finds the
    // channel empty and unconditionally burns the full kCancelAckTimeout
    // (750ms) before falling back — on every single dual-terminal event,
    // not just rare ones. Since OMSCore::apply_update() always mutates its
    // own authoritative record before firing the pub/sub callback that
    // feeds the channel, checking it directly here is a correct and
    // race-condition-proof short-circuit: it doesn't matter whether the
    // channel ever delivered (or lost) the message, only whether the order
    // has already reached a terminal state by the time we ask.
    asio::awaitable<OrderRecord> await_authoritative_after_cancel(std::shared_ptr<Chan> chan,
                                                                    uint64_t key,
                                                                    const std::string& id_for_log) {
        if (auto snap = oms_.get(key); snap && is_terminal(snap->status)) {
            co_return *snap;
        }

        using namespace boost::asio::experimental::awaitable_operators;
        asio::steady_timer ack_timer(exec_);
        ack_timer.expires_after(kCancelAckTimeout);
        auto outcome = co_await (wait_for_terminal_or_filled(chan) ||
                                  ack_timer.async_wait(asio::use_awaitable));
        if (outcome.index() == 0) co_return std::get<0>(outcome);

        std::cerr << "[ExecutionEngine] WARNING: no cancel ack for " << id_for_log << " within "
                  << kCancelAckTimeout.count()
                  << "ms; using last known OMS state.\n";
        auto snap = oms_.get(key);
        co_return snap ? *snap : OrderRecord{};
    }

    // Registers, sends, and hands off tracking of a MARKET hedge order for
    // `remaining` units of `leg.symbol`/`leg.side`. No-ops if there's
    // nothing to hedge. See race-condition note (5) for why the hedge
    // must be tracked to completion rather than fire-and-forgotten.
    asio::awaitable<void> hedge_remaining(const ArbLeg& leg, double remaining) {
        if (remaining <= 1e-12) co_return;
        const std::string hedge_id = next_client_id();
        const Side hedge_side = leg.side;  // hedge in the SAME direction the resting leg
                                            // would have filled, to realize the exposure
                                            // the completed/canceled counterpart opened.
        oms_.register_new_order(hedge_id, leg.symbol, hedge_side, 0.0, remaining);
        auto hedge_chan = get_or_create_waiter(fnv1a64(hedge_id));
        co_await gateway_.send_market(hedge_id, leg.symbol, hedge_side, remaining);

        asio::co_spawn(exec_, watch_hedge(hedge_id, leg.symbol, hedge_side, hedge_chan),
                        [this](std::exception_ptr e) { log_exception("watch_hedge", e); });
    }

    asio::awaitable<void> sweeper_loop(std::chrono::seconds interval, std::chrono::nanoseconds max_age) {
        asio::steady_timer timer(exec_);
        std::vector<OrderRecord> stale;  // reused across ticks — no per-tick allocation
                                          // once it grows to its high-water mark
        for (;;) {
            timer.expires_after(interval);
            co_await timer.async_wait(asio::use_awaitable);

            oms_.get_stale_orders(max_age.count(), stale);
            if (stale.empty()) continue;

            std::cerr << "[ExecutionEngine] sweeper: " << stale.size()
                      << " order(s) stale beyond "
                      << std::chrono::duration_cast<std::chrono::seconds>(max_age).count()
                      << "s — canceling.\n";

            // Best-effort, sequential: just get the cancel requests out.
            // The authoritative outcome of each (Canceled, or a
            // last-instant Filled racing the cancel) arrives via
            // UserDataFeed -> dispatch_update() as normal. If a swept
            // order is one leg of a pair whose monitor_pair() is still
            // running, monitor_pair()'s own "first leg settled without
            // filling" branch (which re-checks the partner's
            // authoritative state before deciding whether to hedge) is
            // what actually handles the fallout — the sweeper's only job
            // is to stop paying rent on stale resting orders, not to
            // duplicate risk decisions monitor_pair already owns.
            for (const auto& rec : stale) {
                co_await gateway_.cancel_order(view(rec.symbol), view(rec.client_order_id));
            }
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

        double& prev_comm = last_commission_[rec.exchange_order_id];
        const double comm_delta = rec.cumulative_commission - prev_comm;
        if (comm_delta > 1e-12) pnl_.apply_commission(leg.symbol, comm_delta);
        prev_comm = rec.cumulative_commission;
    }

    // BUG FIX: last_filled_/last_commission_ are keyed by exchange_order_id
    // (see apply_fill_pnl above), not by the client-id key this function
    // receives — and nothing was ever erasing them. On a long-running live
    // engine that's an unbounded leak: one permanent entry in each map per
    // order ever executed, for the lifetime of the process. Fetch the
    // OMS snapshot *before* releasing the slot (release() erases the
    // index_ entry get() depends on) so we have the exchange_order_id to
    // erase by. exchange_order_id == 0 (orders rejected before the venue
    // ever assigned a real id) is harmless to erase-if-present: those
    // orders never reach apply_fill_pnl (no fill => no map entry created)
    // since a WS-API-level reject carries filled_qty == 0.
    void release_terminal(uint64_t key) {
        if (auto snap = oms_.get(key)) {
            std::lock_guard lk(last_filled_mtx_);
            last_filled_.erase(snap->exchange_order_id);
            last_commission_.erase(snap->exchange_order_id);
        }
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
        // Circuit breaker bookkeeping — independent of whether a monitor
        // is actively waiting on this specific order (waiters_ lookup
        // below): a stuck condition (bad margin, bad precision, exchange
        // rate limit) rejects *every* order it touches, monitored or not.
        if (rec.status == OrderStatus::Rejected) {
            const auto n = consecutive_rejects_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n >= kMaxConsecutiveRejects && !halted_.exchange(true, std::memory_order_acq_rel)) {
                std::cerr << "[ExecutionEngine] CIRCUIT BREAKER TRIPPED: " << n
                          << " consecutive order rejects. Halting new signal processing. "
                             "Check margin/precision/rate-limit state, then call reset_halt().\n";
                write_halt_flag(std::to_string(n) + " consecutive order rejects");
            }
        } else if (rec.status == OrderStatus::Open || rec.status == OrderStatus::Filled) {
            consecutive_rejects_.store(0, std::memory_order_release);
        }

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

    // Collapses a message that would otherwise repeat on every single
    // dropped signal (which, at a 2ms polling interval with a persistent
    // condition, means thousands of identical lines per second) down to
    // one line per kLogThrottleNs, with a count of how many were
    // suppressed in between. This is what actually made the circuit
    // breaker / capacity logs legible instead of drowning the terminal.
    static constexpr int64_t kLogThrottleNs = 2'000'000'000;  // 2s

    void throttled_log(std::atomic<int64_t>& last_ns, std::atomic<uint64_t>& suppressed,
                        const std::string& msg) {
        const int64_t t = now_ns();
        int64_t prev = last_ns.load(std::memory_order_relaxed);
        if (t - prev < kLogThrottleNs) {
            suppressed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (!last_ns.compare_exchange_strong(prev, t, std::memory_order_relaxed)) return;
        const uint64_t n = suppressed.exchange(0, std::memory_order_relaxed);
        std::cerr << "[ExecutionEngine] signal dropped — " << msg;
        if (n > 0) std::cerr << " (+" << n << " more suppressed in the last "
                              << kLogThrottleNs / 1'000'000'000 << "s)";
        std::cerr << "\n";
    }

    void clear_inflight(const std::string& pair_key) {
        std::lock_guard lk(inflight_mtx_);
        inflight_pairs_.erase(pair_key);
    }

    // Best-effort: lets an external monitor/supervisor (cron, systemd
    // watchdog, a dashboard tailing the working directory) notice a trip
    // even if it isn't tailing stdout at the moment it happens — the
    // stderr line alone is easy to miss on an unattended box. Failure to
    // write this file is intentionally non-fatal.
    void write_halt_flag(const std::string& reason) {
        try {
            std::ofstream f("holo_halt.flag", std::ios::trunc);
            f << "HALTED at " << now_ns() << "ns — " << reason << "\n";
        } catch (...) {
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

    static constexpr int kMaxConsecutiveRejects = 5;
    std::size_t max_open_orders_;
    std::atomic<bool> halted_{false};
    std::atomic<int> consecutive_rejects_{0};

    // Throttled-log state — one pair of atomics per distinct drop reason so
    // a burst of one kind (e.g. capacity) never eats the log budget meant
    // for another (e.g. halted), and vice versa.
    std::atomic<int64_t> last_halt_log_ns_{0};
    std::atomic<uint64_t> halt_drops_suppressed_{0};
    std::atomic<int64_t> last_capacity_log_ns_{0};
    std::atomic<uint64_t> capacity_drops_suppressed_{0};
    std::atomic<int64_t> last_dup_log_ns_{0};
    std::atomic<uint64_t> dup_drops_suppressed_{0};

    // Which symbol pairs currently have an unresolved arb in flight — see
    // the dedup guard in on_signal().
    std::mutex inflight_mtx_;
    std::unordered_set<std::string> inflight_pairs_;

    // RAII cleanup for an inflight_pairs_ entry — see on_signal() for why
    // this exists (an exception between insert and co_spawn must not leave
    // the pair permanently stuck). disarm() hands cleanup responsibility
    // off to the spawned coroutine's completion handler once we know it's
    // actually going to run.
    struct InflightGuard {
        ExecutionEngine& eng;
        std::string key;
        bool armed{true};
        InflightGuard(ExecutionEngine& e, std::string k) : eng(e), key(std::move(k)) {}
        InflightGuard(const InflightGuard&) = delete;
        InflightGuard& operator=(const InflightGuard&) = delete;
        void disarm() { armed = false; }
        ~InflightGuard() {
            if (armed) eng.clear_inflight(key);
        }
    };

    std::mutex waiters_mtx_;
    std::unordered_map<uint64_t, std::shared_ptr<Chan>> waiters_;

    std::mutex last_filled_mtx_;
    std::unordered_map<int64_t, double> last_filled_;      // keyed by exchange_order_id
    std::unordered_map<int64_t, double> last_commission_;  // keyed by exchange_order_id
};

}  // namespace holo::net