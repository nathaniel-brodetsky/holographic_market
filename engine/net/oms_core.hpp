#pragma once
//
// oms_core.hpp
// holo::net — Order Management System core (state machine).
//
// A thread-safe, fixed-capacity order state tracker. Steady-state operation
// (updating an existing order) performs no heap allocation: records live in
// a pre-reserved slot pool, and the client-order-id -> slot lookup is done
// via an integer key (FNV-1a hash of the id string) rather than a string
// hash/compare on the hot path.
//
// Concurrency model: reads take a shared_lock, mutations take a unique_lock.
// register_new_order / apply_update copy the record out and release the
// lock *before* invoking the update callback, so a slow subscriber can
// never block a writer (the UserDataFeed's read loop) or another reader.
//
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace holo::net {

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------
// now_ns(): wall-clock (epoch) time. Used where the value needs to mean
// something outside this process — cross-referenced against Binance's own
// event timestamps, or written to a human-readable log (see
// ExecutionEngine's "HALTED at <now_ns()>ns" log and binance_gateway.hpp's
// reject-path event_ns). Do NOT repurpose this for duration/elapsed-time
// math — system_clock is not monotonic and can jump (NTP correction),
// which is exactly the failure mode steady_ns() below exists to avoid.
inline int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// steady_ns(): monotonic clock, meaningless as a wall-clock timestamp (its
// epoch is unspecified and arbitrary) but never jumps backward/forward due
// to clock synchronization. Use this for anything measuring "how long has
// it been" rather than "what time is it" — e.g. order-age/staleness
// checks (see OrderRecord::created_mono_ns / get_stale_orders below).
inline int64_t steady_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// OrderStatus / Side
// ---------------------------------------------------------------------------
enum class OrderStatus : uint8_t {
    PendingNew = 0,  // sent to venue, ack not yet received
    Open       = 1,  // acked, resting on book (Binance "NEW")
    Partial    = 2,  // partially filled
    Filled     = 3,  // fully filled
    Canceled   = 4,  // canceled (by us, or venue e.g. GTX post-only reject/EXPIRED)
    Rejected   = 5,  // rejected on send or by venue
};

constexpr bool is_terminal(OrderStatus s) noexcept {
    return s == OrderStatus::Filled || s == OrderStatus::Canceled || s == OrderStatus::Rejected;
}
constexpr bool is_live(OrderStatus s) noexcept {
    return s == OrderStatus::PendingNew || s == OrderStatus::Open || s == OrderStatus::Partial;
}

enum class Side : uint8_t { Buy = 0, Sell = 1 };

constexpr Side opposite(Side s) noexcept { return s == Side::Buy ? Side::Sell : Side::Buy; }
// Signed direction multiplier: +1 for Buy, -1 for Sell. Used for PnL/position math.
constexpr double signed_dir(Side s) noexcept { return s == Side::Buy ? 1.0 : -1.0; }

// ---------------------------------------------------------------------------
// Fixed-width buffers — avoid std::string heap allocation on the hot path.
// ---------------------------------------------------------------------------
using SymbolBuf   = std::array<char, 16>;
using ClientIdBuf = std::array<char, 40>;  // Binance newClientOrderId <= 36 chars

inline SymbolBuf make_symbol(std::string_view sv) {
    SymbolBuf buf{};
    if (sv.size() > buf.size() - 1) {
        // Unlike client_order_id (Binance-documented <= 36 chars, buffer is
        // 40 — comfortable margin), there's no equivalent guarantee for
        // symbol length here. A silent truncation would hand a *different*,
        // wrong-but-valid-looking symbol string to the rest of the system
        // with zero indication anything went wrong.
        std::cerr << "[OMSCore] WARNING: symbol \"" << sv << "\" (" << sv.size()
                  << " chars) exceeds SymbolBuf capacity (" << (buf.size() - 1)
                  << ") — truncating, this WILL corrupt downstream symbol matching.\n";
    }
    std::memcpy(buf.data(), sv.data(), std::min(sv.size(), buf.size() - 1));
    return buf;
}
inline ClientIdBuf make_client_id(std::string_view sv) {
    ClientIdBuf buf{};
    if (sv.size() > buf.size() - 1) {
        std::cerr << "[OMSCore] WARNING: client_order_id \"" << sv << "\" (" << sv.size()
                  << " chars) exceeds ClientIdBuf capacity (" << (buf.size() - 1)
                  << ") — truncating, this WILL corrupt order-id round-tripping "
                     "with the exchange.\n";
    }
    std::memcpy(buf.data(), sv.data(), std::min(sv.size(), buf.size() - 1));
    return buf;
}
inline std::string_view view(const SymbolBuf& b) noexcept { return std::string_view(b.data()); }
inline std::string_view view(const ClientIdBuf& b) noexcept { return std::string_view(b.data()); }

// FNV-1a 64-bit: derives an integer key from a client-order-id string, so
// order lookups on the hot path are integer-keyed, not string-keyed.
constexpr uint64_t fnv1a64(std::string_view sv) noexcept {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : sv) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

// ---------------------------------------------------------------------------
// OrderRecord — POD, trivially copyable (safe to copy out from under the
// lock and hand to subscribers/coroutines without touching OMSCore again).
// ---------------------------------------------------------------------------
struct OrderRecord {
    ClientIdBuf client_order_id{};
    SymbolBuf   symbol{};
    Side        side{Side::Buy};
    OrderStatus status{OrderStatus::PendingNew};
    double      price{0.0};
    double      qty{0.0};
    double      filled_qty{0.0};
    double      avg_fill_price{0.0};
    int64_t     exchange_order_id{0};
    int64_t     created_ns{0};       // wall-clock (epoch) — for logs/display
    int64_t     updated_ns{0};       // wall-clock (epoch) — for logs/display
    int64_t     created_mono_ns{0};  // steady_clock — for staleness math only

    [[nodiscard]] double remaining_qty() const noexcept { return qty - filled_qty; }
};
static_assert(std::is_trivially_copyable_v<OrderRecord>);

// ---------------------------------------------------------------------------
// OMSCore
// ---------------------------------------------------------------------------
class OMSCore {
public:
    using UpdateCallback = std::function<void(const OrderRecord&)>;

    explicit OMSCore(size_t reserve_capacity = 8192) {
        slots_.reserve(reserve_capacity);
        index_.reserve(reserve_capacity * 2);
        free_list_.reserve(reserve_capacity);
    }

    OMSCore(const OMSCore&)            = delete;
    OMSCore& operator=(const OMSCore&) = delete;

    // Single-subscriber update callback, fired (outside the lock) whenever
    // an order's state changes — new registration or a fill/cancel/reject
    // event applied via apply_update(). Wire this once at startup; it is
    // not designed for concurrent multi-subscriber fan-out.
    void set_update_callback(UpdateCallback cb) {
        std::unique_lock lk(mtx_);
        on_update_ = std::move(cb);
    }

    // Register a freshly-sent order (state = PendingNew). Returns the
    // FNV-1a key used for all subsequent lookups/updates — cache it on the
    // caller side (e.g. in the execution engine) to avoid re-hashing.
    uint64_t register_new_order(std::string_view client_order_id,
                                 std::string_view symbol,
                                 Side side,
                                 double price,
                                 double qty) {
        OrderRecord rec{};
        rec.client_order_id = make_client_id(client_order_id);
        rec.symbol          = make_symbol(symbol);
        rec.side            = side;
        rec.status          = OrderStatus::PendingNew;
        rec.price           = price;
        rec.qty             = qty;
        rec.created_ns      = now_ns();
        rec.updated_ns      = rec.created_ns;
        rec.created_mono_ns = steady_ns();

        const uint64_t key = fnv1a64(client_order_id);

        std::unique_lock lk(mtx_);
        // A key already present here means either an FNV-1a hash collision
        // (astronomically unlikely for the monotonically-incrementing
        // "holo-<seq>" client IDs this OMS is fed in practice) or a caller
        // bug reusing a client_order_id before release()-ing the previous
        // order. Either way, silently overwriting index_[key] would orphan
        // the previous slot: it stays live in slots_ but becomes
        // unreachable, so its fills/rejects can never be routed to it
        // again — a real order the OMS quietly stops tracking. Not
        // something to fail hard on here (this is a hot path called from
        // the signal loop), but it must not be silent.
        if (index_.contains(key)) {
            std::cerr << "[OMSCore] WARNING: client_order_id key collision on "
                         "register_new_order for \"" << client_order_id
                      << "\" — overwriting a live index entry; the previous "
                         "order at that key is now orphaned (its updates will "
                         "no longer be routed).\n";
        }
        const size_t slot = alloc_slot_locked();
        slots_[slot] = rec;
        index_[key]  = slot;
        OrderRecord copy = slots_[slot];
        lk.unlock();

        fire_update(copy);
        return key;
    }

    // Apply an update from the user-data stream / REST ack.
    // `cumulative_filled_qty` is the venue's cumulative fill quantity
    // (Binance "z"), not an incremental delta. Monotonicity of the
    // (status, filled_qty) pair is the caller's responsibility to reason
    // about — Binance's user-data stream is itself ordered per-symbol, so
    // this rarely needs defending against out-of-order delivery, but we
    // never let filled_qty go backwards on a stale/duplicate event.
    bool apply_update(uint64_t key,
                       OrderStatus new_status,
                       double cumulative_filled_qty,
                       double last_fill_price,
                       int64_t exchange_order_id,
                       int64_t event_ns) {
        std::unique_lock lk(mtx_);
        auto it = index_.find(key);
        if (it == index_.end()) return false;
        OrderRecord& rec = slots_[it->second];

        if (cumulative_filled_qty > rec.filled_qty && last_fill_price > 0.0) {
            const double prev_notional = rec.avg_fill_price * rec.filled_qty;
            const double delta_qty     = cumulative_filled_qty - rec.filled_qty;
            rec.avg_fill_price = (prev_notional + last_fill_price * delta_qty) / cumulative_filled_qty;
            rec.filled_qty     = cumulative_filled_qty;
        } else if (cumulative_filled_qty > rec.filled_qty) {
            // Status-only progression (e.g. cumulative qty given without a
            // fill price on this particular event) — still advance qty.
            rec.filled_qty = cumulative_filled_qty;
        }
        rec.status            = new_status;
        rec.exchange_order_id = exchange_order_id;
        rec.updated_ns         = event_ns;

        OrderRecord copy = rec;
        lk.unlock();

        fire_update(copy);
        return true;
    }

    [[nodiscard]] std::optional<OrderRecord> get(uint64_t key) const {
        std::shared_lock lk(mtx_);
        auto it = index_.find(key);
        if (it == index_.end()) return std::nullopt;
        return slots_[it->second];
    }

    [[nodiscard]] std::optional<OrderRecord> get(std::string_view client_order_id) const {
        return get(fnv1a64(client_order_id));
    }

    // Out-param overload: reuse `out` across polling calls to avoid
    // per-call heap allocation on a hot stale-order sweep loop.
    void get_stale_orders(int64_t max_age_ns, std::vector<OrderRecord>& out) const {
        out.clear();
        // Monotonic cutoff: an NTP step correction on now_ns() (system_clock)
        // could otherwise make a genuinely-stuck order look fresh (clock
        // stepped backward) or a brand-new one look stale (clock stepped
        // forward) right when the sweeper runs. created_mono_ns/steady_ns()
        // are immune to that by construction.
        const int64_t cutoff = steady_ns() - max_age_ns;
        std::shared_lock lk(mtx_);
        for (const auto& [key, slot] : index_) {
            const OrderRecord& rec = slots_[slot];
            if (is_live(rec.status) && rec.created_mono_ns < cutoff) out.push_back(rec);
        }
    }

    [[nodiscard]] std::vector<OrderRecord> get_stale_orders(int64_t max_age_ns) const {
        std::vector<OrderRecord> out;
        get_stale_orders(max_age_ns, out);
        return out;
    }

    // Release a terminal order's slot for reuse. Call once you're done
    // consuming a Filled/Canceled/Rejected record (e.g. after logging it,
    // or after the execution engine's monitor coroutine has finished with
    // it). Failing to call this for terminal orders leaks index_ entries
    // (not slot memory — slots_ itself never shrinks by design) forever.
    void release(uint64_t key) {
        std::unique_lock lk(mtx_);
        auto it = index_.find(key);
        if (it == index_.end()) return;
        free_list_.push_back(it->second);
        index_.erase(it);
    }

    [[nodiscard]] size_t live_order_count() const {
        std::shared_lock lk(mtx_);
        return index_.size();
    }

private:
    size_t alloc_slot_locked() {
        if (!free_list_.empty()) {
            const size_t s = free_list_.back();
            free_list_.pop_back();
            return s;
        }
        slots_.emplace_back();
        return slots_.size() - 1;
    }

    void fire_update(const OrderRecord& rec) {
        UpdateCallback cb;
        {
            std::shared_lock lk(mtx_);
            cb = on_update_;
        }
        if (cb) cb(rec);
    }

    mutable std::shared_mutex mtx_;
    std::vector<OrderRecord> slots_;
    std::unordered_map<uint64_t, size_t> index_;
    std::vector<size_t> free_list_;
    UpdateCallback on_update_;
};

}  // namespace holo::net