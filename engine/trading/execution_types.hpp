#pragma once
#include <cstdint>
#include <atomic>

namespace holo::trading {

    enum class OrderState : uint8_t {
        IDLE = 0,
        IN_FLIGHT,
        FILLED,
        REJECTED
    };

    struct alignas(32) OrderRequest {
        uint32_t instrument_id;
        float    qty;
        float    price;
        bool     is_buy;
        uint64_t request_ts_ns;
    };

    struct alignas(32) ExecutionReport {
        uint32_t   instrument_id;
        OrderState status;
        float      exec_price;
        float      exec_qty;
        bool       is_buy;
    };

    struct alignas(64) Position {
        float net_qty{0.0F};
        float avg_entry_price{0.0F};
        float realized_pnl{0.0F};
        uint64_t n_orders{0U};
    };

} // namespace holo::trading