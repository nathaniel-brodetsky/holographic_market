#pragma once
#include <trading/execution_types.hpp>
#include <core/lockfree_ring_buffer.hpp>
#include <array>
#include <cstdio>
#include <cmath>

namespace holo::trading {

class RiskManager final {
public:
    RiskManager(
        core::DynamicSpscRingBuffer<OrderRequest>& tx_ring,
        core::DynamicSpscRingBuffer<ExecutionReport>& rx_ring,
        float max_position_usd)
        : tx_ring_(tx_ring), rx_ring_(rx_ring), max_position_usd_(max_position_usd) {}

    void process_execution_reports() noexcept {
        ExecutionReport rep;
        while (rx_ring_.try_pop(rep)) {
            if (rep.status == OrderState::FILLED) {
                update_position(rep.instrument_id, rep.exec_qty, rep.exec_price, rep.is_buy);
            }
            states_[rep.instrument_id] = OrderState::IDLE;
        }
    }

    [[nodiscard]] bool try_route_signal(uint32_t instr_id, float qty, float price, bool is_buy, uint64_t ts_ns) noexcept {
        if (instr_id >= 4U) return false;

        if (states_[instr_id] == OrderState::IN_FLIGHT) {
            return false;
        }

        const float cur_qty = positions_[instr_id].net_qty;
        if (std::abs(cur_qty) * price > max_position_usd_) {
            return false;
        }

        states_[instr_id] = OrderState::IN_FLIGHT;
        
        OrderRequest req{instr_id, qty, price, is_buy, ts_ns};
        if (!tx_ring_.try_push(req)) {
            states_[instr_id] = OrderState::IDLE;
            return false;
        }
        return true;
    }

    void print_summary() const noexcept {
        std::printf("\n  ── Risk Manager Summary ──\n");
        std::printf("  Total Realized PnL: %.4f USD\n", total_pnl_);
        for (std::size_t i = 0; i < 4; ++i) {
            std::printf("  [Instr %zu] Qty: %.4f | PnL: %.4f | Orders: %llu\n", 
                i, positions_[i].net_qty, positions_[i].realized_pnl, 
                static_cast<unsigned long long>(positions_[i].n_orders));
        }
    }

private:
    void update_position(uint32_t id, float qty, float price, bool is_buy) noexcept {
        auto& pos = positions_[id];
        const float sign = is_buy ? 1.0F : -1.0F;
        const float prev_qty = pos.net_qty;
        const float prev_ep  = pos.avg_entry_price;
        const float new_qty  = prev_qty + sign * qty;

        if (std::abs(new_qty) < 1e-9F) {
            const float rpnl = prev_qty * (price - prev_ep);
            pos.realized_pnl += rpnl;
            total_pnl_ += rpnl;
            pos.net_qty = 0.0F;
            pos.avg_entry_price = 0.0F;
        } else if (prev_qty == 0.0F || (prev_qty > 0.0F) == is_buy) {
            pos.avg_entry_price = (prev_ep * std::abs(prev_qty) + price * qty) / (std::abs(prev_qty) + qty);
            pos.net_qty = new_qty;
        } else {
            const float closed = std::min(std::abs(prev_qty), qty);
            const float rpnl   = (is_buy ? -1.0F : 1.0F) * closed * (price - prev_ep);
            pos.realized_pnl += rpnl;
            total_pnl_ += rpnl;
            pos.net_qty = new_qty;
        }
        pos.n_orders++;
    }

    core::DynamicSpscRingBuffer<OrderRequest>& tx_ring_;
    core::DynamicSpscRingBuffer<ExecutionReport>& rx_ring_;
    
    std::array<OrderState, 4> states_{OrderState::IDLE, OrderState::IDLE, OrderState::IDLE, OrderState::IDLE};
    std::array<Position, 4> positions_{};
    float total_pnl_{0.0F};
    float max_position_usd_;
};

} // namespace holo::trading