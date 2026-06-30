#pragma once
#include <math/cuda_pipeline.cuh>
#include <core/lob_core.hpp>
#include <array>
#include <span>

namespace holo::trading {

    class AlphaModel final {
    public:
        AlphaModel(size_t n_instruments) : n_instruments_(n_instruments) {
            n_edges_ = 0;
            for (std::size_t i = 0; i < n_instruments; ++i) {
                for (std::size_t j = i + 1; j < n_instruments; ++j) {
                    src_edges_[n_edges_] = static_cast<int>(i);
                    dst_edges_[n_edges_] = static_cast<int>(j);
                    ++n_edges_;
                }
            }
        }

        std::span<const float> generate_flows(const cuda::SignalRecord& sig, const core::LobSoA& lob) noexcept {
            for (std::size_t e = 0; e < n_edges_; ++e) {
                float mi = lob.mid_price(src_edges_[e]);
                float mj = lob.mid_price(dst_edges_[e]);
                float denom = mi + mj + 1e-8F;
                flows_[e] = (mi - mj) / denom * sig.yang_mills_action;
            }
            return {flows_.data(), n_edges_};
        }

        std::span<const int> get_src_edges() const noexcept { return {src_edges_.data(), n_edges_}; }
        std::span<const int> get_dst_edges() const noexcept { return {dst_edges_.data(), n_edges_}; }

    private:
        static constexpr size_t k_max_edges = 6;
        size_t n_instruments_;
        size_t n_edges_;
        std::array<int, k_max_edges> src_edges_{};
        std::array<int, k_max_edges> dst_edges_{};
        std::array<float, k_max_edges> flows_{};
    };

} // namespace holo::trading