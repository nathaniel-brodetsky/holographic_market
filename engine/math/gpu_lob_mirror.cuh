#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cuda_runtime.h>
#include <math/cuda_utils.cuh>
#include <core/memory_arena.hpp>
#include <core/lob_core.hpp>

namespace holo::cuda
{
    static constexpr size_t k_gpu_max_instruments = 512U;
    static constexpr size_t k_gpu_max_depth = 16U;

    struct alignas(16) GpuLobSnapshot
    {
        float *d_bid_prices;
        float *d_ask_prices;
        float *d_bid_qtys;
        float *d_ask_qtys;
        uint32_t n_instruments;
        uint32_t depth;
        uint64_t generation;
        uint64_t transfer_ts_ns;
    };

    class GpuLobMirror final
    {
    public:
        explicit GpuLobMirror(
            const LobSoA &lob_soa,
            cudaStream_t transfer_stream)
            : n_instruments_{lob_soa.n_instruments()}, depth_{lob_soa.depth()}, n_floats_{lob_soa.n_instruments() * lob_soa.depth()}, stream_{transfer_stream}, generation_{0U}
        {
            CUDA_CHECK(cudaHostRegister(
                const_cast<float *>(lob_soa.raw_bid_price_array().data()),
                n_floats_ * sizeof(float),
                cudaHostRegisterDefault));
            CUDA_CHECK(cudaHostRegister(
                const_cast<float *>(lob_soa.raw_ask_price_array().data()),
                n_floats_ * sizeof(float),
                cudaHostRegisterDefault));
            CUDA_CHECK(cudaHostRegister(
                const_cast<float *>(lob_soa.raw_bid_qty_array().data()),
                n_floats_ * sizeof(float),
                cudaHostRegisterDefault));
            CUDA_CHECK(cudaHostRegister(
                const_cast<float *>(lob_soa.raw_ask_qty_array().data()),
                n_floats_ * sizeof(float),
                cudaHostRegisterDefault));

            h_bid_prices_ = lob_soa.raw_bid_price_array().data();
            h_ask_prices_ = lob_soa.raw_ask_price_array().data();
            h_bid_qtys_ = lob_soa.raw_bid_qty_array().data();
            h_ask_qtys_ = lob_soa.raw_ask_qty_array().data();

            d_bid_prices_ = device_alloc<float>(n_floats_);
            d_ask_prices_ = device_alloc<float>(n_floats_);
            d_bid_qtys_ = device_alloc<float>(n_floats_);
            d_ask_qtys_ = device_alloc<float>(n_floats_);

            CUDA_CHECK(cudaMemset(d_bid_prices_, 0, n_floats_ * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_ask_prices_, 0, n_floats_ * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_bid_qtys_, 0, n_floats_ * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_ask_qtys_, 0, n_floats_ * sizeof(float)));
        }

        ~GpuLobMirror() noexcept
        {
            device_free(d_bid_prices_);
            device_free(d_ask_prices_);
            device_free(d_bid_qtys_);
            device_free(d_ask_qtys_);

            cudaHostUnregister(const_cast<float *>(h_bid_prices_));
            cudaHostUnregister(const_cast<float *>(h_ask_prices_));
            cudaHostUnregister(const_cast<float *>(h_bid_qtys_));
            cudaHostUnregister(const_cast<float *>(h_ask_qtys_));
        }

        GpuLobMirror(const GpuLobMirror &) = delete;

        GpuLobMirror &operator=(const GpuLobMirror &) = delete;

        GpuLobMirror(GpuLobMirror &&) = delete;

        GpuLobMirror &operator=(GpuLobMirror &&) = delete;

        void async_push(uint64_t current_generation) noexcept
        {
            CUDA_CHECK(cudaMemcpyAsync(
                d_bid_prices_, h_bid_prices_,
                n_floats_ * sizeof(float),
                cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                d_ask_prices_, h_ask_prices_,
                n_floats_ * sizeof(float),
                cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                d_bid_qtys_, h_bid_qtys_,
                n_floats_ * sizeof(float),
                cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                d_ask_qtys_, h_ask_qtys_,
                n_floats_ * sizeof(float),
                cudaMemcpyHostToDevice, stream_));
            generation_ = current_generation;
        }

        [[nodiscard]] GpuLobSnapshot snapshot() const noexcept
        {
            return GpuLobSnapshot{
                .d_bid_prices = d_bid_prices_,
                .d_ask_prices = d_ask_prices_,
                .d_bid_qtys = d_bid_qtys_,
                .d_ask_qtys = d_ask_qtys_,
                .n_instruments = static_cast<uint32_t>(n_instruments_),
                .depth = static_cast<uint32_t>(depth_),
                .generation = generation_,
                .transfer_ts_ns = 0U,
            };
        }

        [[nodiscard]] size_t n_floats() const noexcept { return n_floats_; }
        [[nodiscard]] size_t n_instruments() const noexcept { return n_instruments_; }
        [[nodiscard]] size_t depth() const noexcept { return depth_; }

    private:
        const size_t n_instruments_;
        const size_t depth_;
        const size_t n_floats_;
        cudaStream_t stream_;
        uint64_t generation_;

        const float *h_bid_prices_;
        const float *h_ask_prices_;
        const float *h_bid_qtys_;
        const float *h_ask_qtys_;

        float *d_bid_prices_;
        float *d_ask_prices_;
        float *d_bid_qtys_;
        float *d_ask_qtys_;
    };
} // namespace holo::cuda

// holo::math::cuda is the canonical namespace for V2. holo::cuda remains as alias.
namespace holo::math { namespace cuda = holo::cuda; }
