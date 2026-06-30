#include <compute/cuda_pipeline.cuh>
#include <compute/cuda_utils.cuh>

namespace holo::cuda
{

    GpuLobMirror::GpuLobMirror(const LobSoA &lob_soa, cudaStream_t transfer_stream)
        : n_instruments_{lob_soa.n_instruments()}, depth_{lob_soa.depth()}, n_floats_{n_instruments_ * depth_}, stream_{transfer_stream}, generation_{0U}, h_bid_prices_{lob_soa.raw_bid_price_array().data()}, h_ask_prices_{lob_soa.raw_ask_price_array().data()}, h_bid_qtys_{lob_soa.raw_bid_qty_array().data()}, h_ask_qtys_{lob_soa.raw_ask_qty_array().data()}
    {
        // Регистрируем память CPU как Pinned (Page-locked) для Zero-Copy DMA
        CUDA_CHECK(cudaHostRegister(const_cast<float *>(h_bid_prices_), n_floats_ * sizeof(float), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(const_cast<float *>(h_ask_prices_), n_floats_ * sizeof(float), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(const_cast<float *>(h_bid_qtys_), n_floats_ * sizeof(float), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(const_cast<float *>(h_ask_qtys_), n_floats_ * sizeof(float), cudaHostRegisterDefault));

        // Выделяем память на видеокарте
        d_bid_prices_ = device_alloc<float>(n_floats_);
        d_ask_prices_ = device_alloc<float>(n_floats_);
        d_bid_qtys_ = device_alloc<float>(n_floats_);
        d_ask_qtys_ = device_alloc<float>(n_floats_);
    }

    GpuLobMirror::~GpuLobMirror() noexcept
    {
        CUDA_CHECK(cudaHostUnregister(const_cast<float *>(h_bid_prices_)));
        CUDA_CHECK(cudaHostUnregister(const_cast<float *>(h_ask_prices_)));
        CUDA_CHECK(cudaHostUnregister(const_cast<float *>(h_bid_qtys_)));
        CUDA_CHECK(cudaHostUnregister(const_cast<float *>(h_ask_qtys_)));

        device_free(d_bid_prices_);
        device_free(d_ask_prices_);
        device_free(d_bid_qtys_);
        device_free(d_ask_qtys_);
    }

    void GpuLobMirror::async_push(std::uint64_t current_generation) noexcept
    {
        const size_t bytes = n_floats_ * sizeof(float);
        // Асинхронное копирование по шине PCIe без блокировки CPU
        CUDA_CHECK(cudaMemcpyAsync(d_bid_prices_, h_bid_prices_, bytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_ask_prices_, h_ask_prices_, bytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_bid_qtys_, h_bid_qtys_, bytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_ask_qtys_, h_ask_qtys_, bytes, cudaMemcpyHostToDevice, stream_));
        generation_ = current_generation;
    }

    GpuLobSnapshot GpuLobMirror::snapshot() const noexcept
    {
        return GpuLobSnapshot{
            .d_bid_prices = d_bid_prices_,
            .d_ask_prices = d_ask_prices_,
            .d_bid_qtys = d_bid_qtys_,
            .d_ask_qtys = d_ask_qtys_,
            .n_instruments = static_cast<uint32_t>(n_instruments_),
            .depth = static_cast<uint32_t>(depth_),
            .generation = generation_,
            .transfer_ts_ns = 0U};
    }

} // namespace holo::cuda