#include <math/cuda_pipeline.cuh>
#include <math/cuda_utils.cuh>

#include <atomic>
#include <cstring>

namespace holo::cuda
{
    namespace
    {
        // Bounded retry cap for the seqlock read-side. In practice the
        // host-to-host memcpy below (a few hundred KB at most) completes
        // in low single-digit microseconds, far faster than the interval
        // between LOB updates, so we expect 0 retries almost always and
        // never come close to this bound. It exists purely as a defensive
        // fallback so a pathological writer burst can never turn this into
        // an unbounded/live-locked loop -- if it is ever hit we fall back
        // to whatever was last read (a possibly-torn snapshot) rather than
        // stalling the pipeline thread indefinitely, and we count it so
        // it is visible in metrics() instead of failing silently.
        constexpr int k_max_seqlock_retries = 64;
    }

    GpuLobMirror::GpuLobMirror(const LobSoA &lob_soa, cudaStream_t transfer_stream)
        : n_instruments_{lob_soa.n_instruments()}
        , depth_{lob_soa.depth()}
        , n_floats_{n_instruments_ * depth_}
        , stream_{transfer_stream}
        , generation_{0U}
        , lob_ref_{&lob_soa}
        , h_bid_prices_{lob_soa.raw_bid_price_array().data()}
        , h_ask_prices_{lob_soa.raw_ask_price_array().data()}
        , h_bid_qtys_{lob_soa.raw_bid_qty_array().data()}
        , h_ask_qtys_{lob_soa.raw_ask_qty_array().data()}
    {
        // FIX: we used to cudaHostRegister() LobSoA's own arrays and DMA
        // straight out of them with cudaMemcpyAsync -- but LobSoA::apply()
        // (running on the feed-handler thread) keeps writing into those
        // same arrays with no synchronization at all, so the async copy
        // could race a concurrent writer and hand the GPU a torn/half
        // -updated snapshot.
        //
        // Now we DMA out of a private pinned staging buffer instead.
        // stage_snapshot() fills it via a seqlock-protected host-to-host
        // memcpy (see below), so by the time cudaMemcpyAsync touches it,
        // no writer can be mutating it -- the writer never even sees this
        // buffer. h_bid_prices_ etc. are therefore no longer registered
        // with CUDA; they are just an ordinary (pageable) copy source now.
        stage_bid_prices_ = pinned_alloc<float>(n_floats_);
        stage_ask_prices_ = pinned_alloc<float>(n_floats_);
        stage_bid_qtys_   = pinned_alloc<float>(n_floats_);
        stage_ask_qtys_   = pinned_alloc<float>(n_floats_);

        // Выделяем память на видеокарте
        d_bid_prices_ = device_alloc<float>(n_floats_);
        d_ask_prices_ = device_alloc<float>(n_floats_);
        d_bid_qtys_ = device_alloc<float>(n_floats_);
        d_ask_qtys_ = device_alloc<float>(n_floats_);

        CUDA_CHECK(cudaMemset(d_bid_prices_, 0, n_floats_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ask_prices_, 0, n_floats_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bid_qtys_, 0, n_floats_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ask_qtys_, 0, n_floats_ * sizeof(float)));
    }

    GpuLobMirror::~GpuLobMirror() noexcept
    {
        pinned_free(stage_bid_prices_);
        pinned_free(stage_ask_prices_);
        pinned_free(stage_bid_qtys_);
        pinned_free(stage_ask_qtys_);

        device_free(d_bid_prices_);
        device_free(d_ask_prices_);
        device_free(d_bid_qtys_);
        device_free(d_ask_qtys_);
    }

    void GpuLobMirror::stage_snapshot() noexcept
    {
        const size_t bytes = n_floats_ * sizeof(float);

        std::uint64_t seq1 = 0U;
        std::uint64_t seq2 = 0U;
        int attempt = 0;

        for (;; ++attempt)
        {
            seq1 = lob_ref_->seq();

            const bool writer_busy = (seq1 & 1U) != 0U;
            const bool give_up     = attempt >= k_max_seqlock_retries;

            if (!writer_busy || give_up)
            {
                // Acquire fence: the four memcpy's below (sequenced after
                // this point) must not be hoisted before the seq1 read.
                std::atomic_thread_fence(std::memory_order_acquire);

                std::memcpy(stage_bid_prices_, h_bid_prices_, bytes);
                std::memcpy(stage_ask_prices_, h_ask_prices_, bytes);
                std::memcpy(stage_bid_qtys_,   h_bid_qtys_,   bytes);
                std::memcpy(stage_ask_qtys_,   h_ask_qtys_,   bytes);

                // Acquire fence again: prevents the memcpy's above from
                // being reordered after the seq2 read that follows.
                std::atomic_thread_fence(std::memory_order_acquire);
                seq2 = lob_ref_->seq();

                if (give_up && (writer_busy || seq1 != seq2))
                {
                    metrics_.forced_after_max_retries.fetch_add(1U, std::memory_order_relaxed);
                    break;
                }
                if (!writer_busy && seq1 == seq2)
                {
                    break; // Clean, consistent snapshot.
                }
            }

            metrics_.retries.fetch_add(1U, std::memory_order_relaxed);
        }

        metrics_.snapshots_taken.fetch_add(1U, std::memory_order_relaxed);
    }

    void GpuLobMirror::async_push(std::uint64_t current_generation) noexcept
    {
        stage_snapshot();

        const size_t bytes = n_floats_ * sizeof(float);
        // Асинхронное копирование по шине PCIe без блокировки CPU.
        // Source is now the staging buffer, never LobSoA's live arrays.
        CUDA_CHECK(cudaMemcpyAsync(d_bid_prices_, stage_bid_prices_, bytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_ask_prices_, stage_ask_prices_, bytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_bid_qtys_, stage_bid_qtys_, bytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_ask_qtys_, stage_ask_qtys_, bytes, cudaMemcpyHostToDevice, stream_));
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
