#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

namespace holo::cuda
{

#define CUDA_CHECK(expr)                                             \
    do                                                               \
    {                                                                \
        cudaError_t _holo_err = (expr);                              \
        if (_holo_err != cudaSuccess) [[unlikely]]                   \
        {                                                            \
            ::std::fprintf(stderr,                                   \
                           "[CUDA_CHECK] %s:%d  »%s«  →  %s (%d)\n", \
                           __FILE__, __LINE__, #expr,                \
                           ::cudaGetErrorString(_holo_err),          \
                           static_cast<int>(_holo_err));             \
            ::std::abort();                                          \
        }                                                            \
    } while (0)

#define CUBLAS_CHECK(expr)                                               \
    do                                                                   \
    {                                                                    \
        cublasStatus_t _holo_stat = (expr);                              \
        if (_holo_stat != CUBLAS_STATUS_SUCCESS) [[unlikely]]            \
        {                                                                \
            ::std::fprintf(stderr,                                       \
                           "[CUBLAS_CHECK] %s:%d  »%s«  →  status %d\n", \
                           __FILE__, __LINE__, #expr,                    \
                           static_cast<int>(_holo_stat));                \
            ::std::abort();                                              \
        }                                                                \
    } while (0)

#define CUSPARSE_CHECK(expr)                                             \
    do                                                                   \
    {                                                                    \
        cusparseStatus_t _holo_stat = (expr);                            \
        if (_holo_stat != CUSPARSE_STATUS_SUCCESS) [[unlikely]]          \
        {                                                                \
            ::std::fprintf(stderr,                                       \
                           "[CUSPARSE_CHECK] %s:%d  »%s«  →  %s (%d)\n", \
                           __FILE__, __LINE__, #expr,                    \
                           ::cusparseGetErrorString(_holo_stat),         \
                           static_cast<int>(_holo_stat));                \
            ::std::abort();                                              \
        }                                                                \
    } while (0)

#define CUSOLVER_CHECK(expr)                                               \
    do                                                                     \
    {                                                                      \
        cusolverStatus_t _holo_stat = (expr);                              \
        if (_holo_stat != CUSOLVER_STATUS_SUCCESS) [[unlikely]]            \
        {                                                                  \
            ::std::fprintf(stderr,                                         \
                           "[CUSOLVER_CHECK] %s:%d  »%s«  →  status %d\n", \
                           __FILE__, __LINE__, #expr,                      \
                           static_cast<int>(_holo_stat));                  \
            ::std::abort();                                                \
        }                                                                  \
    } while (0)

    template <typename T>
    [[nodiscard]] T *device_alloc(std::size_t count)
    {
        T *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&ptr),
            sizeof(T) * count));
        return ptr;
    }

    template <typename T>
    void device_free(T *ptr) noexcept
    {
        if (ptr != nullptr)
        {
            cudaFree(reinterpret_cast<void *>(ptr));
        }
    }

    template <typename T>
    [[nodiscard]] T *pinned_alloc(std::size_t count)
    {
        T *ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void **>(&ptr),
            sizeof(T) * count));
        return ptr;
    }

    template <typename T>
    void pinned_free(T *ptr) noexcept
    {
        if (ptr != nullptr)
        {
            cudaFreeHost(reinterpret_cast<void *>(ptr));
        }
    }

    class StreamGuard final
    {
    public:
        StreamGuard()
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &stream_, cudaStreamNonBlocking));
        }

        ~StreamGuard() noexcept
        {
            if (stream_ != nullptr)
            {
                cudaStreamDestroy(stream_);
            }
        }

        StreamGuard(const StreamGuard &) = delete;
        StreamGuard &operator=(const StreamGuard &) = delete;
        StreamGuard(StreamGuard &&) = delete;
        StreamGuard &operator=(StreamGuard &&) = delete;

        void sync() const
        {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        [[nodiscard]] operator cudaStream_t() const noexcept
        {
            return stream_;
        }

        [[nodiscard]] cudaStream_t get() const noexcept
        {
            return stream_;
        }

    private:
        cudaStream_t stream_{nullptr};
    };

    class EventGuard final
    {
    public:
        explicit EventGuard(unsigned int flags = cudaEventDefault)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
        }

        ~EventGuard() noexcept
        {
            if (event_ != nullptr)
            {
                cudaEventDestroy(event_);
            }
        }

        EventGuard(const EventGuard &) = delete;
        EventGuard &operator=(const EventGuard &) = delete;
        EventGuard(EventGuard &&) = delete;
        EventGuard &operator=(EventGuard &&) = delete;

        void record(cudaStream_t stream)
        {
            CUDA_CHECK(cudaEventRecord(event_, stream));
        }

        [[nodiscard]] float elapsed_ms(const EventGuard &start) const
        {
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
            return ms;
        }

        [[nodiscard]] operator cudaEvent_t() const noexcept
        {
            return event_;
        }

        [[nodiscard]] cudaEvent_t get() const noexcept
        {
            return event_;
        }

    private:
        cudaEvent_t event_{nullptr};
    };

    class CudaHandles final
    {
    public:
        CudaHandles()
        {
            CUBLAS_CHECK(cublasCreate(&cublas_));
            CUSPARSE_CHECK(cusparseCreate(&cusparse_));
            CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
        }

        ~CudaHandles() noexcept
        {
            if (cublas_)
                cublasDestroy(cublas_);
            if (cusparse_)
                cusparseDestroy(cusparse_);
            if (cusolver_)
                cusolverDnDestroy(cusolver_);
        }

        CudaHandles(const CudaHandles &) = delete;
        CudaHandles &operator=(const CudaHandles &) = delete;
        CudaHandles(CudaHandles &&) = delete;
        CudaHandles &operator=(CudaHandles &&) = delete;

        void set_stream(cudaStream_t s)
        {
            CUBLAS_CHECK(cublasSetStream(cublas_, s));
            CUSPARSE_CHECK(cusparseSetStream(cusparse_, s));
            CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, s));
        }

        [[nodiscard]] cublasHandle_t cublas() const noexcept { return cublas_; }
        [[nodiscard]] cusparseHandle_t cusparse() const noexcept { return cusparse_; }
        [[nodiscard]] cusolverDnHandle_t cusolver() const noexcept { return cusolver_; }

    private:
        cublasHandle_t cublas_{nullptr};
        cusparseHandle_t cusparse_{nullptr};
        cusolverDnHandle_t cusolver_{nullptr};
    };

    [[nodiscard]] inline int query_sm_count(int device = 0) noexcept
    {
        int val = 0;
        cudaDeviceGetAttribute(
            &val, cudaDevAttrMultiProcessorCount, device);
        return val;
    }

    [[nodiscard]] inline std::size_t query_shared_mem_per_block(int device = 0) noexcept
    {
        int val = 0;
        cudaDeviceGetAttribute(
            &val, cudaDevAttrMaxSharedMemoryPerBlock, device);
        return static_cast<std::size_t>(val);
    }

    [[nodiscard]] inline std::size_t query_global_mem(int device = 0) noexcept
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        return prop.totalGlobalMem;
    }

} // namespace holo::cuda
// holo::math::cuda is the canonical namespace for V2. holo::cuda remains as alias.
namespace holo::math { namespace cuda = holo::cuda; }
