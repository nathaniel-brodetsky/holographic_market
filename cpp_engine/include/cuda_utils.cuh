#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

namespace holo::cuda {
#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            ::std::fprintf(stderr,                                          \
                "[CUDA] %s:%d  %s  →  %s\n",                              \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));         \
            ::std::abort();                                                 \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(expr)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (expr);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            ::std::fprintf(stderr,                                          \
                "[cuBLAS] %s:%d  %s  →  status %d\n",                     \
                __FILE__, __LINE__, #expr, static_cast<int>(_s));           \
            ::std::abort();                                                 \
        }                                                                   \
    } while (0)

#define CUSPARSE_CHECK(expr)                                                \
    do {                                                                    \
        cusparseStatus_t _s = (expr);                                       \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                \
            ::std::fprintf(stderr,                                          \
                "[cuSPARSE] %s:%d  %s  →  %s\n",                          \
                __FILE__, __LINE__, #expr,                                  \
                cusparseGetErrorString(_s));                                \
            ::std::abort();                                                 \
        }                                                                   \
    } while (0)

#define CUSOLVER_CHECK(expr)                                                \
    do {                                                                    \
        cusolverStatus_t _s = (expr);                                       \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                \
            ::std::fprintf(stderr,                                          \
                "[cuSOLVER] %s:%d  %s  →  status %d\n",                   \
                __FILE__, __LINE__, #expr, static_cast<int>(_s));           \
            ::std::abort();                                                 \
        }                                                                   \
    } while (0)

    struct CudaHandles {
        cublasHandle_t cublas{nullptr};
        cusparseHandle_t cusparse{nullptr};
        cusolverDnHandle_t cusolver{nullptr};

        CudaHandles() {
            CUBLAS_CHECK(cublasCreate(&cublas));
            CUSPARSE_CHECK(cusparseCreate(&cusparse));
            CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        }

        ~CudaHandles() {
            if (cublas) cublasDestroy(cublas);
            if (cusparse) cusparseDestroy(cusparse);
            if (cusolver) cusolverDnDestroy(cusolver);
        }

        CudaHandles(const CudaHandles &) = delete;

        CudaHandles &operator=(const CudaHandles &) = delete;

        CudaHandles(CudaHandles &&) = delete;

        CudaHandles &operator=(CudaHandles &&) = delete;
    };

    [[nodiscard]] inline int query_sm_count() noexcept {
        int device = 0;
        int sm_count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count,
                               cudaDevAttrMultiProcessorCount, device);
        return sm_count;
    }

    [[nodiscard]] inline int query_warp_size() noexcept {
        int device = 0;
        int warp = 32;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, device);
        return warp;
    }

    [[nodiscard]] inline size_t query_shared_mem_per_block() noexcept {
        int device = 0;
        int smem = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&smem,
                               cudaDevAttrMaxSharedMemoryPerBlock, device);
        return static_cast<size_t>(smem);
    }

    template<typename T>
    [[nodiscard]] T *device_alloc(size_t count) {
        T *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr),
            sizeof(T) * count));
        return ptr;
    }

    template<typename T>
    void device_free(T *ptr) noexcept {
        if (ptr) cudaFree(ptr);
    }

    template<typename T>
    [[nodiscard]] T *pinned_alloc(size_t count) {
        T *ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr),
            sizeof(T) * count));
        return ptr;
    }

    template<typename T>
    void pinned_free(T *ptr) noexcept {
        if (ptr) cudaFreeHost(ptr);
    }

    struct StreamGuard {
        cudaStream_t stream{nullptr};

        StreamGuard() {
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking));
        }

        ~StreamGuard() noexcept {
            if (stream) cudaStreamDestroy(stream);
        }

        StreamGuard(const StreamGuard &) = delete;

        StreamGuard &operator=(const StreamGuard &) = delete;

        void sync() const {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        operator cudaStream_t() const noexcept { return stream; }
    };

    struct EventGuard {
        cudaEvent_t event{nullptr};

        explicit EventGuard(unsigned flags = cudaEventDefault) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
        }

        ~EventGuard() noexcept {
            if (event) cudaEventDestroy(event);
        }

        EventGuard(const EventGuard &) = delete;

        EventGuard &operator=(const EventGuard &) = delete;

        void record(cudaStream_t stream) {
            CUDA_CHECK(cudaEventRecord(event, stream));
        }

        float elapsed_ms(const EventGuard &start) const {
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start.event, event));
            return ms;
        }

        operator cudaEvent_t() const noexcept { return event; }
    };
} // namespace holo::cuda
