#pragma once

#include <cstdint>
#include <thread>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <time.h>
#endif

namespace holo
{

[[nodiscard]] inline std::uint64_t now_ns() noexcept
{
#if defined(__linux__)
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
#else
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

inline void pin_thread(std::thread& t, int core_id) noexcept
{
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(static_cast<std::size_t>(core_id), &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#else
    (void)t; (void)core_id;
#endif
}

inline void sleep_ns(long ns) noexcept
{
#if defined(__linux__)
    struct timespec req{0, ns};
    nanosleep(&req, nullptr);
#else
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
#endif
}

} // namespace holo
