#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <optional>
#include <type_traits>
#include <memory_arena.hpp>

namespace holo
{
    template <typename T>
    concept RingElement =
        std::is_trivially_copyable_v<T> &&
        std::is_trivially_destructible_v<T>;

    template <RingElement T, std::size_t Capacity>
        requires(Capacity >= 2U) && ((Capacity & (Capacity - 1U)) == 0U)
    class alignas(k_cache_line) SpscRingBuffer final
    {
    public:
        static constexpr std::size_t k_capacity = Capacity;
        static constexpr std::size_t k_mask = Capacity - 1U;
        static constexpr std::size_t k_slot_size = sizeof(T);

        static_assert(
            (k_capacity & k_mask) == 0U,
            "SpscRingBuffer capacity must be a power of two.");

        SpscRingBuffer() noexcept
            : head_{0U}, _pad0_{}, tail_{0U}, _pad1_{}
        {
        }

        SpscRingBuffer(const SpscRingBuffer &) = delete;

        SpscRingBuffer &operator=(const SpscRingBuffer &) = delete;

        SpscRingBuffer(SpscRingBuffer &&) = delete;

        SpscRingBuffer &operator=(SpscRingBuffer &&) = delete;

        [[nodiscard]] bool try_push(const T &item) noexcept
        {
            const std::size_t head = head_.load(std::memory_order_relaxed);
            const std::size_t next = (head + 1U) & k_mask;

            if (next == tail_.load(std::memory_order_acquire)) [[unlikely]]
            {
                return false;
            }

            slots_[head] = item;
            head_.store(next, std::memory_order_release);
            return true;
        }

        [[nodiscard]] bool try_push(T &&item) noexcept
        {
            const std::size_t head = head_.load(std::memory_order_relaxed);
            const std::size_t next = (head + 1U) & k_mask;

            if (next == tail_.load(std::memory_order_acquire)) [[unlikely]]
            {
                return false;
            }

            slots_[head] = static_cast<T &&>(item);
            head_.store(next, std::memory_order_release);
            return true;
        }

        [[nodiscard]] bool try_pop(T &out) noexcept
        {
            const std::size_t tail = tail_.load(std::memory_order_relaxed);

            if (tail == head_.load(std::memory_order_acquire)) [[unlikely]]
            {
                return false;
            }

            out = slots_[tail];
            tail_.store((tail + 1U) & k_mask, std::memory_order_release);
            return true;
        }

        void push_blocking(const T &item) noexcept
        {
            while (!try_push(item)) [[unlikely]]
            {
                __builtin_ia32_pause();
            }
        }

        void push_blocking(T &&item) noexcept
        {
            while (!try_push(static_cast<T &&>(item))) [[unlikely]]
            {
                __builtin_ia32_pause();
            }
        }

        void pop_blocking(T &out) noexcept
        {
            while (!try_pop(out)) [[unlikely]]
            {
                __builtin_ia32_pause();
            }
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
        }

        [[nodiscard]] std::size_t size_approx() const noexcept
        {
            const std::size_t h = head_.load(std::memory_order_relaxed);
            const std::size_t t = tail_.load(std::memory_order_relaxed);
            return (h - t + k_capacity) & k_mask;
        }

        [[nodiscard]] static constexpr std::size_t capacity() noexcept
        {
            return k_capacity - 1U;
        }

    private:
        alignas(k_cache_line) std::atomic<std::size_t> head_;
        std::byte _pad0_[k_cache_line - sizeof(std::atomic<std::size_t>)];

        alignas(k_cache_line) std::atomic<std::size_t> tail_;
        std::byte _pad1_[k_cache_line - sizeof(std::atomic<std::size_t>)];

        alignas(k_cache_line) T slots_[k_capacity];
    };

    template <RingElement T>
    class alignas(k_cache_line) DynamicSpscRingBuffer final
    {
    public:
        DynamicSpscRingBuffer(MemoryArena &arena, std::size_t capacity)
            : capacity_{capacity}, mask_{capacity - 1U}, slots_{arena.alloc_span<T>(capacity)}, head_{0U}, _pad0_{}, tail_{0U}, _pad1_{}
        {
            assert((capacity & mask_) == 0U && "Capacity must be power of two.");
            assert(!slots_.empty() && "Arena allocation failed.");
        }

        DynamicSpscRingBuffer(const DynamicSpscRingBuffer &) = delete;

        DynamicSpscRingBuffer &operator=(const DynamicSpscRingBuffer &) = delete;

        DynamicSpscRingBuffer(DynamicSpscRingBuffer &&) = delete;

        DynamicSpscRingBuffer &operator=(DynamicSpscRingBuffer &&) = delete;

        [[nodiscard]] bool try_push(const T &item) noexcept
        {
            const std::size_t head = head_.load(std::memory_order_relaxed);
            const std::size_t next = (head + 1U) & mask_;

            if (next == tail_.load(std::memory_order_acquire)) [[unlikely]]
            {
                return false;
            }

            slots_[head] = item;
            head_.store(next, std::memory_order_release);
            return true;
        }

        [[nodiscard]] bool try_pop(T &out) noexcept
        {
            const std::size_t tail = tail_.load(std::memory_order_relaxed);

            if (tail == head_.load(std::memory_order_acquire)) [[unlikely]]
            {
                return false;
            }

            out = slots_[tail];
            tail_.store((tail + 1U) & mask_, std::memory_order_release);
            return true;
        }

        void push_blocking(const T &item) noexcept
        {
            while (!try_push(item)) [[unlikely]]
            {
                __builtin_ia32_pause();
            }
        }

        void pop_blocking(T &out) noexcept
        {
            while (!try_pop(out)) [[unlikely]]
            {
                __builtin_ia32_pause();
            }
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
        }

        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_ - 1U; }

    private:
        const std::size_t capacity_;
        const std::size_t mask_;
        std::span<T> slots_;

        alignas(k_cache_line) std::atomic<std::size_t> head_;
        std::byte _pad0_[k_cache_line - sizeof(std::atomic<std::size_t>)];

        alignas(k_cache_line) std::atomic<std::size_t> tail_;
        std::byte _pad1_[k_cache_line - sizeof(std::atomic<std::size_t>)];
    };
}
