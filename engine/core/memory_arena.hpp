#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <span>
#include <type_traits>
#include <cassert>
#include <new>

namespace holo::core
{
    static constexpr std::size_t k_cache_line = 64U;
    static constexpr std::size_t k_page_size = 4096U;
    static constexpr std::size_t k_arena_align = k_page_size;

    template <typename T>
    concept ArenaAllocatable =
        std::is_trivially_destructible_v<T> &&
        std::is_standard_layout_v<T>;

    [[nodiscard]] constexpr std::size_t align_up(
        std::size_t value,
        std::size_t alignment) noexcept
    {
        assert((alignment & (alignment - 1U)) == 0U);
        return (value + alignment - 1U) & ~(alignment - 1U);
    }

    class MemoryArena final
    {
    public:
        explicit MemoryArena(std::size_t capacity_bytes)
            : capacity_{align_up(capacity_bytes, k_arena_align)}, base_{
                                                                      static_cast<std::byte *>(
                                                                          ::operator new(capacity_, std::align_val_t{k_arena_align}))},
              cursor_{0U}
        {
            std::memset(base_, 0, capacity_);
        }

        ~MemoryArena() noexcept
        {
            ::operator delete(base_, std::align_val_t{k_arena_align});
        }

        MemoryArena(const MemoryArena &) = delete;

        MemoryArena &operator=(const MemoryArena &) = delete;

        MemoryArena(MemoryArena &&) = delete;

        MemoryArena &operator=(MemoryArena &&) = delete;

        template <ArenaAllocatable T>
        [[nodiscard]] T *alloc(std::size_t count = 1U) noexcept
        {
            const std::size_t alignment = alignof(T) < k_cache_line
                                              ? alignof(T)
                                              : k_cache_line;
            const std::size_t aligned_cursor = align_up(cursor_, alignment);
            const std::size_t required = sizeof(T) * count;

            if (aligned_cursor + required > capacity_) [[unlikely]]
            {
                return nullptr;
            }

            T *const ptr = reinterpret_cast<T *>(base_ + aligned_cursor);
            cursor_ = aligned_cursor + required;
            return ptr;
        }

        template <ArenaAllocatable T>
        [[nodiscard]] std::span<T> alloc_span(std::size_t count) noexcept
        {
            T *const ptr = alloc<T>(count);
            if (ptr == nullptr) [[unlikely]]
            {
                return {};
            }
            return std::span<T>{ptr, count};
        }

        template <ArenaAllocatable T, typename... Args>
            requires std::is_constructible_v<T, Args...>
        [[nodiscard]] T *emplace(Args &&...args) noexcept
        {
            T *const ptr = alloc<T>(1U);
            if (ptr == nullptr) [[unlikely]]
            {
                return nullptr;
            }
            ::new (ptr) T(static_cast<Args &&>(args)...);
            return ptr;
        }

        void reset() noexcept { cursor_ = 0U; }

        [[nodiscard]] std::size_t used() const noexcept { return cursor_; }
        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] std::size_t remaining() const noexcept { return capacity_ - cursor_; }

        [[nodiscard]] double utilization() const noexcept
        {
            return static_cast<double>(cursor_) / static_cast<double>(capacity_);
        }

        [[nodiscard]] const std::byte *base_ptr() const noexcept { return base_; }

    private:
        std::size_t capacity_;
        std::byte *base_;
        std::size_t cursor_;
    };
} // namespace holo::core

// Backward-compatibility: holo:: still resolves to holo::core:: for existing code.
namespace holo { using namespace holo::core; }
