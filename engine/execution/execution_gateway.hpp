#pragma once
#include <trading/execution_types.hpp>
#include <core/lockfree_ring_buffer.hpp>

#include <common/types.hpp>

namespace holo
{

class ExecutionGateway
{
public:
    virtual ~ExecutionGateway() = default;

    virtual void start() = 0;
    virtual void stop()  noexcept = 0;

    // Submit a pair of orders for a routed edge.
    // mid_src / mid_dst are current mid-prices for sizing.
    virtual void submit(
        const RoutedEdge& edge,
        float             mid_src,
        float             mid_dst) noexcept = 0;
};

} // namespace holo
