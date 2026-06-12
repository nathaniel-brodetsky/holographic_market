// app/main_backtest.cpp
// Phase VI stub: historical LOB CSV replay → full pipeline → PnL/Sharpe.
// Replaces BinanceFeedHandler with CsvReplayHandler (same ring interface).
// TODO: implement CsvReplayHandler in net/csv_replay.hpp

#include <core/memory_arena.hpp>
#include <core/lockfree_ring_buffer.hpp>
#include <core/lob_core.hpp>
#include <math/cuda_pipeline.cuh>
#include <net/signal_router.hpp>

#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::puts("Usage: holographic_backtest <lob_data.csv>");
        return 1;
    }
    std::printf("Backtest stub — CSV: %s\n", argv[1]);
    std::puts("Phase VI not yet implemented.");
    return 0;
}
