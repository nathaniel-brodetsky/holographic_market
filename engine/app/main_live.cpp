#include <app/live_node.hpp>
#include <execution/binance_fapi.hpp>

#include <memory>

int main()
{
    using namespace holo;

    const LiveConfig cfg{};

    // Inject a paper-trading gateway (empty keys = testnet stub).
    // Swap in BinanceFapiGateway with real keys for live trading.
    RiskManager risk{cfg.max_position_usd};  // shared between node and gateway
    auto gateway = std::make_unique<BinanceFapiGateway>(
        "",          // api_key
        "",          // api_secret
        100.0F,      // order_size_usd
        risk);

    LiveNode node{cfg, std::move(gateway)};
    node.run();

    return 0;
}
