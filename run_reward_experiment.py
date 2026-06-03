import os

import matplotlib.pyplot as plt

from src.trading_sdk.sdk import TradingSDK


def run_experiment():
    print("Starting Reward Function Experiment...")
    sdk = TradingSDK()
    ticker = "AAPL"

    # Configure less episodes for quick experiment
    sdk.config_manager.setup["hyperparameters"]["episodes"] = 5

    # 1. Basic Reward (No friction, no risk penalty)
    print("\n--- Running Basic Reward (Profit Only) ---")
    sdk.trainer.env_cfg["commission_fee"] = 0.0
    sdk.trainer.env_cfg["slippage_fee"] = 0.0
    sdk.trainer.env_cfg["sharpe_lambda"] = 0.0
    sdk.backtester.env_cfg["commission_fee"] = 0.0
    sdk.backtester.env_cfg["slippage_fee"] = 0.0

    sdk.run_training_pipeline(ticker)

    # Get test data for backtest directly to retrieve metrics
    cfg = sdk.config_manager.setup["data"]
    raw_df = sdk.data_client.download_ticker(
        ticker, cfg["start_date"], cfg["end_date"], cfg["interval"]
    )
    states, prices = sdk._build_states_and_prices(raw_df)
    _, _, test_states = sdk.feature_engineer.split_data(states)
    _, _, test_prices = sdk.feature_engineer.split_data(prices)

    basic_metrics = sdk.backtester.run_backtest(test_states, test_prices)
    basic_equity = basic_metrics["equity_curve"]

    # 2. Advanced Reward (With friction and risk penalty)
    print("\n--- Running Advanced Reward (Friction + Sharpe) ---")
    sdk.trainer.env_cfg["commission_fee"] = 0.001
    sdk.trainer.env_cfg["slippage_fee"] = 0.001
    sdk.trainer.env_cfg["sharpe_lambda"] = 1.0
    sdk.backtester.env_cfg["commission_fee"] = 0.001
    sdk.backtester.env_cfg["slippage_fee"] = 0.001

    sdk.run_training_pipeline(ticker)

    advanced_metrics = sdk.backtester.run_backtest(test_states, test_prices)
    advanced_equity = advanced_metrics["equity_curve"]
    # We need to compute bnh_equity since backtester doesn't return it in the dict directly
    first = max(float(test_prices[0]), 1e-8)
    shares = advanced_equity[0] / first
    bnh_equity = [float(shares * p) for p in test_prices]

    # Plot Comparison
    print("\nPlotting Comparison...")
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(basic_equity, label="Basic Reward (No Costs)", color="red", linestyle="--")
    plt.plot(advanced_equity, label="Advanced Reward (Costs + Sharpe)", color="blue")
    plt.plot(bnh_equity, label="Buy & Hold", color="orange", alpha=0.6)

    plt.title("Reward Function Comparison: Basic vs Advanced")
    plt.xlabel("Test Steps")
    plt.ylabel("Portfolio Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reward_comparison.png"))
    plt.close()
    print("Experiment Complete. Graph saved to data/results/reward_comparison.png")


if __name__ == "__main__":
    run_experiment()
