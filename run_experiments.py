#!/usr/bin/env python3
"""Comparative experiment runner: Train/backtest on multiple tickers."""
import sys

from src.trading_sdk.sdk import TradingSDK


def run_comparative_experiments() -> None:
    """Run training and backtesting on AAPL (primary) and SPY (comparative)."""
    sdk = TradingSDK()
    
    # Primary experiment: AAPL
    print("\n" + "="*70)
    print("PRIMARY EXPERIMENT: AAPL (2020-01-01 to 2023-01-01)")
    print("="*70)
    sdk.run_training_pipeline("AAPL")
    sdk.evaluate_strategy("AAPL")
    
    # Comparative experiment: SPY
    print("\n" + "="*70)
    print("COMPARATIVE EXPERIMENT: SPY (same date range, same mechanism)")
    print("="*70)
    try:
        sdk.run_training_pipeline("SPY")
        sdk.evaluate_strategy("SPY")
    except Exception as exc:  # noqa: BLE001
        print(f"SPY experiment failed: {exc}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("All experiments completed successfully.")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_comparative_experiments()
