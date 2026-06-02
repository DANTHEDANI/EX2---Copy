#!/usr/bin/env python3
"""Data verification script: Load, engineer, and split data from Yahoo Finance."""
import sys
from pathlib import Path

import pandas as pd

from src.trading_sdk.data.client import YFinanceDataClient
from src.trading_sdk.data.preprocessor import FeatureEngineer
from src.trading_sdk.shared.config import ConfigManager


def main() -> None:
    """Verify data loading, features, and chronological splits."""
    config = ConfigManager()
    data_cfg = config.setup["data"]
    client = YFinanceDataClient(config)
    engineer = FeatureEngineer(
        window_size=config.setup["environment"]["window_size"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
    )
    
    ticker = "AAPL"
    start = data_cfg["start_date"]
    end = data_cfg["end_date"]
    interval = data_cfg["interval"]
    
    print(f"\n{'='*70}")
    print(f"Data Verification: {ticker} ({start} to {end})")
    print(f"{'='*70}\n")
    
    raw_df = client.download_ticker(ticker, start, end, interval)
    if raw_df is None or raw_df.empty:
        print(f"ERROR: Could not load data for {ticker}")
        sys.exit(1)
    
    print(f"[OK] Downloaded {len(raw_df)} rows from Yahoo Finance\n")
    
    print("First 5 raw OHLCV rows:")
    print(raw_df[["Open", "High", "Low", "Close", "Volume"]].head())
    print()
    
    states = engineer.engineer_features(raw_df)
    print(f"[OK] Engineered features: {states.shape}")
    print(f"  - Windows (N): {states.shape[0]}")
    print(f"  - Window size: {states.shape[1]}")
    print(f"  - Features: {states.shape[2]}")
    print()
    
    # Reconstruct feature frame for display
    frame = engineer._compute_feature_frame(raw_df)
    feature_cols = [
        "log_return", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_pct", "vwap_dist", "volume_norm", "position", "unrealised_pnl",
    ]
    print("First 5 engineered feature rows:")
    print(frame[feature_cols].head())
    print()
    
    train_states, val_states, test_states = engineer.split_data(states)
    print(f"[OK] Chronological split (no shuffle):")
    print(f"  - Train: {len(train_states)} windows ({len(train_states)/len(states)*100:.1f}%)")
    print(f"  - Val:   {len(val_states)} windows ({len(val_states)/len(states)*100:.1f}%)")
    print(f"  - Test:  {len(test_states)} windows ({len(test_states)/len(states)*100:.1f}%)")
    print(f"  - Total: {len(states)} windows\n")
    
    print(f"[OK] Verification complete. Data is ready for training.\n")


if __name__ == "__main__":
    main()
