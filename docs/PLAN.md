# Architecture Plan & Strategy

## Implemented Architecture

### 1) Data Layer
- `YFinanceDataClient`: downloads and caches market data locally.
- `FeatureEngineer`: computes 10 sequential features and produces `(30, 10)` windows.
- Leakage prevention via expanding normalization and chronological split.

### 2) Environment Layer
- `TradingEnv` tracks cash, shares, position state, and equity transitions.
- Action semantics enforce all-in/all-out behavior.
- `RewardFunction` applies transaction/slippage penalties and Sharpe shaping.

### 3) Model Layer
- `DuelingDQNNetwork` uses Conv1D temporal extraction.
- Split streams: value `V(s)` and advantage `A(s,a)`.
- Aggregation follows dueling equation for stable action-value decomposition.

### 4) Service Layer
- `TrainingService`: replay buffer sampling, epsilon-greedy decay, Bellman target, target network update.
- `BacktestService`: deterministic evaluation (`epsilon=0`) with Sharpe, max drawdown, win rate, and equity curve output.
- Helper modules: `metrics.py` and `plots.py`.

### 5) Facade Layer
- `TradingSDK` is the only integration gateway for CLI/UI.
- CLI delegates to `TradingSDK` actions: `train` and `backtest`.

## Operational Flow

1. Fetch/cache raw OHLCV data.
2. Engineer features and generate rolling states.
3. Chronologically split into train/val/test.
4. Train Dueling DQN with replay + target stabilization.
5. Backtest on test split and save visual outputs.
