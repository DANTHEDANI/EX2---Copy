# Educational Dueling DQN Trading System

An academic Deep Reinforcement Learning project that teaches how to build a **Dueling DQN** agent for stock trading on historical Yahoo Finance data (AAPL).  
This repository prioritizes **correct RL formulation, software architecture, testability, and reproducibility** over financial promises.

---

## 1) Project Overview

The system learns a trading policy in a custom Gymnasium-style environment using a rolling market history.  
Objective: understand how a Dueling DQN can map market state to actions in a constrained all-in/all-out setup.

| Property | Value |
|---|---|
| Task type | Educational RL (not production trading advice) |
| Agent | Dueling Deep Q-Network (PyTorch) |
| Market data | Yahoo Finance OHLCV |
| Main ticker | `AAPL` |
| Data period | Config-driven (`config/setup.json`) |

---

## 2) RL Problem Formulation

### State Space

At each step, the environment provides a tensor:

$$
s_t \in \mathbb{R}^{30 \times 10}
$$

- **30 rows**: a chronological sliding window of recent timesteps.
- **10 columns**: engineered features:
  1. `log_return`
  2. `rsi_14`
  3. `macd`
  4. `macd_signal`
  5. `macd_hist`
  6. `bb_pct`
  7. `vwap_dist`
  8. `volume_norm`
  9. `position`
  10. `unrealised_pnl`

Why rolling windows? A single candle is insufficient for regime context. A 30-step sequence allows the model to infer trend, momentum, volatility, and position-dependent behavior.

**Data leakage prevention:** feature scaling uses **expanding normalization** (past-only min/max up to current index), never future statistics.

### Action Space

Discrete action set:

| Action ID | Meaning | Portfolio effect |
|---|---|---|
| `0` | SELL | If holding, liquidate all shares |
| `1` | HOLD | Keep current position |
| `2` | BUY | If flat, invest all cash |

This is a strict **all-in/all-out** formulation to keep action semantics simple and pedagogically clear.

### Reward Function

Reward is defined as:

$$
r_t = \Delta V_t - C_t - S_t + \lambda \cdot \text{Sharpe}_t
$$

Where:
- $\Delta V_t$: portfolio equity change,
- $C_t$: transaction cost,
- $S_t$: slippage cost,
- $\lambda \cdot \text{Sharpe}_t$: risk-adjusted stability bonus.

This is critical in RL trading: raw profit-only rewards often produce unstable, over-trading policies. Penalizing friction discourages churn, while Sharpe-based shaping nudges toward smoother return profiles.

---

## 3) Dueling DQN Architecture

The network uses a Conv1D temporal backbone, then splits into:

- **Value stream** $V(s)$: how good the state is overall.
- **Advantage stream** $A(s,a)$: how much better/worse an action is in that state.

Aggregation:

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a')
$$

Why Conv1D? It learns local temporal motifs (short-term momentum, reversals, volatility bursts) efficiently across the 30-step sequence.

---

## 4) Training Stability Mechanisms

- **Experience Replay Buffer**: samples shuffled transitions to reduce temporal correlation and improve sample efficiency.
- **Target Network**: computes Bellman targets with delayed/frozen weights to stabilize learning targets.
- **Epsilon-Greedy Decay**: balances exploration early and exploitation later.
- **Huber/MSE Config Toggle**: robust loss selection via config.

---

## 5) Software Architecture (OOP + SDK Facade)

The code is organized as a strict layered system:

| Layer | Responsibility |
|---|---|
| `data/` | Download + cache + feature engineering |
| `env/` | Gymnasium trading dynamics + reward computation |
| `model/` | Dueling DQN network definition |
| `memory/` | Replay buffer |
| `services/` | Training/backtesting orchestration + metrics + plots |
| `sdk.py` | `TradingSDK` facade (single entrypoint for consumers) |

**Facade rule:** CLI and any future UI must call `TradingSDK`, not internal layers directly.  
**Quality rule:** all source/test files stay under the `<150 lines` architectural constraint.

---

## 6) Installation & Usage

### Install

```bash
uv sync --all-groups
```

### Lint + Tests

```bash
uv run ruff check .
uv run pytest
```

### Train

```bash
uv run trading-sdk --action train --ticker AAPL
```

### Backtest

```bash
uv run trading-sdk --action backtest --ticker AAPL
```

---

## 7) Results & Visualizations

Generated artifacts are saved in `data/results/`.

![Learning Curve](data/results/learning_curve.png)
![Backtest Results](data/results/backtest_results.png)

---

## Project Tree (Condensed)

```text
config/
data/
docs/
src/trading_sdk/
tests/
```

---

## Notes

- All runtime parameters are config-driven via `config/setup.json`.
- This repository is intended for **education and experimentation**.
