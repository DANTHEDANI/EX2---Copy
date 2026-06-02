# Project Requirements Document (PRD)

## Topic Focus

Build an educational **Dueling DQN** trading system that demonstrates end-to-end RL architecture, not speculative trading promises.

## Final Product Scope

| Area | Requirement |
|---|---|
| Data | Yahoo Finance ingestion with local cache (Parquet/CSV) |
| Features | 10 engineered indicators + leakage-safe scaling |
| Environment | Gymnasium-compatible all-in/all-out portfolio simulator |
| Agent | Conv1D-based Dueling DQN |
| Services | Training + Backtesting + metrics + plots |
| Access | `TradingSDK` facade as the single gateway |

## Non-Negotiable Engineering Constraints

1. **Facade architecture:** all application flows route through `TradingSDK`.
2. **No hardcoded business parameters:** config-driven from `config/setup.json`.
3. **Chronological data splits:** `70/15/15` (train/val/test), no shuffle.
4. **File size guardrail:** `<150 lines` per implementation/test file.
5. **Quality gates:** strict linting and high automated test coverage.

## RL Formulation Requirements

- **State:** `(30, 10)` rolling tensor.
- **Actions:** `0=SELL, 1=HOLD, 2=BUY`.
- **Reward:** $r_t = \Delta V_t - C_t - S_t + \lambda \cdot \text{Sharpe}_t$.
- **Stability tools:** replay buffer + target network Bellman targets.

## Success Criteria

- User can train and backtest via one-command CLI (`uv run trading-sdk ...`).
- Generated outputs include learning and backtest plots.
- Codebase remains readable, modular, and pedagogically explainable.
