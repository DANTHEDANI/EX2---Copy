import numpy as np


def calculate_total_return(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])


def calculate_sharpe_ratio(equity_curve: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0
    return float((np.mean(returns) - risk_free_rate) / std_ret * np.sqrt(252))


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + 1e-8)
    return float(np.min(drawdowns))


def calculate_win_rate(actions: np.ndarray, rewards: np.ndarray) -> float:
    active_mask = actions != 1
    if not np.any(active_mask):
        return 0.0
    wins = np.sum((rewards > 0) & active_mask)
    return float(wins / np.sum(active_mask))
