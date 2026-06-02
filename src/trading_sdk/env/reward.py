from collections import deque

import numpy as np


class RewardFunction:
    """Computes the operational step reward incorporating Sharpe penalization."""

    def __init__(
        self,
        lambda_sharpe: float = 0.1,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
    ) -> None:
        self.lambda_sharpe = lambda_sharpe
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.returns_history = deque(maxlen=252)

    def calculate(self, prev_equity: float, current_equity: float, trade_value: float) -> float:
        """Implements r_t = DeltaV_t - C_t - S_t + lambda * Sharpe_t."""
        delta_v = current_equity - prev_equity

        cost = self.commission_rate * trade_value
        slippage = self.slippage_rate * trade_value

        daily_return = delta_v / (prev_equity + 1e-8)
        self.returns_history.append(daily_return)

        sharpe = 0.0
        if len(self.returns_history) > 1:
            ret_array = np.array(self.returns_history)
            std = ret_array.std() + 1e-8
            mean = ret_array.mean()
            sharpe = (mean / std) * np.sqrt(252)

        reward = delta_v - cost - slippage + (self.lambda_sharpe * sharpe)
        return float(reward)

    def reset(self) -> None:
        self.returns_history.clear()
