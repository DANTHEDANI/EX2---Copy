from typing import Any

import gymnasium as gym
import numpy as np

from .reward import RewardFunction


class TradingEnv(gym.Env):
    """All-in/all-out stock env with portfolio-aware state updates."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        states_3d: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 10000.0,
        commission_fee: float = 0.001,
        slippage_fee: float = 0.001,
        sharpe_lambda: float = 0.1,
        invalid_action_penalty: float = 5.0,
    ) -> None:
        super().__init__()
        self.states = states_3d
        self.prices = prices[: len(states_3d)]
        self.max_steps = max(0, len(self.states) - 1)
        self.initial_balance = initial_balance
        self.invalid_action_penalty = invalid_action_penalty
        self.reward_fn = RewardFunction(sharpe_lambda, commission_fee, slippage_fee)
        self.action_space = gym.spaces.Discrete(3)
        window_size = int(states_3d.shape[1])
        features_count = int(states_3d.shape[2])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, features_count),
            dtype=np.float32,
        )
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.position = 0
        self.avg_price = 0.0
        self.reward_fn.reset()
        return self._get_state(), self._get_info()

    def _get_state(self) -> np.ndarray:
        state_window = np.copy(self.states[self.current_step])
        current_price = self.prices[self.current_step]
        unrealised_pnl = (
            0.0 if self.avg_price <= 0 else (current_price - self.avg_price) / self.avg_price
        )
        state_window[-1, 8] = float(self.position)
        state_window[-1, 9] = float(unrealised_pnl if self.position else 0.0)
        return state_window.astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        current_price = self.prices[self.current_step]
        equity = self.cash + (self.shares * current_price if self.position else 0.0)
        return {"step": self.current_step, "equity": equity, "position": self.position}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        current_price = self.prices[self.current_step]
        prev_equity = self.cash + (self.shares * current_price if self.position else 0.0)
        trade_value = 0.0
        penalty = 0.0

        if action == 2:
            if self.position == 0:
                trade_value = self.cash
                self.shares = trade_value / current_price
                self.cash = 0.0
                self.avg_price = current_price
                self.position = 1
            else:
                penalty = -self.invalid_action_penalty
        elif action == 0:
            if self.position:
                trade_value = self.shares * current_price
                self.cash += trade_value
                self.shares = 0.0
                self.position = 0
                self.avg_price = 0.0
            else:
                penalty = -self.invalid_action_penalty

        self.current_step += 1
        next_price = self.prices[min(self.current_step, self.max_steps)]
        current_equity = self.cash + (self.shares * next_price if self.position else 0.0)
        reward = self.reward_fn.calculate(prev_equity, current_equity, trade_value) + penalty
        terminated = (
            current_equity <= (self.initial_balance * 0.1)
            or self.current_step >= self.max_steps
        )
        return self._get_state(), float(reward), terminated, False, self._get_info()
