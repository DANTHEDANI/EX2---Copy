import logging
from pathlib import Path

import numpy as np
import torch

from ..env.trading_env import TradingEnv
from ..model.network import DuelingDQNNetwork
from ..shared.config import ConfigManager
from ..shared.constants import ROOT_DIR
from .metrics import calculate_max_drawdown, calculate_sharpe_ratio, calculate_win_rate
from .plots import plot_backtest_results


class BacktestService:
    """Run deterministic policy evaluation on the test split."""

    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.env_cfg = self.config.setup["environment"]
        self.paths_cfg = self.config.setup["paths"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_backtest(
        self,
        states: np.ndarray,
        prices: np.ndarray,
    ) -> dict[str, float | list[float]]:
        env = TradingEnv(
            states_3d=states,
            prices=prices,
            initial_balance=self.env_cfg["initial_balance"],
            commission_fee=self.env_cfg["commission_fee"],
            slippage_fee=self.env_cfg["slippage_fee"],
            sharpe_lambda=self.env_cfg["sharpe_lambda"],
            invalid_action_penalty=self.env_cfg["invalid_action_penalty"],
        )
        model = DuelingDQNNetwork(action_dim=env.action_space.n).to(self.device)
        model_path = (
            ROOT_DIR
            / Path(self.paths_cfg["model_dir"])
            / self.paths_cfg["model_filename"]
        )
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        state, info = env.reset()
        equities, actions, rewards = [info["equity"]], [], []
        while True:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state_t).argmax(dim=1).item())
            state, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            equities.append(info["equity"])
            if terminated or truncated:
                break
        eq = np.asarray(equities, dtype=float)
        metrics = {
            "final_equity": float(eq[-1]),
            "sharpe_ratio": calculate_sharpe_ratio(
                eq,
                self.config.setup["backtest"]["risk_free_rate"],
            ),
            "max_drawdown": calculate_max_drawdown(eq),
            "win_rate": calculate_win_rate(np.asarray(actions), np.asarray(rewards)),
        }
        plot_backtest_results(
            {"agent_equity": list(eq), "bnh_equity": self._buy_and_hold(prices, eq[0])},
            self.paths_cfg["results_dir"],
        )
        self.logger.info(
            "Backtest done. sharpe=%.3f mdd=%.3f",
            metrics["sharpe_ratio"],
            metrics["max_drawdown"],
        )
        return {**metrics, "equity_curve": list(eq)}

    @staticmethod
    def _buy_and_hold(prices: np.ndarray, initial_equity: float) -> list[float]:
        first = max(float(prices[0]), 1e-8)
        shares = initial_equity / first
        return [float(shares * p) for p in prices]
