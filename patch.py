import os

base = r"c:\Users\danie\Documents\VIBE_CODING\EX2\src\trading_sdk"

preprocessor_code = '''import numpy as np
import pandas as pd
from typing import Tuple

class FeatureEngineer:
    """Preprocesses raw OHLCV market data into 3D state tensors for a DQN agent."""
    
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.features_count = 10

    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Generates the 10 required features strictly avoiding data leakage."""
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        c, v = df["Close"], df["Volume"]
        
        df["log_return"] = np.log(c / c.shift(1))
        
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        
        ema_12 = c.ewm(span=12, adjust=False).mean()
        ema_26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        sma_20 = c.rolling(20).mean()
        std_20 = c.rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        df["bb_pct"] = (c - lower) / (upper - lower + 1e-8)
        
        typ_price = (df["High"] + df["Low"] + c) / 3.0
        rolling_vp = (typ_price * v).rolling(14).sum()
        rolling_v = v.rolling(14).sum()
        vwap = rolling_vp / (rolling_v + 1e-8)
        df["vwap_dist"] = (c - vwap) / (vwap + 1e-8)
        
        vol_sma = v.rolling(14).mean()
        df["volume_norm"] = v / (vol_sma + 1e-8)
        
        df["position"] = 0.0
        df["unrealised_pnl"] = 0.0
        
        df.dropna(inplace=True)
        
        features = ["log_return", "rsi_14", "macd", "macd_signal", "macd_hist",
                    "bb_pct", "vwap_dist", "volume_norm", "position", "unrealised_pnl"]
        
        # Expanding normalisation avoids future lookahead (Min-Max style bounds)
        for col in features[:-2]:
            roll_min = df[col].expanding(min_periods=1).min()
            roll_max = df[col].expanding(min_periods=1).max()
            df[col] = (df[col] - roll_min) / (roll_max - roll_min + 1e-8)
            
        data_mat = df[features].to_numpy(dtype=np.float32)
        
        num_windows = len(data_mat) - self.window_size + 1
        states = np.array([
            data_mat[i : i + self.window_size] 
            for i in range(num_windows)
        ], dtype=np.float32)
        
        return states

    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronologically splits the 3D tensor to strictly prevent data leakage."""
        n = len(data)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        return data[:train_end], data[train_end:val_end], data[val_end:]
'''

reward_code = '''import numpy as np
from collections import deque

class RewardFunction:
    """Computes the operational step reward incorporating Sharpe penalization."""
    
    def __init__(self, lambda_sharpe: float = 0.1, commission_rate: float = 0.001, slippage_rate: float = 0.001) -> None:
        self.lambda_sharpe = lambda_sharpe
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.returns_history = deque(maxlen=252)

    def calculate(
        self, prev_equity: float, current_equity: float, trade_value: float
    ) -> float:
        """
        Calculates: r_t = ∆V_t - C_t - S_t + λ * Sharpe_t
        """
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
'''

env_code = '''import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
from .reward import RewardFunction

class TradingEnv(gym.Env):
    """
    Gymnasium environment simulating a stock trading market for RL agent.
    Receives pre-windowed 3D state data.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, states_3d: np.ndarray, prices: np.ndarray = None, initial_balance: float = 10000.0) -> None:
        super().__init__()
        self.states = states_3d
        self.prices = prices if prices is not None else np.ones(len(states_3d))
        self.max_steps = len(self.states) - 1
        self.initial_balance = initial_balance
        
        self.reward_fn = RewardFunction()
        
        # 0 = SELL, 1 = HOLD, 2 = BUY
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(30, 10), dtype=np.float32
        )
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.position = 0  # 0 = No Position, 1 = Holding
        self.avg_price = 0.0
        
        self.reward_fn.reset()
        return self._get_state(), self._get_info()

    def _get_state(self) -> np.ndarray:
        state_window = np.copy(self.states[self.current_step])
        
        unrealised_pnl = 0.0
        if self.position == 1 and self.shares > 0:
            current_price = self.prices[self.current_step]
            if self.avg_price > 0:
                unrealised_pnl = (current_price - self.avg_price) / self.avg_price
            
        # Inject position and unrealised_pnl into last row (current step)
        state_window[-1, 8] = float(self.position)
        state_window[-1, 9] = float(unrealised_pnl)
        
        return state_window.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        current_price = self.prices[self.current_step]
        equity = self.cash + (self.shares * current_price if self.position == 1 else 0)
        return {"step": self.current_step, "equity": equity, "position": self.position}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.prices[self.current_step]
        prev_equity = self.cash + (self.shares * current_price if self.position == 1 else 0)
        
        trade_value = 0.0
        penalty = 0.0
        
        if action == 2:  # BUY
            if self.position == 0:
                trade_value = self.cash * 0.99
                self.shares = trade_value / current_price
                self.cash -= trade_value
                self.avg_price = current_price
                self.position = 1
            else:
                penalty = -5.0
        elif action == 0:  # SELL
            if self.position == 1:
                trade_value = self.shares * current_price
                self.cash += trade_value
                self.shares = 0.0
                self.position = 0
            else:
                penalty = -5.0
                
        # HOLD (1) passes implicitly
                
        current_equity = self.cash + (self.shares * current_price if self.position == 1 else 0)
        
        reward = self.reward_fn.calculate(prev_equity, current_equity, trade_value)
        reward += penalty
        
        self.current_step += 1
        terminated = current_equity <= self.initial_balance * 0.1 or self.current_step >= self.max_steps
        
        return self._get_state(), float(reward), terminated, False, self._get_info()
'''

network_code = '''import torch
import torch.nn as nn

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN with Conv1D Backbone for temporal financial state processing.
    """
    def __init__(self, window_size: int = 30, features_count: int = 10, action_dim: int = 3) -> None:
        super().__init__()
        
        # Expects input (batch, channels=features, sequence_length=window)
        self.conv_backbone = nn.Sequential(
            nn.Conv1d(in_channels=features_count, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        flatten_dim = 64 * window_size
        
        self.value_stream = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass aggregation.
        Args:
            x: State tensor (batch_size, window_size, features_count)
        Returns:
            Q-values Tensor (batch_size, action_dim)
        """
        # Permute to (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        conv_out = self.conv_backbone(x)
        features = torch.flatten(conv_out, start_dim=1)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q = V(s) + A(s,a) - mean(A(s,a))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_vals
'''

files = {
    "data/preprocessor.py": preprocessor_code,
    "env/reward.py": reward_code,
    "env/trading_env.py": env_code,
    "model/network.py": network_code,
}

for path, code in files.items():
    with open(os.path.join(base, path), "w", encoding="utf-8") as f:
        f.write(code)

print("Updated all target files successfully.")
