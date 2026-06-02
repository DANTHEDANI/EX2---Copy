import numpy as np
import pandas as pd


class FeatureEngineer:
    """Generate leakage-safe features and chronological train/val/test splits."""

    def __init__(
        self,
        window_size: int = 30,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> None:
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.features = [
            "log_return", "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_pct", "vwap_dist", "volume_norm", "position", "unrealised_pnl",
        ]

    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        frame = self._compute_feature_frame(df)
        matrix = frame[self.features].to_numpy(dtype=np.float32)
        count = max(0, len(matrix) - self.window_size + 1)
        return np.array([matrix[i:i + self.window_size] for i in range(count)], dtype=np.float32)

    def split_data(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(states)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        return states[:train_end], states[train_end:val_end], states[val_end:]

    def _compute_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close, volume = df["Close"], df["Volume"]
        df["log_return"] = np.log(close / close.shift(1))
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        ema_12 = close.ewm(12, adjust=False).mean()
        ema_26 = close.ewm(26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        sma, std = close.rolling(20).mean(), close.rolling(20).std()
        lower, upper = sma - (2 * std), sma + (2 * std)
        df["bb_pct"] = (close - lower) / (upper - lower + 1e-8)
        typical = (df["High"] + df["Low"] + close) / 3
        vwap = (typical * volume).rolling(14).sum() / (volume.rolling(14).sum() + 1e-8)
        df["vwap_dist"] = (close - vwap) / (vwap + 1e-8)
        df["volume_norm"] = volume / (volume.rolling(14).mean() + 1e-8)
        df["position"], df["unrealised_pnl"] = 0.0, 0.0
        df = df.dropna().copy()
        for col in self.features[:-2]:
            lo = df[col].expanding().min()
            hi = df[col].expanding().max()
            df[col] = (df[col] - lo) / (hi - lo + 1e-8)
        return df
