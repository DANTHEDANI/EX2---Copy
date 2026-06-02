import logging

import numpy as np
import pandas as pd

from .data.client import YFinanceDataClient
from .data.preprocessor import FeatureEngineer
from .services.backtest import BacktestService
from .services.plots import plot_learning_curve
from .services.training import TrainingService
from .shared.config import ConfigManager


class TradingSDK:
    """Facade routing all app actions through service layer."""

    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.logger = logging.getLogger(__name__)
        data_cfg = self.config_manager.setup["data"]
        paths_cfg = self.config_manager.setup["paths"]
        self.data_client = YFinanceDataClient(self.config_manager)
        self.feature_engineer = FeatureEngineer(
            window_size=self.config_manager.setup["environment"]["window_size"],
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
        )
        self.results_dir = paths_cfg["results_dir"]
        self.trainer = TrainingService(self.config_manager)
        self.backtester = BacktestService(self.config_manager)

    def run_training_pipeline(self, ticker: str) -> None:
        config_data = self.config_manager.setup["data"]
        raw_df = self.data_client.download_ticker(
            ticker=ticker,
            start=config_data["start_date"],
            end=config_data["end_date"],
            interval=config_data["interval"],
        )
        if raw_df is None or raw_df.empty:
            self.logger.error("No input rows for %s", ticker)
            return
        states, prices = self._build_states_and_prices(raw_df)
        train_states, _, _ = self.feature_engineer.split_data(states)
        train_prices, _, _ = self.feature_engineer.split_data(prices)
        history = self.trainer.train(train_states, train_prices)
        plot_learning_curve(
            history["losses"],
            history["rewards"],
            self.results_dir,
        )
        self.logger.info("Training complete for %s", ticker)

    def evaluate_strategy(self, ticker: str) -> None:
        cfg = self.config_manager.setup["data"]
        raw_df = self.data_client.download_ticker(
            ticker,
            cfg["start_date"],
            cfg["end_date"],
            cfg["interval"],
        )
        if raw_df is None or raw_df.empty:
            self.logger.error("Cannot backtest without data for %s", ticker)
            return
        states, prices = self._build_states_and_prices(raw_df)
        _, _, test_states = self.feature_engineer.split_data(states)
        _, _, test_prices = self.feature_engineer.split_data(prices)
        metrics = self.backtester.run_backtest(test_states, test_prices)
        self.logger.info("Backtest metrics for %s: %s", ticker, metrics)

    def _build_states_and_prices(
        self,
        raw_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = self.feature_engineer.engineer_features(raw_df)
        close = raw_df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        clean_close = close.dropna().to_numpy(dtype=np.float32)
        offset = len(clean_close) - len(states)
        return states, clean_close[max(offset, 0):]
