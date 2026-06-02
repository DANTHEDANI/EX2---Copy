import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..shared.config import ConfigManager
from ..shared.constants import DATA_DIR


class YFinanceDataClient:
    """Download/caches Yahoo data using config-driven storage format."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.logger = logging.getLogger(__name__)
        self.rate_limits = config_manager.rate_limits.get("yfinance", {})
        self.data_cfg = config_manager.setup["data"]
        self.cache_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str, interval: str) -> Path:
        suffix = self.data_cfg.get("cache_format", "parquet").lower()
        return self.cache_dir / f"{ticker}_{start}_{end}_{interval}.{suffix}"

    @staticmethod
    def _read_cached(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def _write_cached(path: Path, frame: pd.DataFrame) -> None:
        if path.suffix == ".parquet":
            frame.to_parquet(path)
        else:
            frame.to_csv(path)

    def download_ticker(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame | None:
        cache_path = self._cache_path(ticker, start, end, interval)
        if cache_path.exists():
            self.logger.info("Loading %s from cache %s", ticker, cache_path)
            return self._read_cached(cache_path)

        try:
            calls_per_second = self.rate_limits.get("calls_per_second", 2)
            time.sleep(1.0 / max(calls_per_second, 1))
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
            )
            if frame.empty:
                self.logger.warning("No rows returned for %s", ticker)
                return None
            self._write_cached(cache_path, frame)
            self.logger.info("Cached %s rows at %s", len(frame), cache_path)
            return frame
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Yahoo fetch failed for %s: %s", ticker, exc)
            return None
