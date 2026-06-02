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

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        suffix = self.data_cfg.get("cache_format", "parquet").lower()
        return self.cache_dir / f"{ticker}_{start}_{end}.{suffix}"

    def _fallback_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.csv"

    def _read_cached(self, path: Path) -> pd.DataFrame | None:
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path, index_col="Date", parse_dates=True)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to read cache at %s: %s", path, exc)
            return None

    def _write_cached(self, path: Path, frame: pd.DataFrame) -> None:
        try:
            if path.suffix == ".parquet":
                frame.to_parquet(path, compression="snappy")
            else:
                frame.to_csv(path)
            self.logger.info("Cached %s rows at %s", len(frame), path)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to write cache at %s: %s", path, exc)

    def download_ticker(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame | None:
        cache_path = self._cache_path(ticker, start, end)
        if cache_path.exists():
            self.logger.info("Loading %s from parquet cache: %s", ticker, cache_path)
            df = self._read_cached(cache_path)
            if df is not None:
                return df

        try:
            calls_per_second = self.rate_limits.get("calls_per_second", 2)
            time.sleep(1.0 / max(calls_per_second, 1))
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            if frame.empty:
                self.logger.warning("No rows from Yahoo for %s", ticker)
                raise ValueError(f"Empty result for {ticker}")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.droplevel(1)
            self._write_cached(cache_path, frame)
            return frame
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Yahoo fetch failed for %s: %s, trying CSV fallback", ticker, exc)
            fallback = self._fallback_path(ticker)
            if fallback.exists():
                self.logger.info("Loading %s from CSV fallback: %s", ticker, fallback)
                df = self._read_cached(fallback)
                if df is not None:
                    return df
            self.logger.error("No data available for %s (cache, online, or fallback)", ticker)
            return None
