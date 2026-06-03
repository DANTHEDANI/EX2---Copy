# Tests - Pytest Conftest Shared Setup


import numpy as np
import pandas as pd
import pytest

from src.trading_sdk.shared.config import ConfigManager


@pytest.fixture
def mock_config() -> ConfigManager:
    """Provides a functional ConfigManager without side effects."""
    return ConfigManager()


@pytest.fixture
def dummy_market_data() -> pd.DataFrame:
    """Returns static verifiable dataset for environment injection."""
    dates = pd.date_range("2020-01-01", periods=100)
    data = {
        "Open": np.linspace(100, 110, 100),
        "High": np.linspace(102, 112, 100),
        "Low": np.linspace(98, 108, 100),
        "Close": np.linspace(101, 111, 100),
        "Volume": np.random.randint(1000, 5000, size=100),
    }
    return pd.DataFrame(data, index=dates)
