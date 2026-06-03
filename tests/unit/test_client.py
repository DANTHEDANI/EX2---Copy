from unittest.mock import patch

import pandas as pd

from src.trading_sdk.data.client import YFinanceDataClient


@patch("src.trading_sdk.data.client.yf.download")
def test_data_client_download(mock_download, mock_config, dummy_market_data, tmp_path):
    mock_download.return_value = dummy_market_data

    # Override cache dir to tmp_path to avoid real IO caching issues
    client = YFinanceDataClient(mock_config)
    client.cache_dir = tmp_path

    df = client.download_ticker("AAPL", "2020-01-01", "2020-01-10", "1d")

    assert df is not None
    assert not df.empty

    # Secondary call should load from cache
    df_cache = client.download_ticker("AAPL", "2020-01-01", "2020-01-10", "1d")
    assert df_cache is not None


@patch("src.trading_sdk.data.client.yf.download")
def test_data_client_download_empty(mock_download, mock_config, tmp_path):
    mock_download.return_value = pd.DataFrame()
    client = YFinanceDataClient(mock_config)
    client.cache_dir = tmp_path

    df = client.download_ticker("AAPL", "2030-01-01", "2030-01-10", "1d")
    assert df is None


@patch("src.trading_sdk.data.client.yf.download")
def test_data_client_download_exception(mock_download, mock_config, tmp_path):
    mock_download.side_effect = Exception("API Error")
    client = YFinanceDataClient(mock_config)
    client.cache_dir = tmp_path

    df = client.download_ticker("AAPL", "2030-01-01", "2030-01-10", "1d")
    assert df is None
