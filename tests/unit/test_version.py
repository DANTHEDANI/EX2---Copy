import pytest
from src.trading_sdk.shared.version import VERSION

def test_version_format():
    assert isinstance(VERSION, str)
    assert float(VERSION) > 0
