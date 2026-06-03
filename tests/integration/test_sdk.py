from src.trading_sdk.sdk import TradingSDK


def test_sdk_instantiation():
    """Confirms primary integration endpoint acts correctly mapping logic components."""
    sdk = TradingSDK()
    assert sdk.config_manager is not None
    assert sdk.data_client is not None
    assert sdk.trainer is not None
