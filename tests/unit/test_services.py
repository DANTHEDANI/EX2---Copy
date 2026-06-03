from src.trading_sdk.data.preprocessor import FeatureEngineer
from src.trading_sdk.services.backtest import BacktestService
from src.trading_sdk.services.inference import InferenceService


def test_backtest_service(mock_config, dummy_market_data):
    service = BacktestService(mock_config)
    fe = FeatureEngineer()
    states = fe.engineer_features(dummy_market_data)
    prices = dummy_market_data["Close"].to_numpy()[-len(states) :]
    result = service.run_backtest(states, prices)
    assert isinstance(result, dict)


def test_inference_service(mock_config):
    service = InferenceService(mock_config)
    result = service.predict(current_state=None)
    assert result == 1
