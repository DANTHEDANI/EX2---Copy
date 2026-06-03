from src.trading_sdk.data.preprocessor import FeatureEngineer
from src.trading_sdk.services.training import TrainingService


def test_training_service(mock_config, dummy_market_data):
    # Mock environment and model parameters for quick completion
    service = TrainingService(mock_config)

    fe = FeatureEngineer()
    states = fe.engineer_features(dummy_market_data)
    prices = dummy_market_data["Close"].to_numpy()[-len(states) :]

    # We want to run a quick train iteration without it taking forever.
    # We can fake the epochs count inside the service
    service.hyper["batch_size"] = 2
    service.hyper["episodes"] = 1
    service.hyper["warmup_steps"] = 2
    service.train(states, prices)

    assert len(service.memory) > 0
    assert service.device is not None
