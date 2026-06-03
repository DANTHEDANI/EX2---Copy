import torch

from src.trading_sdk.model.network import DuelingDQNNetwork


def test_model_forward():
    model = DuelingDQNNetwork(action_dim=3)
    # Batch size 4, 30 days, 10 features
    dummy_input = torch.randn(4, 30, 10)
    output = model(dummy_input)
    assert output.shape == (4, 3)
