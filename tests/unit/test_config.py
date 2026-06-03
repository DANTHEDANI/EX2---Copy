from src.trading_sdk.shared.config import ConfigManager
from src.trading_sdk.shared.constants import STATE_DIM


def test_config_loads_properly(mock_config: ConfigManager):
    """Verifies that dependency files correctly map content dynamically."""
    setup = mock_config.setup
    assert "hyperparameters" in setup
    assert setup["hyperparameters"]["learning_rate"] > 0
    assert STATE_DIM == 10
