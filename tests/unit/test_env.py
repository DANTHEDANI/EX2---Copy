from src.trading_sdk.env.trading_env import TradingEnv
from src.trading_sdk.data.preprocessor import FeatureEngineer

def test_environment_resets(dummy_market_data):
    """Assert initialization passes basic sanity thresholds returning states dynamically."""
    
    fe = FeatureEngineer()
    states = fe.engineer_features(dummy_market_data)
    prices = dummy_market_data["Close"].to_numpy()[-len(states):]
    env = TradingEnv(states, prices)
    state, info = env.reset()
    
    assert state is not None
    assert state.shape == (30, 10)  # 30-day window, 10 features
    
def test_environment_step(dummy_market_data):
    fe = FeatureEngineer()
    states = fe.engineer_features(dummy_market_data)
    prices = dummy_market_data["Close"].to_numpy()[-len(states):]
    env = TradingEnv(states, prices)
    
    new_state, reward, terminated, truncated, info = env.step(1)
    assert not terminated
    assert reward >= -10.0
