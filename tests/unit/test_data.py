import numpy as np
import pytest
from src.trading_sdk.data.preprocessor import FeatureEngineer

def test_feature_engineering_dimensions(dummy_market_data):
    fe = FeatureEngineer()
    result = fe.engineer_features(dummy_market_data)
    
    # Needs exactly 10 output columns per time step window 30x10
    assert result.shape[1] == 30
    assert result.shape[2] == 10
    assert not np.isnan(result).any()

def test_chronological_split(dummy_market_data):
    fe = FeatureEngineer()
    result = fe.engineer_features(dummy_market_data)
    train, val, test = fe.split_data(result)
    assert len(train) + len(val) + len(test) == len(result)
    assert len(train) >= len(val) >= 0
