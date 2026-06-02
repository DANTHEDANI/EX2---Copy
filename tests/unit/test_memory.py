from src.trading_sdk.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.trading_sdk.memory.replay_buffer import ReplayBuffer
import numpy as np

def test_replay_buffer_operations():
    buffer = ReplayBuffer(capacity=10)
    state = np.random.rand(12)
    next_state = np.random.rand(12)
    
    buffer.push(state, 1, 0.5, next_state, False)
    assert len(buffer) == 1
    
    state_b, action_b, reward_b, _, _ = buffer.sample(1)
    
    assert state_b.shape == (1, 12)
    assert action_b[0] == 1
    assert reward_b[0] == 0.5

def test_prioritized_replay_buffer_operations():
    buffer = PrioritizedReplayBuffer(capacity=10)
    state = np.random.rand(12)
    next_state = np.random.rand(12)
    for _ in range(5):
        buffer.push(state, 2, 1.0, next_state, False)
    (state_b, _, _, _, _), indices, weights = buffer.sample(
        3,
    )
    assert state_b.shape == (3, 12)
    assert len(indices) == 3
    assert weights.shape == (3,)
    buffer.update_priorities(indices, np.array([0.1, 0.2, 0.3]))
