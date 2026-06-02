import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer utilizing standard deque circular queue properties.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Stores standard transition experience."""
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Samples minibatches of transitions for learning randomly."""
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*batch, strict=False)

        # Vertically stack batches for contiguous memory placement
        return (
            np.concatenate(state),
            action,
            reward,
            np.concatenate(next_state),
            done
        )

    def __len__(self) -> int:
        return len(self.buffer)
