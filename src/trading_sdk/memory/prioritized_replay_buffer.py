from collections import deque

import numpy as np


class PrioritizedReplayBuffer:
    """Proportional prioritized replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.eps = 1e-6

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = (
            np.expand_dims(state, 0),
            action,
            reward,
            np.expand_dims(next_state, 0),
            done,
        )
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(transition)
        self.priorities.append(max_priority)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[tuple[np.ndarray, tuple, tuple, np.ndarray, tuple], np.ndarray, np.ndarray]:
        probs = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        transitions = (
            np.concatenate(states),
            actions,
            rewards,
            np.concatenate(next_states),
            dones,
        )
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors, strict=False):
            self.priorities[idx] = float(abs(err) + self.eps)

    def __len__(self) -> int:
        return len(self.buffer)
