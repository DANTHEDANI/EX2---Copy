import torch
import torch.nn as nn


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN with Conv1D Backbone for temporal financial state processing.
    """
    def __init__(
        self, window_size: int = 30, features_count: int = 10, action_dim: int = 3
    ) -> None:
        super().__init__()

        # Expects input (batch, channels=features, sequence_length=window)
        self.conv_backbone = nn.Sequential(
            nn.Conv1d(in_channels=features_count, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        flatten_dim = 64 * window_size

        self.value_stream = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass aggregation.
        Args:
            x: State tensor (batch_size, window_size, features_count)
        Returns:
            Q-values Tensor (batch_size, action_dim)
        """
        # Permute to (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)

        conv_out = self.conv_backbone(x)
        features = torch.flatten(conv_out, start_dim=1)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q = V(s) + A(s,a) - mean(A(s,a))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_vals
