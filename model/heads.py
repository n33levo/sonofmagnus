import torch
import torch.nn as nn


class PolicyHead(nn.Module):

    def __init__(self, hidden_channels: int, num_groups: int = 8):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, hidden_channels)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, 73, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.activation(x)
        x = self.conv2(x)  # (B, 73, 8, 8)

        # Permute to (B, 8, 8, 73) for easier indexing
        x = x.permute(0, 2, 3, 1)

        return x


class ValueHead(nn.Module):

    def __init__(self, hidden_channels: int, num_groups: int = 8):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, 32), 32)
        self.activation = nn.SiLU()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.activation(x)

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (B, 32)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        # Tanh to bound to [-1, 1]
        x = torch.tanh(x)

        return x


class ChessHeads(nn.Module):

    def __init__(self, hidden_channels: int, num_groups: int = 8):
        super().__init__()
        self.policy_head = PolicyHead(hidden_channels, num_groups)
        self.value_head = ValueHead(hidden_channels, num_groups)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
