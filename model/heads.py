"""
Policy and value heads for chess neural network.

Policy head: Outputs 8×8×73 logits for move selection
Value head: Outputs scalar in [-1, 1] for position evaluation
"""

import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """
    Policy head that outputs 8×8×73 move logits.

    Architecture:
        hidden -> Conv(3×3) -> GroupNorm -> SiLU -> Conv(1×1) -> 8×8×73 logits
    """

    def __init__(self, hidden_channels: int, num_groups: int = 8):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, hidden_channels)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, 73, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Hidden features of shape (B, hidden_channels, 8, 8)

        Returns:
            Policy logits of shape (B, 8, 8, 73)
        """
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.activation(x)
        x = self.conv2(x)  # (B, 73, 8, 8)

        # Permute to (B, 8, 8, 73) for easier indexing
        x = x.permute(0, 2, 3, 1)

        return x


class ValueHead(nn.Module):
    """
    Value head that outputs a scalar position evaluation in [-1, 1].

    Architecture:
        hidden -> Conv(1×1) -> GroupNorm -> SiLU -> GlobalAvgPool -> FC -> SiLU -> FC -> tanh
    """

    def __init__(self, hidden_channels: int, num_groups: int = 8):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, 32), 32)
        self.activation = nn.SiLU()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Hidden features of shape (B, hidden_channels, 8, 8)

        Returns:
            Value estimates of shape (B, 1) in range [-1, 1]
        """
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
    """
    Combined policy and value heads.
    """

    def __init__(self, hidden_channels: int, num_groups: int = 8):
        super().__init__()
        self.policy_head = PolicyHead(hidden_channels, num_groups)
        self.value_head = ValueHead(hidden_channels, num_groups)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Hidden features of shape (B, hidden_channels, 8, 8)

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: shape (B, 8, 8, 73)
                - value: shape (B, 1)
        """
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


if __name__ == "__main__":
    # Test heads
    print("Testing policy and value heads...")

    batch_size = 4
    hidden_channels = 128

    # Test policy head
    policy_head = PolicyHead(hidden_channels)
    x = torch.randn(batch_size, hidden_channels, 8, 8)
    policy_logits = policy_head(x)
    print(f"\nPolicy head:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {policy_logits.shape}")
    print(f"  Expected: ({batch_size}, 8, 8, 73)")
    assert policy_logits.shape == (batch_size, 8, 8, 73)

    # Test value head
    value_head = ValueHead(hidden_channels)
    value = value_head(x)
    print(f"\nValue head:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {value.shape}")
    print(f"  Expected: ({batch_size}, 1)")
    print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    assert value.shape == (batch_size, 1)
    assert value.min() >= -1.0 and value.max() <= 1.0

    # Test combined heads
    heads = ChessHeads(hidden_channels)
    policy, value = heads(x)
    print(f"\nCombined heads:")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")

    print("\n✓ All head tests passed!")
