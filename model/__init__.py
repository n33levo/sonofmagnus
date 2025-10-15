"""
Complete chess neural network model (policy + value).
"""

import torch
import torch.nn as nn
from model.resnet import ChessResNet, create_resnet_main, create_resnet_mini
from model.heads import ChessHeads


class ChessNet(nn.Module):
    """
    Complete chess network with ResNet backbone and policy+value heads.
    """

    def __init__(
        self,
        input_channels: int = 18,
        hidden_channels: int = 128,
        num_blocks: int = 12,
        num_groups: int = 8,
    ):
        super().__init__()

        self.backbone = ChessResNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_groups=num_groups,
        )

        self.heads = ChessHeads(hidden_channels=hidden_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Board encoding of shape (B, input_channels, 8, 8)

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: shape (B, 8, 8, 73)
                - value: shape (B, 1) in range [-1, 1]
        """
        features = self.backbone(x)
        policy, value = self.heads(features)
        return policy, value

    def get_num_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_main(input_channels: int = 18) -> ChessNet:
    """
    Create main model configuration (~6-8M params).
    """
    return ChessNet(
        input_channels=input_channels,
        hidden_channels=128,
        num_blocks=12,
        num_groups=8,
    )


def create_model_mini(input_channels: int = 18) -> ChessNet:
    """
    Create mini model configuration (~2-4M params for ≤10 MB target).
    """
    return ChessNet(
        input_channels=input_channels,
        hidden_channels=64,
        num_blocks=8,
        num_groups=8,
    )


__all__ = [
    'ChessNet',
    'create_model_main',
    'create_model_mini',
    'ChessResNet',
    'ChessHeads',
]


if __name__ == "__main__":
    print("Testing complete ChessNet model...")

    # Main config
    model_main = create_model_main()
    print(f"\nMain model:")
    print(f"  Parameters: {model_main.get_num_params():,}")

    # Mini config
    model_mini = create_model_mini()
    print(f"\nMini model:")
    print(f"  Parameters: {model_mini.get_num_params():,}")

    # Forward pass
    batch_size = 2
    x = torch.randn(batch_size, 18, 8, 8)

    print(f"\nForward pass test (batch_size={batch_size})...")
    with torch.no_grad():
        policy, value = model_main(x)

    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

    print("\n✓ ChessNet tests passed!")
