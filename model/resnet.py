"""
ResNet backbone for chess neural network.

Architecture:
- Initial conv layer to project input to hidden dimension
- N residual blocks (10-14 for main, 6-8 for mini)
- GroupNorm + SiLU activations
- 128 channels (main) or 64 channels (mini)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU.

    Architecture:
        x -> Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> (+x) -> SiLU
    """

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + residual
        out = self.activation(out)

        return out


class ChessResNet(nn.Module):
    """
    ResNet trunk for chess position encoding.

    Args:
        input_channels: Number of input planes (18 for base encoding)
        hidden_channels: Number of channels in residual blocks
        num_blocks: Number of residual blocks
        num_groups: Number of groups for GroupNorm
    """

    def __init__(
        self,
        input_channels: int = 18,
        hidden_channels: int = 128,
        num_blocks: int = 12,
        num_groups: int = 8,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks

        # Initial projection
        self.input_conv = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.input_gn = nn.GroupNorm(num_groups, hidden_channels)
        self.input_activation = nn.SiLU()

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, num_groups) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, 8, 8) where C = input_channels

        Returns:
            Hidden representation of shape (B, hidden_channels, 8, 8)
        """
        # Initial projection
        x = self.input_conv(x)
        x = self.input_gn(x)
        x = self.input_activation(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        return x

    def get_num_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_resnet_main(input_channels: int = 18) -> ChessResNet:
    """
    Create main ResNet configuration (~6-8M params).

    Args:
        input_channels: Number of input planes

    Returns:
        ChessResNet model
    """
    return ChessResNet(
        input_channels=input_channels,
        hidden_channels=128,
        num_blocks=12,
        num_groups=8,
    )


def create_resnet_mini(input_channels: int = 18) -> ChessResNet:
    """
    Create mini ResNet configuration (~2-4M params for ≤10 MB target).

    Args:
        input_channels: Number of input planes

    Returns:
        ChessResNet model
    """
    return ChessResNet(
        input_channels=input_channels,
        hidden_channels=64,
        num_blocks=8,
        num_groups=8,
    )


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing ResNet backbone...")

    # Main config
    model_main = create_resnet_main()
    print(f"\nMain config:")
    print(f"  Hidden channels: {model_main.hidden_channels}")
    print(f"  Num blocks: {model_main.num_blocks}")
    print(f"  Num parameters: {model_main.get_num_params():,}")

    # Mini config
    model_mini = create_resnet_mini()
    print(f"\nMini config:")
    print(f"  Hidden channels: {model_mini.hidden_channels}")
    print(f"  Num blocks: {model_mini.num_blocks}")
    print(f"  Num parameters: {model_mini.get_num_params():,}")

    # Forward pass test
    batch_size = 4
    x = torch.randn(batch_size, 18, 8, 8)

    print(f"\nTesting forward pass (batch_size={batch_size})...")
    with torch.no_grad():
        out_main = model_main(x)
        out_mini = model_mini(x)

    print(f"Main output shape: {out_main.shape}")
    print(f"Mini output shape: {out_mini.shape}")
    print("\n✓ ResNet backbone tests passed!")
