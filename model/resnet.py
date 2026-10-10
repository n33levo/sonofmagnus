import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

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
        # Initial projection
        x = self.input_conv(x)
        x = self.input_gn(x)
        x = self.input_activation(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_resnet_main(input_channels: int = 18) -> ChessResNet:
    return ChessResNet(
        input_channels=input_channels,
        hidden_channels=160,
        num_blocks=12,
        num_groups=8,
    )


def create_resnet_mini(input_channels: int = 18) -> ChessResNet:
    return ChessResNet(
        input_channels=input_channels,
        hidden_channels=64,
        num_blocks=8,
        num_groups=8,
    )
