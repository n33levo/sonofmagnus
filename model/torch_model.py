import torch
import torch.nn as nn
from model.resnet import ChessResNet, create_resnet_main, create_resnet_mini
from model.heads import ChessHeads


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def create_model(config):
    defaults = {"main": (160, 12), "mini": (64, 8)}
    if isinstance(config, str):
        config = {"model_size": config}
    channels, blocks = defaults[config.get("model_size", "main")]
    return ChessNet(config.get("input_channels", 18), config.get("hidden_channels", channels),
                    config.get("num_blocks", blocks), config.get("num_groups", 8))


def load_model(path, config="main"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(checkpoint.get("config", config))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


class ChessNet(nn.Module):

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
        self.q_head = nn.Conv2d(hidden_channels, 73, 1)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        policy, value = self.heads(features)
        return policy, value

    def forward_with_q(self, x):
        features = self.backbone(x)
        policy, value = self.heads(features)
        return policy, value, self.q_head(features).permute(0, 2, 3, 1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_main(input_channels: int = 18) -> ChessNet:
    return ChessNet(
        input_channels=input_channels,
        hidden_channels=160,
        num_blocks=12,
        num_groups=8,
    )


def create_model_mini(input_channels: int = 18) -> ChessNet:
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
