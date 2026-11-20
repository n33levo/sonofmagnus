"""Load the PyTorch model only when its API is requested."""
__all__ = ['ChessNet', 'create_model_main', 'create_model_mini', 'ChessResNet',
           'ChessHeads', 'create_resnet_main', 'create_resnet_mini']


def __getattr__(name):
    if name not in (*__all__, 'resolve_device', 'create_model', 'load_model'):
        raise AttributeError(name)
    from model import torch_model
    value = getattr(torch_model, name)
    globals()[name] = value
    return value
