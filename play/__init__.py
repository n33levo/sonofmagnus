__all__ = ['ChessRunner']


def __getattr__(name):
    if name == 'ChessRunner':
        from play.runner import ChessRunner
        return ChessRunner
    raise AttributeError(name)
