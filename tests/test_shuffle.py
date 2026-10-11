import torch

from train.train import ShuffledPositions


def test_large_epoch_shuffle_visits_each_position_once_and_reproduces_the_seed():
    sampler = ShuffledPositions(100003)
    torch.manual_seed(7)
    first, second = list(sampler), list(sampler)
    assert len(sampler) == 100003
    assert sorted(first) == list(range(len(sampler)))
    assert first != second
    torch.manual_seed(7)
    assert list(sampler) == first
