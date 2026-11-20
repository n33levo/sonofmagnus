import numpy as np

from eval.calibrate_lichess_history import fit, predict, performance_index, game_features


def test_calibration_uses_training_scaling_and_improves_a_known_signal():
    x = np.arange(60, dtype=float)[:, None] / 60
    y = (x[:, 0] < .5).astype(float)
    model = fit(x[:45], y[:45], np.zeros(45), alpha=.1)
    np.testing.assert_allclose(model['mean'], x[:45].mean(0))
    before = {key: value.copy() for key, value in model.items()}
    result = predict(model, x[45:] * 100, np.zeros(15))
    assert (result < .5).all()
    for key in before:np.testing.assert_array_equal(model[key], before[key])


def test_rating_performance_index_matches_balanced_results():
    assert abs(performance_index(np.array([2400., 2400.]), np.array([1., 0.])) - 2400) < 1e-6


def test_game_features_do_not_read_the_game_result():
    root = dict(scores={'played': dict(point_loss=.2, cp_loss=150)})
    np.testing.assert_allclose(game_features([root]), [.2, .2, 1., 0.])
