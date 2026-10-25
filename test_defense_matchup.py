import numpy as np
import pytest
from defense_engine import lineup_contest_modifier


def test_missing_observations_neutral():
    court = np.ones((3, 5), dtype=bool)
    np.testing.assert_array_equal(lineup_contest_modifier(None, court, [0, 2, 4]), 0.)


def test_zone_sensitive_contest_weights():
    court = np.ones((2, 5), dtype=bool)
    mods = np.zeros((5, 5)); mods[0, 0] = -.04; mods[1, 4] = .04
    rates = {'block': np.array([1., 0., 0., 0., 0.]), 'steal': np.array([0., 1., 0., 0., 0.])}
    x = lineup_contest_modifier(mods, court, np.array([0, 4]), rates)
    assert x[0] < 0 < x[1]
    assert np.all(np.abs(x) <= .03)


def test_no_invented_effect_and_validation():
    court = np.ones((1, 5), dtype=bool)
    assert lineup_contest_modifier(np.zeros((5, 5)), court, [0])[0] == 0
    with pytest.raises(ValueError):
        lineup_contest_modifier(np.zeros((5, 5)), np.zeros((1, 5), dtype=bool), [0])
