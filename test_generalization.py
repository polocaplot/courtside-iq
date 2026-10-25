import numpy as np
import pandas as pd
from generalization import lineup_evidence_weight, blend_rates, sample_reliability
from lineup_engine import choose_lineups
from defense_engine import defensive_rates


def test_lineup_prior_cannot_take_over_even_large_sample():
    counts = np.array([100000.])
    valid = np.ones((30, 1), dtype=bool)
    assert np.all(lineup_evidence_weight(counts, valid) < .66)
    assert np.all(lineup_evidence_weight(counts, valid) > 0)


def test_injured_lineups_are_ignored():
    rng = np.random.default_rng(123)
    names = list('ABCDEFG')
    minutes = np.tile(np.array([32, 31, 30, 28, 25, 20, 0.]), (500, 1))
    prior = pd.DataFrame([['BOS', 'A', 'B', 'C', 'D', 'G', 50000]],
        columns=['TEAM_ABBREVIATION', 'PLAYER_1', 'PLAYER_2', 'PLAYER_3', 'PLAYER_4', 'PLAYER_5', 'POSSESSIONS'])
    selected = choose_lineups(rng, minutes, names, 'BOS', prior)
    assert (selected.sum(axis=1) == 5).all()
    assert not selected[:, 6].any()


def test_sparse_observed_lineup_does_not_prevent_new_units():
    rng = np.random.default_rng(7)
    names = list('ABCDEFG')
    minutes = np.tile(np.array([35, 35, 34, 33, 31, 29, 25.]), (4000, 1))
    prior = pd.DataFrame([['BOS', 'A', 'B', 'C', 'D', 'E', 100000]],
        columns=['TEAM_ABBREVIATION', 'PLAYER_1', 'PLAYER_2', 'PLAYER_3', 'PLAYER_4', 'PLAYER_5', 'POSSESSIONS'])
    selected = choose_lineups(rng, minutes, names, 'BOS', prior)
    assert selected[:, 5:].any()
    assert selected[:, 5:].sum() > 100


def test_defense_one_game_shrinks_more_than_many_games():
    row = dict(MIN=32., STL=2.4, BLK=2., REB=9.)
    small = pd.DataFrame([{**row, 'GP': 1}])
    large = pd.DataFrame([{**row, 'GP': 70}])
    minute = np.full((1, 1), 32.)
    x = defensive_rates(small, minute)
    y = defensive_rates(large, minute)
    assert abs(x['steal'][0]) < abs(y['steal'][0])
    assert abs(x['block'][0]) < abs(y['block'][0])


def test_new_season_reliability_and_missing_prior():
    assert sample_reliability([1])[0] < .1
    assert sample_reliability([60])[0] > .7
    assert np.isclose(blend_rates([25], [10], [1])[0], 10 + 15 / 19)
    assert blend_rates([25], [np.nan], [1])[0] == 25
