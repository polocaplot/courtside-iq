import numpy as np
import pandas as pd
import pytest
from lineup_engine import choose_lineups, validate_lineups
from defender_intelligence import validate_defenders, defender_zone_adjustments
from calibration import evaluate_calibration, validate_results


def test_five_on_court_and_injuries():
    m = np.tile(np.array([36, 34, 30, 28, 26, 22, 18, 6, 0, 0.]), (100, 1))
    selected = choose_lineups(np.random.default_rng(4), m, [f'P{i}' for i in range(10)], 'BOS')
    assert (selected.sum(axis=1) == 5).all()
    assert not selected[:, 8:].any()


def test_observed_lineups_only_when_eligible():
    rows = pd.DataFrame([['BOS', 'P0', 'P1', 'P2', 'P3', 'P4', 100]], columns=[
        'TEAM_ABBREVIATION', 'PLAYER_1', 'PLAYER_2', 'PLAYER_3', 'PLAYER_4', 'PLAYER_5', 'POSSESSIONS'])
    prior = validate_lineups(rows)
    m = np.tile(np.array([30, 30, 30, 30, 30, 30.]), (15, 1))
    selected = choose_lineups(np.random.default_rng(1), m, [f'P{i}' for i in range(6)], 'BOS', prior)
    assert (selected.sum(axis=1) == 5).all()
    # Historical units are evidence, not mandatory starters: new combinations
    # must remain possible even when a prior unit exists.
    assert selected[:, 5].any()
    m[:, 4] = 0
    selected = choose_lineups(np.random.default_rng(1), m, [f'P{i}' for i in range(6)], 'BOS', prior)
    assert not selected[:, 4].any()
    assert selected[:, 5].all()


def test_defender_shrinkage_and_missing():
    rows = pd.DataFrame([['BOS', 'P0', 'Rim', 10, 2]], columns=[
        'TEAM_ABBREVIATION', 'DEFENDER_NAME', 'ZONE', 'FGA_CONTESTED', 'FGM_ALLOWED'])
    d = validate_defenders(rows)
    roster = pd.DataFrame({'PLAYER_NAME': ['P0', 'P1']})
    a = defender_zone_adjustments(roster, 'BOS', d)
    assert a.shape == (2, 5)
    assert -.04 <= a[0, 0] < 0
    assert np.all(a[1] == 0)


def test_calibration_no_future_leakage():
    d = pd.DataFrame({'GAME_DATE': pd.date_range('2025-01-01', periods=120),
                      'HOME_WIN_PROB': np.linspace(.2, .8, 120),
                      'HOME_WON': ([0, 1] * 60),
                      'FORECAST_SAVED_AT': pd.date_range('2024-12-31', periods=120)})
    a = evaluate_calibration(d)
    assert a['train_games'] + a['holdout_games'] == 120
    assert a['holdout_games'] >= 15
    d.loc[0, 'FORECAST_SAVED_AT'] = d.loc[0, 'GAME_DATE']
    with pytest.raises(ValueError, match='after tipoff'):
        validate_results(d)
