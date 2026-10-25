import numpy as np
import pandas as pd
from ensemble import blended_inputs, role_usage_adjustments, prediction_deltas
from data_sources import demo_dataset
from fantasy import FantasyScoring
from model import Config, run


def test_small_sample_shrinks_rate_and_minutes_toward_prior():
    _, roster = demo_dataset()
    current = roster.head(1).copy()
    current['GP'] = 1
    current['PTS'] = 40.
    current['MIN'] = 40.
    prior = current.copy()
    prior['PTS'] = 20.
    prior['MIN'] = 30.
    out, audit = blended_inputs(current, prior)
    assert 30 < out.iloc[0]['MIN'] < 35
    assert 19 < out.iloc[0]['PTS'] < 26
    assert audit.iloc[0]['Current weight'] < .1


def test_no_prior_preserves_baseline():
    _, players = demo_dataset()
    out, audit = blended_inputs(players)
    for col in ('MIN', 'PTS', 'REB', 'AST', 'FGA'):
        np.testing.assert_allclose(out[col], players[col])
    assert not audit['Prior matched'].any()


def test_role_does_not_override_manual_and_ignores_healthy_team():
    _, players = demo_dataset()
    subset = players[players['TEAM_ABBREVIATION'].isin(['BOS', 'NYK'])]
    overrides, audit = role_usage_adjustments(subset, {'BOS Sample 1': 'Out'}, {'BOS Sample 2': 1.42})
    assert overrides['BOS Sample 2'] == 1.42
    assert 1 < overrides['BOS Sample 3'] <= 1.2
    assert not any(k.startswith('NYK') for k in overrides)
    assert len(audit) > 0


def test_same_engine_baseline_remains_reproducible():
    teams, players = demo_dataset()
    cfg = Config(iterations=100, seed=53)
    original = run(teams, players, 'BOS', 'NYK', cfg)
    original_again = run(teams, players, 'BOS', 'NYK', cfg)
    np.testing.assert_array_equal(original['home_score'], original_again['home_score'])
    smart_players, _ = blended_inputs(players)
    smart = run(teams, smart_players, 'BOS', 'NYK', cfg)
    deltas = prediction_deltas(original, smart, FantasyScoring())
    assert len(deltas) == len(original['box'])
    assert (deltas['Fantasy change'].abs() < .00001).all()
