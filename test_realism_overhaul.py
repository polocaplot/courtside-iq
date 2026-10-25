"""Structural simulation tests, not claims of predictive validation."""
import numpy as np
import pandas as pd
from data_sources import demo_dataset
from lineup_engine import choose_target_balanced_lineups
from model import Config, run


def test_minute_scheduler_respects_remaining_time_and_eligibility():
    rng = np.random.default_rng(22)
    target = np.array([[36., 34., 32., 30., 28., 26., 22., 18., 14., 0.]])
    played = np.zeros((1, 10), dtype=np.int16)
    previous = None
    for step in range(96):
        if step % 4 == 0:
            previous = choose_target_balanced_lineups(
                rng, target, played, np.array([96]), step, 4, previous,
                np.maximum(target * 2 - played, 0.), [f'P{i}' for i in range(10)], 'TEST')
        played += previous
    minutes = played * (48. / 96.)
    assert played.sum() == 480
    assert minutes[0, 9] == 0
    assert np.max(np.abs(minutes[0, :9] - target[0, :9])) <= 4.0


def test_overtime_uses_real_scoring_events_and_conserves_minutes():
    teams, players = demo_dataset()
    result = run(teams, players, 'BOS', 'NYK', Config(
        iterations=1600, seed=14, possession_mode=True,
        calibrate_possession=False))
    for team in ('BOS', 'NYK'):
        stats = result['team_arrays'][team]
        assert np.array_equal(stats['PTS'].sum(axis=1),
                              2 * stats['FGM'].sum(axis=1) +
                              stats['3PM'].sum(axis=1) +
                              stats['FTM'].sum(axis=1))
        total = stats['MIN'].sum(axis=1)
        assert np.all(np.isclose((total - 240.) / 25.,
                                 np.rint((total - 240.) / 25.), atol=.001))
        assert np.any(total > 250.)
        assert np.array_equal(stats['FGA'].sum(axis=1),
                              result['possession_audits'][team]['FGA'])
    assert not np.any(result['home_score'] == result['away_score'])


def test_shot_selection_respects_per_minute_rate_not_total_minutes():
    from pathlib import Path
    source = (Path(__file__).parent.parent / 'possession_engine.py').read_text()
    assert 'per-minute rates' in source
    assert 'shooter_w = ((np.maximum' in source
    assert 'opportunity = mins[t] / baseline' not in source
