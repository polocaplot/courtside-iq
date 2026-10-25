import numpy as np
import pandas as pd
from model import _minutes, _player_list, Config


def test_rotation_can_include_eleventh_meaningful_player():
    prior = [33, 31, 29, 27, 25, 23, 21, 19, 17, 13, 11, 4, 2]
    roster = pd.DataFrame({'PLAYER_NAME': [f'P{i}' for i in range(len(prior))],
                           'TEAM_ABBREVIATION': ['MIA'] * len(prior), 'MIN': prior})
    minutes, _ = _minutes(np.random.default_rng(17), 1500, roster,
                          Config(minutes_sd=0), np.zeros(1500))
    assert np.allclose(minutes.sum(axis=1), 240, atol=.001)
    assert (minutes[:, 10] > 0).mean() > .5
    assert 9.5 < (minutes > 0).sum(axis=1).mean() <= 11.2


def test_roster_uses_effective_recent_minutes_for_ranking():
    prior = [20] * 14 + [4]
    last5 = [20] * 14 + [35]
    roster = pd.DataFrame({'PLAYER_NAME': [f'P{i}' for i in range(15)],
                           'TEAM_ABBREVIATION': ['MIA'] * 15,
                           'MIN': prior, 'MIN_L5': last5, 'MIN_L10': last5,
                           'GAMES_PLAYED': [20] * 15})
    chosen = _player_list(roster, 'MIA')
    assert len(chosen) == 15
    assert 'P14' in set(chosen.PLAYER_NAME)
