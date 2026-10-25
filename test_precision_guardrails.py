import numpy as np
import pandas as pd
from model import _minute_prior, _input_warnings, Config


def test_recent_minutes_are_shrunk_not_blindly_replaced():
    d = pd.DataFrame({'MIN': [30.], 'MIN_L10': [38.], 'GAMES_PLAYED': [2]})
    assert 30 < _minute_prior(d)[0] < 33


def test_recent_minutes_with_enough_history_are_weighted():
    d = pd.DataFrame({'MIN': [30.], 'MIN_L10': [38.], 'GAMES_PLAYED': [20]})
    assert abs(_minute_prior(d)[0] - 33.2) < 1e-6


def test_invalid_recent_minutes_ignored():
    d = pd.DataFrame({'MIN': [30.], 'MIN_L5': [float('nan')], 'GAMES_PLAYED': [20]})
    assert _minute_prior(d)[0] == 30


def test_missing_starters_and_recent_history_are_flagged():
    d = pd.DataFrame({'PLAYER_NAME': ['A'] * 5, 'MIN': [30.] * 5,
                      'FGA': [10.] * 5, 'PTS': [12.] * 5, 'REB': [4.] * 5, 'AST': [2.] * 5})
    warnings = _input_warnings({'AAA': d}, Config())
    assert any('starters not confirmed' in x for x in warnings)
    assert any('recent player minutes unavailable' in x for x in warnings)
