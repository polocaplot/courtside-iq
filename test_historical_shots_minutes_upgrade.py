import numpy as np
import pandas as pd
from shot_intelligence import historical_seasons, fetch_player_shots, merge_historical_profiles
from model import _minute_prior, _minutes, Config


def test_historical_seasons_correct_across_years():
    assert historical_seasons('2026-27', 3) == ('2025-26', '2024-25', '2023-24')


def test_historical_shot_weight_and_team_remap():
    current = pd.DataFrame([dict(TEAM_ABBREVIATION='MIA',PLAYER_NAME='Player',ZONE='Rim',FGA=10,FGM=6)])
    prior = pd.DataFrame([dict(TEAM_ABBREVIATION='MIA',PLAYER_NAME='Player',ZONE='Rim',FGA=100,FGM=50)])
    combined = merge_historical_profiles(current, prior, weight=.4)
    assert combined.iloc[0]['FGA'] == 50
    assert combined.iloc[0]['FGM'] == 26


def test_shot_fetch_uses_player_id_and_no_historical_team(monkeypatch):
    import sys, types
    captured = {}
    class FakeEndpoint:
        def __init__(self, **kw): captured.update(kw)
        def get_data_frames(self):
            return [pd.DataFrame([dict(PLAYER_NAME='Old Name', SHOT_ZONE_BASIC='Restricted Area',
                                       SHOT_ZONE_AREA='Center(C)', SHOT_TYPE='2PT Field Goal',
                                       SHOT_MADE_FLAG=1)])]
    endpoint = types.ModuleType('nba_api.stats.endpoints')
    endpoint.shotchartdetail = types.SimpleNamespace(ShotChartDetail=FakeEndpoint)
    stats = types.ModuleType('nba_api.stats'); stats.endpoints = endpoint
    api = types.ModuleType('nba_api'); api.stats = stats
    monkeypatch.setitem(sys.modules,'nba_api',api)
    monkeypatch.setitem(sys.modules,'nba_api.stats',stats)
    monkeypatch.setitem(sys.modules,'nba_api.stats.endpoints',endpoint)
    frame = fetch_player_shots(123, '2025-26', 'MIA', 'New Name')
    assert captured['player_id'] == 123 and captured['team_id'] == 0
    assert frame.iloc[0]['TEAM_ABBREVIATION'] == 'MIA'
    assert frame.iloc[0]['PLAYER_NAME'] == 'New Name'


def test_recent_minutes_do_not_double_count_and_are_bounded():
    roster = pd.DataFrame({'MIN':[32,25], 'MIN_L5':[38, np.nan],
                           'MIN_L10':[34, 30], 'GAMES_PLAYED':[3,20]})
    result = _minute_prior(roster)
    assert 32 < result[0] < 37
    assert abs(result[1]-27) < 1e-8


def test_minute_conservation_with_recent_role_shifts():
    roster = pd.DataFrame({'PLAYER_NAME':[f'P{i}' for i in range(11)],
                           'MIN':[34,33,32,31,30,27,25,22,20,15,12],
                           'MIN_L5':[37,35,34,32,30,25,20,15,10,7,4],
                           'MIN_L10':[36,34,33,31,29,26,22,19,14,10,7],
                           'GAMES_PLAYED':[8]*11})
    mins,_ = _minutes(np.random.default_rng(9), 200, roster, Config(), np.zeros(200))
    assert np.allclose(mins.sum(axis=1), 240, atol=.001)
    assert mins[:,0].mean() > mins[:,8].mean()
