import numpy as np
import pandas as pd
import pytest
from shot_intelligence import aggregate_nba_shots, validate_profiles, validate_defense, player_zone_model, ZONES
from data_sources import demo_dataset
from model import run, Config


def test_nba_zones_and_aggregation():
    raw = pd.DataFrame([
        {'PLAYER_NAME':'BOS Sample 1', 'SHOT_ZONE_BASIC':'Restricted Area','SHOT_ZONE_AREA':'Center(C)','SHOT_TYPE':'2PT Field Goal','SHOT_MADE_FLAG':1},
        {'PLAYER_NAME':'BOS Sample 1', 'SHOT_ZONE_BASIC':'Left Corner 3','SHOT_ZONE_AREA':'Left Side(L)','SHOT_TYPE':'3PT Field Goal','SHOT_MADE_FLAG':0},
    ])
    data = aggregate_nba_shots(raw, 'BOS')
    assert set(data['ZONE']) == {'Rim', 'Corner 3'}
    assert data['FGA'].sum() == 2


def test_shrinkage_prevents_perfect_sample():
    p = pd.DataFrame([{'TEAM_ABBREVIATION':'BOS', 'PLAYER_NAME':'BOS Sample 1', 'ZONE':'Rim', 'FGA':1, 'FGM':1}])
    z = player_zone_model('BOS Sample 1', 'BOS', validate_profiles(p))
    assert 0.65 < z['make_prob'][0] < .7
    assert abs(z['shares'].sum()-1) < 1e-10


def test_invalid_data_rejected():
    with pytest.raises(ValueError):
        validate_profiles(pd.DataFrame([{'TEAM_ABBREVIATION':'BOS','PLAYER_NAME':'A','ZONE':'Rim','FGA':2,'FGM':3}]))


def test_defensive_adjustments_are_bounded():
    defense = validate_defense(pd.DataFrame([{'TEAM_ABBREVIATION':'NYK', 'ZONE':'Rim', 'FGA_ALLOWED':200, 'FGM_ALLOWED':190}]))
    model = player_zone_model('BOS Sample 1', 'BOS', None, defense, 'NYK', 1.5)
    assert 0 < model['defense_adjustment'][0] <= .06


def test_shot_mode_scores_reconcile_and_seed():
    teams, players = demo_dataset()
    data = pd.DataFrame([{'TEAM_ABBREVIATION':'BOS','PLAYER_NAME':'BOS Sample 1','ZONE':'Above-break 3','FGA':300,'FGM':120}])
    cfg = Config(iterations=1100, seed=91, shot_mode=True, shot_profiles=validate_profiles(data))
    first = run(teams, players, 'BOS', 'NYK', cfg)
    second = run(teams, players, 'BOS', 'NYK', cfg)
    assert np.array_equal(first['home_score'], second['home_score'])
    for team, score in [('BOS',first['home_score']),('NYK',first['away_score'])]:
        arr = first['team_arrays'][team]
        assert np.array_equal(arr['PTS'].sum(axis=1), score)
        assert np.all(arr['PTS'] >= arr['3PM'] * 3)
        assert np.all(arr['MIN'].sum(axis=1) >= 239.9)
