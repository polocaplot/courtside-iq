import numpy as np
import pandas as pd
from generalization import sample_reliability
from shot_intelligence import player_zone_model, ZONES
from diagnostic_bundle import make_report


def test_two_games_have_low_historical_weight():
    assert sample_reliability([2], prior_games=18)[0] == .1


def test_real_two_game_chart_does_not_override_long_term_shooting_style():
    # Early-season fluke: interior player takes 8 threes in two games.
    chart = pd.DataFrame([
        dict(TEAM_ABBREVIATION='ATL', PLAYER_NAME='Interior Player', ZONE=z,
             FGA=n, FGM=n//2) for z,n in zip(ZONES, [7,4,3,3,5])
    ])
    base = dict(FGA=14., FG3A=1.0, FG3M=.25, GP=2)
    m = player_zone_model('Interior Player','ATL',chart,player_baseline=base)
    assert m['shares'][3:].sum() < .16
    assert np.isclose(m['shares'].sum(),1.)


def test_historical_evidence_exported_in_one_box():
    class Cfg:
        iterations=100
        seed=42
    inputs = pd.DataFrame([dict(TEAM_ABBREVIATION='ATL',PLAYER_NAME='Test',GP=2,
                                HISTORICAL_PRIOR_USED=True,HISTORICAL_PRIOR_SEASON='2025-26',
                                HISTORICAL_CURRENT_WEIGHT=.1,HISTORICAL_MINUTES_WEIGHT=2/7,
                                HISTORICAL_WARNING='')])
    result=dict(home='ATL',away='HOU',home_score=np.zeros(100),away_score=np.zeros(100),
                config=Cfg(),box=pd.DataFrame())
    import json
    d=json.loads(make_report(result,game='HOU @ ATL',season='2026-27',data_mode='Live NBA data',
                             source='test',player_inputs=inputs))
    assert d['data_quality']['historical_prior_player_matches']==1
    assert d['historical_player_evidence'][0]['HISTORICAL_CURRENT_WEIGHT']==.1
