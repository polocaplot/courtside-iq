from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from training import build_features, train, load, forecast, STATS
from markets import american_probability, compare_props, parse_props


def fabricated_logs(seed=9):
    """TEST FIXTURE ONLY: entirely synthetic, never shipped as NBA training data."""
    rng = np.random.default_rng(seed)
    rows = []
    first = datetime(2024, 10, 20, tzinfo=timezone.utc)
    for game in range(65):
        date = first + timedelta(days=game)
        for team, rival in [('BOS', 'NYK'), ('NYK', 'BOS')]:
            for j in range(10):
                minute = 31 - j*2 + rng.normal(0,2)
                pts = max(0, round((minute*.56 + 2*(team == 'BOS')) + rng.normal(0, 5)))
                rows.append({'GAME_ID': f'00{game:04}', 'GAME_DATE': date.isoformat(),
                    'PLAYER_ID': f'{team}{j}', 'PLAYER_NAME': f'{team} Player {j}',
                    'TEAM_ABBREVIATION': team, 'MATCHUP': f'{team} vs. {rival}' if team == 'BOS' else f'{team} @ {rival}',
                    'MIN': minute, 'PTS': pts, 'REB': max(0, round(minute*.15+rng.normal(0,2))),
                    'AST':max(0,round(minute*.1+rng.normal(0,2))),
                    'STL':int(rng.poisson(.6)), 'BLK':int(rng.poisson(.5)),
                    'TOV':int(rng.poisson(1.2)), 'FG3M':int(rng.poisson(1.)),
                    'FGA':int(rng.poisson(9)), 'FTA':int(rng.poisson(3)),
                    'OREB':int(rng.poisson(1.))})
    return pd.DataFrame(rows)


def test_no_target_leak_same_game():
    logs=fabricated_logs()
    one=build_features(logs)
    altered=logs.copy()
    ix=altered[(altered['PLAYER_ID']=='BOS0') & (altered['GAME_ID']=='000020')].index
    assert len(ix)==1
    altered.loc[ix, 'PTS'] += 500
    two=build_features(altered)
    a=one[(one['PLAYER_ID']=='BOS0') & (one['GAME_ID']=='000020')].iloc[0]
    b=two[(two['PLAYER_ID']=='BOS0') & (two['GAME_ID']=='000020')].iloc[0]
    assert a['PTS_L10']==b['PTS_L10']
    assert a['OPP_DEF_P10']==b['OPP_DEF_P10']
    # Later feature *should* incorporate the previously played game.
    later1=one[(one['PLAYER_ID']=='BOS0') & (one['GAME_ID']=='000021')].iloc[0]
    later2=two[(two['PLAYER_ID']=='BOS0') & (two['GAME_ID']=='000021')].iloc[0]
    assert later1['PTS_L10']!=later2['PTS_L10']


def test_chronological_training_and_forecast(tmp_path):
    logs=fabricated_logs()
    m=train(logs, folder=tmp_path, min_games=500)
    assert m['all_games']>500
    assert m['holdout_games']>100
    assert all('live_model_weight' in r for r in m['results'])
    m2, models, history=load(tmp_path)
    assert len(models)==8
    assert m2['holdout_from']==m['holdout_from']
    recent = logs.sort_values('GAME_DATE').groupby('PLAYER_ID').tail(1).copy()
    players = recent[['PLAYER_NAME','TEAM_ABBREVIATION',*STATS]].rename(columns={'FG3M':'FG3M'})
    date=(pd.Timestamp(logs['GAME_DATE'].max()) + pd.Timedelta(days=1)).isoformat()
    updated, audit, _=forecast(players, 'BOS', 'NYK', date, tmp_path)
    assert len(audit)==len(players)
    assert (updated['PTS']>=0).all()
    assert (updated['PTS']<=70).all()


def test_moneyline_and_prop_comparison():
    assert abs(american_probability(-110) - 110/210)<1e-10
    assert abs(american_probability(120) - 100/220)<1e-10
    data={'bookmakers':[{'title':'Sample Book','markets':[{'key':'player_points','outcomes':[
        {'name':'Over','description':'BOS Player 0','point':23.5,'price':-110},
        {'name':'Under','description':'BOS Player 0','point':23.5,'price':-110}]}]}]}
    df=parse_props(data)
    assert len(df)==2
    samples={'BOS|BOS Player 0': {'PTS':np.array([20,25,30,20,30])}}
    out=compare_props({'samples':samples},df)
    assert len(out)==2
    assert abs(out[out['Side']=='Over'].iloc[0]['Model probability']-.6)<1e-10
    assert abs(out[out['Side']=='Over'].iloc[0]['No-vig fair probability']-.5)<1e-10
