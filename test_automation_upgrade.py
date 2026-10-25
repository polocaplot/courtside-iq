from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest
from advanced_feeds import parse_lineup_table, parse_defender_tables, read_cache
from automatic_audit import grade_completed, save_if_due, calibration_frame
from fantasy import FantasyScoring


def test_lineup_ids_map_to_real_names_and_no_fake_possessions():
    table = pd.DataFrame([{'GROUP_ID': '-1-2-3-4-5', 'TEAM_ID': 99, 'MIN': 20},
                          {'GROUP_ID': '-1-2-3-4-6', 'TEAM_ID': 99, 'MIN': 0},
                          {'GROUP_ID': '-1-2-3-4-9', 'TEAM_ID': 99, 'MIN': 5}])
    got = parse_lineup_table(table, {str(i): f'Player {i}' for i in range(1, 6)}, {'99': 'BOS'})
    assert len(got) == 1
    assert got.iloc[0]['PLAYER_5'] == 'Player 5'
    assert got.iloc[0]['POSSESSIONS'] == 20


def test_defender_buckets_are_explicit_proxies():
    rim = pd.DataFrame([{'TEAM_ID': 99, 'PLAYER_NAME': 'Defender', 'D_FGA': 25, 'D_FGM': 10}])
    three = pd.DataFrame([{'TEAM_ID': 99, 'PLAYER_NAME': 'Defender', 'D_FGA': 50, 'D_FGM': 20}])
    got = parse_defender_tables(rim, three, {'99': 'BOS'})
    assert set(got.ZONE) == {'Rim', 'Above-break 3'}
    assert len(got) == 2
    assert got.FGA_CONTESTED.sum() == 75


def test_no_cache_returns_empty(tmp_path):
    lp, dp, meta = read_cache('2026-27', tmp_path)
    assert lp.empty and dp.empty and meta == {}


def test_pregame_only_and_dedup(tmp_path):
    now = datetime.now(timezone.utc)
    game = {'id': 'g1', 'state': 'pre', 'tipoff': (now + timedelta(hours=4)).isoformat(),
            'home': {'abbr': 'BOS'}, 'away': {'abbr': 'NYK'}}
    result = {'box': pd.DataFrame([{'Player':'P1','Team':'BOS','MIN':30,'PTS':20,'REB':5,
                'AST':4,'STL':1,'BLK':0,'TOV':2}]), 'config': type('C', (), {'seed':42,'iterations':150000})(),
              'home_mean': 105., 'away_mean': 101., 'home_win': .57}
    scoring = FantasyScoring()
    path = tmp_path / 'history.json'
    assert save_if_due(result, game, scoring, 'NBA.com', path) is not None
    assert save_if_due(result, game, scoring, 'NBA.com', path) is None
    game['state'] = 'post'
    assert save_if_due(result, game, scoring, 'NBA.com', path) is None


def test_grade_only_final_and_reuse_actual(tmp_path):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=2)
    snapshot = {'id':'a','game_id':'g1','tipoff':old.isoformat(),'saved_at':(old-timedelta(hours=3)).isoformat(),
                'matchup':'NYK @ BOS','scoring':{'PTS':1,'REB':2,'AST':2,'STL':3,'BLK':3,'TOV':-1},
                'players':[{'Player':'P1','Team':'BOS','PTS':20,'FANTASY':32}],
                'home_projected':100,'away_projected':90,'home_win_prob':.65}
    import json
    history = tmp_path / 'hist.json'; history.write_text(json.dumps([snapshot]))
    final = {'id':'g1','state':'post','home':{'score':'105'},'away':{'score':'95'}}
    actual = pd.DataFrame([{'TEAM_ABBREVIATION':'BOS','PLAYER_NAME':'P1','PTS':22,'REB':5,
                             'AST':4,'STL':1,'BLK':0,'TOV':2}])
    calls = []
    def box(gid):
        calls.append(gid)
        return actual
    grades, errors = grade_completed(lambda d:[final],box,history,tmp_path/'grades.json')
    assert not errors and len(grades) == 1 and calls == ['g1']
    grades, errors = grade_completed(lambda d:[final],box,history,tmp_path/'grades.json')
    assert calls == ['g1']
    assert len(calibration_frame(grades)) == 1
    assert calibration_frame(grades).iloc[0]['FORECAST_SAVED_AT'] == snapshot['saved_at']
