"""Structural checks, NOT proof of predictive accuracy."""
import numpy as np
from data_sources import demo_dataset
from model import run, Config
from fantasy import FantasyScoring, fantasy_values


def _run(n=1200, **kw):
    teams, players = demo_dataset()
    return run(teams, players, 'BOS', 'NYK', Config(iterations=n, seed=192, possession_mode=True, **kw))


def test_events_consistent_with_points_and_makes():
    r = _run()
    for team in ('BOS', 'NYK'):
        a = r['possession_audits'][team]
        stats = r['team_arrays'][team]
        score = r['home_score'] if team == 'BOS' else r['away_score']
        assert np.array_equal(stats['PTS'].sum(axis=1), score)
        assert np.all(a['FGM'] <= a['FGA'])
        assert np.all(a['FTM'] <= a['FTA'])
        assert np.all(stats['3PM'].sum(axis=1) <= a['FGM'])
        assert np.all(stats['AST'].sum(axis=1) <= a['FGM'])
        assert np.all(a['OREB'] <= a['FGA'] - a['FGM'])
        assert np.all(a['POSS'] >= 78)
        assert np.all(stats['MIN'].sum(axis=1) >= 239.95)
        assert np.all(stats['PTS'] >= 0)
        assert np.all(stats['REB'] >= 0)


def test_stolen_balls_are_turnovers():
    r = _run()
    for offense, defense in (('BOS', 'NYK'), ('NYK', 'BOS')):
        assert np.all(r['team_arrays'][defense]['STL'].sum(axis=1) <=
                      r['team_arrays'][offense]['TOV'].sum(axis=1))


def test_possession_reproducible_and_injured_zero():
    a = _run(injury_overrides={'BOS Sample 1': 'Out'})
    b = _run(injury_overrides={'BOS Sample 1': 'Out'})
    assert np.array_equal(a['home_score'], b['home_score'])
    assert np.array_equal(a['team_arrays']['NYK']['REB'], b['team_arrays']['NYK']['REB'])
    for key in ('MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'):
        assert np.all(a['samples']['BOS|BOS Sample 1'][key] == 0)


def test_fantasy_scoring_from_possession_samples():
    r = _run()
    vals = r['samples']['BOS|BOS Sample 2']
    expected = vals['PTS'] + 2*vals['REB'] + 2*vals['AST'] + 3*vals['STL'] + 3*vals['BLK'] - vals['TOV']
    assert np.array_equal(fantasy_values(vals, FantasyScoring()), expected)


def test_player_shooting_box_is_event_consistent():
    r = _run()
    for team in ('BOS', 'NYK'):
        data = r['team_arrays'][team]
        audit = r['possession_audits'][team]
        for stat in ('FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB'):
            assert np.array_equal(data[stat].sum(axis=1), audit[stat])
        assert np.all(data['FGM'] <= data['FGA'])
        assert np.all(data['FTM'] <= data['FTA'])
        assert np.all(data['3PM'] <= data['3PA'])
        assert np.all(data['REB'] == data['OREB'] + data['DREB'])


def test_nonshooting_fouls_outside_bonus_continue_possession():
    result = _run(n=300)
    for team in ('BOS', 'NYK'):
        audit = result['possession_audits'][team]
        assert audit['NONSHOOTING_CONTINUATIONS'].sum() > 0
        assert np.all(audit['NONSHOOTING_CONTINUATIONS'] <= audit['POSS'])
