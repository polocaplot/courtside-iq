import pandas as pd
from advanced_feeds import parse_defender_tables, previous_season, historical_defender_weight
from defender_intelligence import defender_zone_adjustments


def test_actual_nba_defender_columns_resolve_and_feed_model():
    rim = pd.DataFrame([dict(CLOSE_DEF_PERSON_ID=77, PLAYER_NAME='Test Defender',
        PLAYER_LAST_TEAM_ABBREVIATION='OLD', GP=60, D_FGA=120, D_FGM=42)])
    three = pd.DataFrame([dict(CLOSE_DEF_PERSON_ID=77, PLAYER_NAME='Test Defender',
        PLAYER_LAST_TEAM_ABBREVIATION='OLD', GP=60, D_FGA=100, D_FGM=27)])
    d = parse_defender_tables(rim, three, {}, {'77':'BOS'})
    assert len(d) == 2
    assert set(d.TEAM_ABBREVIATION) == {'BOS'}
    assert d.loc[d.ZONE.eq('Rim'), 'FGA_CONTESTED'].iloc[0] == 120
    adjust = defender_zone_adjustments(pd.DataFrame({'PLAYER_NAME':['Test Defender']}), 'BOS', d)
    assert adjust.shape == (1, 5)
    assert adjust[0, 0] != 0


def test_history_scales_and_traded_players_not_misattributed():
    prior = pd.DataFrame([dict(CLOSE_DEF_PERSON_ID=77, PLAYER_NAME='Test Defender',
        PLAYER_LAST_TEAM_ABBREVIATION='OLD', GP=60, D_FGA=120, D_FGM=42),
        dict(CLOSE_DEF_PERSON_ID=88, PLAYER_NAME='Former Player',
        PLAYER_LAST_TEAM_ABBREVIATION='OLD', GP=60, D_FGA=120, D_FGM=42)])
    weight = historical_defender_weight([prior])
    assert weight == .2
    d = parse_defender_tables(prior, prior, {}, {'77':'BOS'}, season_weight=weight)
    assert set(d.DEFENDER_NAME) == {'Test Defender'}
    assert set(d.TEAM_ABBREVIATION) == {'BOS'}
    assert d.loc[d.ZONE.eq('Rim'), 'FGA_CONTESTED'].iloc[0] == 24
    assert previous_season('2026-27') == '2025-26'


def test_three_point_aggregate_not_double_counted():
    three = pd.DataFrame([dict(CLOSE_DEF_PERSON_ID=77, PLAYER_NAME='Test Defender',
        GP=50, D_FGA=80, D_FGM=24)])
    d = parse_defender_tables(pd.DataFrame(), three, {}, {'77': 'BOS'})
    assert len(d) == 1
    assert d.iloc[0]['ZONE'] == 'Above-break 3'
    assert d['FGA_CONTESTED'].sum() == 80


def test_missing_defense_cache_retries_early(tmp_path):
    from advanced_feeds import should_refresh, _paths, FEED_SCHEMA_VERSION
    import json
    from datetime import datetime, timezone
    lp, dp, mp = _paths('2026-27', tmp_path)
    mp.write_text(json.dumps({'checked_at': datetime.now(timezone.utc).isoformat(), 'errors':['defense failed']}))
    assert should_refresh('2026-27', hours=24, root=tmp_path)  # legacy failed cache must be retried
    mp.write_text(json.dumps({'checked_at': datetime.now(timezone.utc).isoformat(), 'feed_schema_version': FEED_SCHEMA_VERSION}))
    assert not should_refresh('2026-27', hours=24, root=tmp_path)
    from datetime import timedelta
    mp.write_text(json.dumps({'checked_at': (datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()}))
    assert should_refresh('2026-27', hours=24, root=tmp_path)


def test_unambiguous_roster_name_recovers_mismatched_nba_ids():
    from advanced_feeds import _roster_identity
    roster = pd.DataFrame([
        {'PLAYER_ID': 12, 'PLAYER_NAME': 'José Alvarado', 'TEAM_ABBREVIATION': 'NYK'},
        {'PLAYER_ID': None, 'PLAYER_NAME': 'Giannis Antetokounmpo', 'TEAM_ABBREVIATION': 'MIA'},
    ])
    by_id, by_name = _roster_identity(roster)
    frame = pd.DataFrame([
        {'CLOSE_DEF_PERSON_ID': 999, 'PLAYER_NAME': 'Jose Alvarado',
         'PLAYER_LAST_TEAM_ABBREVIATION': 'OLD', 'D_FGA': 25, 'D_FGM': 8},
        {'CLOSE_DEF_PERSON_ID': 777, 'PLAYER_NAME': 'Giannis Antetokounmpo',
         'PLAYER_LAST_TEAM_ABBREVIATION': 'OLD', 'D_FGA': 40, 'D_FGM': 12},
        {'CLOSE_DEF_PERSON_ID': 555, 'PLAYER_NAME': 'Not on roster',
         'PLAYER_LAST_TEAM_ABBREVIATION': 'OLD', 'D_FGA': 40, 'D_FGM': 12},
    ])
    result = parse_defender_tables(frame, pd.DataFrame(), {},
        {pid: team for pid, (team, _) in by_id.items()}, roster_names=by_name)
    assert len(result) == 2
    assert set(result.TEAM_ABBREVIATION) == {'NYK', 'MIA'}
    assert set(result.DEFENDER_NAME) == {'José Alvarado', 'Giannis Antetokounmpo'}
    assert result.FGA_CONTESTED.sum() == 65


def test_name_matching_does_not_guess_duplicate_roster_names():
    from advanced_feeds import _roster_identity
    roster = pd.DataFrame([
        {'PLAYER_NAME': 'Alex Smith', 'TEAM_ABBREVIATION': 'NYK'},
        {'PLAYER_NAME': 'Alex Smith', 'TEAM_ABBREVIATION': 'BOS'},
    ])
    ids, names = _roster_identity(roster)
    assert 'alexsmith' not in names
