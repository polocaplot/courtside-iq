import pandas as pd
from advanced_feeds import parse_defender_tables


def test_defended_counts_alias_supported():
    raw = pd.DataFrame([{'CLOSE_DEF_PERSON_ID': 7, 'PLAYER_NAME': 'Example Defender',
                         'DEFENDED_FGA': 17, 'DEFENDED_FGM': 6}])
    result = parse_defender_tables(raw, pd.DataFrame(), {}, roster_teams={'7': 'MIA'})
    assert len(result) == 1
    assert result.iloc[0]['FGA_CONTESTED'] == 17
    assert result.iloc[0]['FGM_ALLOWED'] == 6


def test_generic_offensive_counts_not_accepted():
    raw = pd.DataFrame([{'CLOSE_DEF_PERSON_ID': 7, 'PLAYER_NAME': 'Example Defender',
                         'FGA': 17, 'FGM': 6}])
    result = parse_defender_tables(raw, pd.DataFrame(), {}, roster_teams={'7': 'MIA'})
    assert result.empty


def test_real_nba_rim_and_three_schemas_current_roster():
    rim = pd.DataFrame([{'CLOSE_DEF_PERSON_ID': 7, 'PLAYER_NAME': 'Example Defender',
                         'FGA_LT_06': 23, 'FGM_LT_06': 12}])
    three = pd.DataFrame([{'CLOSE_DEF_PERSON_ID': 7, 'PLAYER_NAME': 'Example Defender',
                           'FG3A': 31, 'FG3M': 11}])
    result = parse_defender_tables(rim, three, {}, roster_teams={'7': 'MIA'})
    assert len(result) == 2
    by_zone = result.set_index('ZONE')
    assert by_zone.loc['Rim', 'FGA_CONTESTED'] == 23
    assert by_zone.loc['Rim', 'FGM_ALLOWED'] == 12
    assert by_zone.loc['Above-break 3', 'FGA_CONTESTED'] == 31
    assert by_zone.loc['Above-break 3', 'FGM_ALLOWED'] == 11

def test_current_and_prior_defender_weight_and_roster_mapping():
    rim = pd.DataFrame([{'CLOSE_DEF_PERSON_ID': 7, 'PLAYER_NAME': 'Example Defender',
                         'FGA_LT_06': 50, 'FGM_LT_06': 20}])
    result = parse_defender_tables(rim, pd.DataFrame(), {},
                                   roster_teams={'7': 'CHA'}, season_weight=0.2)
    assert len(result) == 1
    assert result.iloc[0]['TEAM_ABBREVIATION'] == 'CHA'
    assert result.iloc[0]['FGA_CONTESTED'] == 10
    assert result.iloc[0]['FGM_ALLOWED'] == 4
