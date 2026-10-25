import pandas as pd
from advanced_feeds import parse_defender_tables


def test_current_official_defender_rows_survive_roster_endpoint_mismatch():
    rim = pd.DataFrame([{'PLAYER_NAME':'Example Defender', 'CLOSE_DEF_PERSON_ID':123,
        'PLAYER_LAST_TEAM_ABBREVIATION':'MIA','D_FGA':80,'D_FGM':38}])
    teams={'1610612748':'MIA'}
    missing_roster={'999':'MIA'}
    assert parse_defender_tables(rim, pd.DataFrame(), teams,
        roster_teams=missing_roster).empty
    actual = parse_defender_tables(rim, pd.DataFrame(), teams,
        roster_teams=missing_roster, allow_official_team_fallback=True)
    assert len(actual)==1
    assert actual.iloc[0]['TEAM_ABBREVIATION']=='MIA'
    assert actual.iloc[0]['FGA_CONTESTED']==80


def test_official_fallback_never_accepts_unrecognized_team():
    rim = pd.DataFrame([{'PLAYER_NAME':'Example Defender', 'CLOSE_DEF_PERSON_ID':123,
        'PLAYER_LAST_TEAM_ABBREVIATION':'FAKE','D_FGA':80,'D_FGM':38}])
    assert parse_defender_tables(rim, pd.DataFrame(), {'1':'MIA'},
        roster_teams={'999':'MIA'}, allow_official_team_fallback=True).empty
