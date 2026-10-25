from pathlib import Path


def test_historical_downloads_are_scoped_to_selected_matchup():
    code = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    section = code.split('def ensure_selected_shot_profiles()', 1)[1].split('\ndef launch_sim(', 1)[0]
    assert "players_for_selected['TEAM_ABBREVIATION'].isin((away, home))" in section
    assert 'for row in roster_rows:' in section
    assert 'for idx, prior_season in enumerate(seasons):' in section
    assert "sort_values('_missing_evidence', ascending=False" in section
    assert 'for missing_first in (True, False):' in section
    assert 'covered_cached' in section
    assert 'historical_seasons(season_str, 5)' in section
    assert "included = selected_roster[['TEAM_ABBREVIATION', 'PLAYER_NAME']]" in section
