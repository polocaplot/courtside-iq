"""Checks that updates distinguish signals from confirmed roster/injury facts."""
from pathlib import Path
import pandas as pd
from autosync import make_snapshot, changes, record, history, reconcile_editor


def test_detect_roster_moves_and_injury_changes():
    old = make_snapshot({"BOS": [{"player": "Player One"}],
                         "NYK": [{"player": "Player Two"}]},
                        {"BOS": [{"player": "Player One", "status": "Questionable", "note": "ankle"}]})
    new = make_snapshot({"BOS": [],
                         "NYK": [{"player": "Player Two"}, {"player": "Player One"}]},
                        {"BOS": [{"player": "Player One", "status": "Out", "note": "ankle"}]})
    found = changes(old, new)
    assert any(e["type"] == "Possible team move" for e in found)
    assert any(e["type"] == "Roster departure" for e in found)
    assert any(e["type"] == "Injury status change" and e["after"] == "Out" for e in found)


def test_feeds_not_returned_cannot_imply_removed_players():
    old = make_snapshot({"BOS": [{"player": "Player One"}]},
                        {"BOS": [{"player": "Player One", "status": "Out"}]})
    new = make_snapshot({}, {})
    assert changes(old, new) == []


def test_record_persists_and_first_read_is_baseline(tmp_path):
    one = make_snapshot({"BOS": [{"player": "Player One"}]})
    assert record(one, tmp_path) == []
    two = make_snapshot({"BOS": [{"player": "Player One"}, {"player": "Player Two"}]})
    diff = record(two, tmp_path)
    assert len(diff) == 1 and diff[0]["type"] == "Roster addition"
    assert len(history(tmp_path)) == 1
    assert record(make_snapshot(), tmp_path) == []  # missing feed must not erase cached team
    assert record(two, tmp_path) == []


def test_automatic_injuries_refresh_only_unlocked_rows():
    players = pd.DataFrame([{"PLAYER_NAME": "Player One", "TEAM_ABBREVIATION": "BOS", "MIN": 33},
                            {"PLAYER_NAME": "Player Two", "TEAM_ABBREVIATION": "BOS", "MIN": 25}])
    prev = reconcile_editor(None, players, {"Player One": "Questionable"})
    assert prev.loc[0, "Availability"] == "Questionable"
    prev.loc[0, "Manual lock"] = True
    prev.loc[0, "Availability"] = "Healthy"
    updated = reconcile_editor(prev, players, {"Player One": "Out", "Player Two": "Out"})
    assert updated.loc[0, "Availability"] == "Healthy"
    assert updated.loc[1, "Availability"] == "Out"
    assert updated.loc[0, "Manual lock"]
