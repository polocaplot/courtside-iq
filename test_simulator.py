"""Offline invariants. None of these tests claim real-world predictive calibration."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from data_sources import demo_dataset, normalize_players, normalize_teams, injury_status
from model import Config, run, prop_probability
from fantasy import FantasyScoring, fantasy_values, leaderboard
from tracking import grade_snapshot, store_snapshot


@pytest.fixture(scope="module")
def example():
    t, p = demo_dataset()
    return run(t, p, "BOS", "NYK", Config(iterations=1200, seed=123))


def test_scoring_matches_league_settings():
    score = FantasyScoring()
    arrays = {"PTS": np.array([12]), "REB": np.array([7]), "AST": np.array([5]),
              "STL": np.array([2]), "BLK": np.array([1]), "TOV": np.array([3]),
              "3PM": np.array([4])}
    assert fantasy_values(arrays, score).item() == 12 + 7*2 + 5*2 + 2*3 + 1*3 - 3
    assert fantasy_values(arrays, FantasyScoring(THREE_PM=1)).item() == 12 + 7*2 + 5*2 + 2*3 + 1*3 - 3 + 4


def test_consistent_scores_and_regulation_minutes(example):
    for t, target in (("BOS", example["home_score"]), ("NYK", example["away_score"])):
        data = example["team_arrays"][t]
        assert np.array_equal(data["PTS"].sum(axis=1), target)
        assert np.all(data["MIN"].sum(axis=1) >= 239.95)
        assert np.all(data["MIN"].sum(axis=1) <= 265.05)  # Overtime has 25 more team minutes.
        assert np.all(data["PTS"] >= 0)


def test_injuries_remove_out_players():
    t, p = demo_dataset()
    result = run(t, p, "BOS", "NYK", Config(iterations=900, injury_overrides={"BOS Sample 1": "Out"}))
    a = result["samples"]["BOS|BOS Sample 1"]
    for stat in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"):
        assert np.all(a[stat] == 0)


def test_seed_reproducible():
    t, p = demo_dataset()
    kwargs = Config(iterations=1000, seed=1001)
    one = run(t, p, "BOS", "NYK", kwargs)
    two = run(t, p, "BOS", "NYK", kwargs)
    assert np.array_equal(one["home_score"], two["home_score"])
    assert np.array_equal(one["samples"]["BOS|BOS Sample 2"]["AST"], two["samples"]["BOS|BOS Sample 2"]["AST"])


def test_fantasy_rankings_based_on_actual_samples(example):
    board = leaderboard(example, FantasyScoring())
    assert len(board) == 20
    assert board["Fantasy AVG"].is_monotonic_decreasing
    who = board.iloc[0]
    val = fantasy_values(example["samples"][who["ID"]], FantasyScoring())
    assert abs(float(who["Fantasy AVG"]) - float(val.mean())) < .011


def test_props_over_under_push(example):
    key = "BOS|BOS Sample 1"
    test = prop_probability(example, key, "PTS", 25.)
    assert abs(test["over"] + test["under"] + test["push"] - 1.0) < 1e-10


def test_real_data_does_not_fallback_to_fictional_data():
    t, p = demo_dataset()
    assert all("Sample" in x for x in p["PLAYER_NAME"])
    assert not p["PLAYER_NAME"].str.contains("LeBron|Jokic|Curry").any()


def test_injury_name_parsing():
    assert injury_status("Out") == "Out"
    assert injury_status("Questionable - ankle") == "Questionable"
    assert injury_status("Day-to-Day") == "Questionable"
    assert injury_status("Probable") == "Probable"


def test_demo_snapshots_not_allowed(example, tmp_path):
    info = {"id":"1", "state":"pre", "tipoff":"2035-01-01T20:00:00Z",
            "away":{"abbr":"NYK"},"home":{"abbr":"BOS"}}
    with pytest.raises(ValueError, match="Demo"):
        store_snapshot(example, info, FantasyScoring(), "Synthetic demo", tmp_path / "test.json")


def test_accuracy_grade_uses_league_fantasy_weights():
    snap = {"players": [{"Player":"Test Example", "Team":"BOS", "FANTASY":30., "PTS":20.}],
            "scoring": FantasyScoring().weights()}
    actual = pd.DataFrame([{"PLAYER_NAME":"Test Example", "TEAM_ABBREVIATION":"BOS",
                           "PTS":12, "REB":7, "AST":5, "STL":2, "BLK":1, "TOV":3}])
    grade = grade_snapshot(snap,actual)
    expected=12+7*2+5*2+2*3+1*3-3
    assert grade["Actual FP"].iloc[0] == expected
    assert grade["FP Error"].iloc[0] == 30-expected
