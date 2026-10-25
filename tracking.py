"""Local, timestamped prediction snapshots and actual-result evaluation.

Only snapshots saved *before* a scheduled tipoff qualify for pregame evaluation.
Actual results must be final, and are matched by normalized athlete names/team.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
import pandas as pd
from data_sources import normalize_name
from fantasy import FantasyScoring, score_boxscore

DEFAULT_PATH = Path.home() / ".courtsideiq" / "history.json"


def store_snapshot(result: dict, game: dict, scoring: FantasyScoring, source: str, path: Path = DEFAULT_PATH) -> dict:
    if source.lower().startswith("synthetic") or source.lower().startswith("demo"):
        raise ValueError("Demo scenarios are not eligible for accuracy tracking.")
    if game.get("state") != "pre":
        raise ValueError("Only pregame matchups can be saved for a fair accuracy audit.")
    tipoff_text = game.get("tipoff", "")
    if not tipoff_text:
        raise ValueError("Game start time is required to save a pregame forecast.")
    tipoff = datetime.fromisoformat(tipoff_text.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= tipoff:
        raise ValueError("Tipoff has already passed; cannot create a pregame forecast.")
    box = result["box"].copy()
    box["FANTASY"] = score_boxscore(box, scoring)
    record = {
        "id": uuid.uuid4().hex[:12], "saved_at": datetime.now(timezone.utc).isoformat(),
        "game_id": str(game["id"]), "tipoff": tipoff.isoformat(),
        "matchup": f'{game["away"]["abbr"]} @ {game["home"]["abbr"]}',
        "source": source, "seed": result["config"].seed,
        "simulations": result["config"].iterations, "scoring": scoring.weights(),
        "home_projected": result["home_mean"], "away_projected": result["away_mean"],
        "home_win_prob": float(result["home_win"]),
        "players": box[["Player", "Team", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FANTASY"]].to_dict("records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(path)
    history.append(record)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return record


def load_history(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (ValueError, OSError):
        return []


def grade_snapshot(snapshot: dict, actual: pd.DataFrame) -> pd.DataFrame:
    if actual.empty:
        return pd.DataFrame()
    actual = actual.copy()
    actual["key"] = actual["TEAM_ABBREVIATION"].astype(str).str.upper() + "|" + actual["PLAYER_NAME"].map(normalize_name)
    actual = actual.set_index("key")
    weights = snapshot["scoring"]
    rows = []
    for p in snapshot["players"]:
        key = f'{p["Team"]}|{normalize_name(p["Player"])}'
        if key not in actual.index:
            continue
        a = actual.loc[key]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        actual_fantasy = sum(float(a.get(k, 0)) * float(w) for k, w in weights.items())
        rows.append({"Player": p["Player"], "Team": p["Team"],
                     "Predicted FP": round(float(p["FANTASY"]), 2), "Actual FP": round(actual_fantasy, 2),
                     "FP Error": round(float(p["FANTASY"]) - actual_fantasy, 2),
                     "Abs. Error": round(abs(float(p["FANTASY"]) - actual_fantasy), 2),
                     "Projected PTS": p["PTS"], "Actual PTS": a.get("PTS", 0)})
    return pd.DataFrame(rows)
