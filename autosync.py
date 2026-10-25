"""Local feed-change detection for an *open* COURTSIDE IQ Streamlit dashboard.

Sources are advisory: a roster delta is not automatically a confirmed trade,
absence from a secondary injury feed does not prove availability. Snapshots for
failed sources are omitted, rather than misread as deleted players or recoveries.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from data_sources import injury_status, normalize_name

DEFAULT_DIR = Path.home() / ".courtsideiq"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_snapshot(rosters: dict[str, list[dict]] | None = None,
                  injury_feeds: dict[str, list[dict]] | None = None,
                  game: dict | None = None,
                  player_baselines: dict[str, dict[str, float]] | None = None) -> dict:
    """Create a clean comparable snapshot; only include successfully read sources."""
    snap: dict[str, Any] = {"checked_at": _utc(), "rosters": {}, "injuries": {},
                            "baselines": {}, "games": {}}
    for team, roster in (rosters or {}).items():
        members = {}
        for p in roster:
            n = normalize_name(p.get("player", ""))
            if n:
                members[n] = str(p.get("player", "")).strip()
        snap["rosters"][team] = members
    for team, items in (injury_feeds or {}).items():
        reports = {}
        for i in items:
            n = normalize_name(i.get("player", ""))
            if n:
                reports[n] = {"name": i.get("player", ""),
                              "status": injury_status(i.get("status", "")),
                              "raw_status": str(i.get("status", "")),
                              "note": str(i.get("note", ""))[:160]}
        snap["injuries"][team] = reports
    for team, people in (player_baselines or {}).items():
        snap["baselines"][team] = {
            normalize_name(n): {"name": n, "min": round(float(v.get("MIN", 0)), 1),
                                "pts": round(float(v.get("PTS", 0)), 1),
                                "ast": round(float(v.get("AST", 0)), 1),
                                "reb": round(float(v.get("REB", 0)), 1)}
            for n, v in people.items() if normalize_name(n)
        }
    if game and game.get("id"):
        snap["games"][str(game["id"])] = {
            "name": f'{game.get("away", {}).get("abbr", "?")} @ {game.get("home", {}).get("abbr", "?")}',
            "state": game.get("state", ""), "tipoff": game.get("tipoff", ""),
            "score": [game.get("away", {}).get("score", ""), game.get("home", {}).get("score", "")],
        }
    return snap


def changes(old: dict, new: dict) -> list[dict]:
    """Compare only sources present in both snapshots; emit honest classifications."""
    events = []
    stamp = new.get("checked_at") or _utc()
    def add(kind, team, name, before, after, detail):
        events.append({"timestamp": stamp, "type": kind, "team": team, "player": name,
                       "before": str(before), "after": str(after), "detail": detail})
    for team, current in new.get("rosters", {}).items():
        prior = old.get("rosters", {}).get(team)
        if prior is None:
            continue
        for key in sorted(current.keys() - prior.keys()):
            # Not a confirmed signing or trade unless team history corroborates it.
            old_teams = [t for t, r in old.get("rosters", {}).items() if t != team and key in r]
            kind = "Possible team move" if old_teams else "Roster addition"
            note = (f"Previously seen on {', '.join(old_teams)}; verify transaction." if old_teams
                    else "New on the polled roster; verify signing or call-up.")
            add(kind, team, current[key], ", ".join(old_teams) or "Not previously on roster", team, note)
        for key in sorted(prior.keys() - current.keys()):
            newer_teams = [t for t, r in new.get("rosters", {}).items() if t != team and key in r]
            add("Roster departure", team, prior[key], team, ", ".join(newer_teams) or "Not listed",
                "Missing from latest roster; not necessarily a trade or waiver.")
    for team, reports in new.get("injuries", {}).items():
        old_reports = old.get("injuries", {}).get(team)
        if old_reports is None:
            continue
        for key, current in reports.items():
            last = old_reports.get(key)
            if not last:
                add("New injury listing", team, current["name"], "Not listed", current["status"],
                    current.get("note", "") or "Listed on the latest secondary injury feed.")
            elif last["status"] != current["status"]:
                add("Injury status change", team, current["name"], last["status"], current["status"],
                    current.get("note", "") or "Status changed on the injury feed.")
        for key, last in old_reports.items():
            if key not in reports:
                add("Injury listing removed", team, last["name"], last["status"], "Not listed",
                    "No longer in this feed; this does NOT confirm the player is cleared to play.")
    for team, current in new.get("baselines", {}).items():
        prior = old.get("baselines", {}).get(team)
        if prior is None:
            continue
        for key, p in current.items():
            if key not in prior:
                continue
            oldp = prior[key]
            dmin = p["min"] - oldp["min"]
            dpts = p["pts"] - oldp["pts"]
            if abs(dmin) >= 2.0:
                add("Minutes trend", team, p["name"], f'{oldp["min"]:.1f}', f'{p["min"]:.1f}',
                    f"Season/recent blended minutes changed {dmin:+.1f}. This is not a confirmed rotation change.")
            elif abs(dpts) >= 3.0:
                add("Production trend", team, p["name"], f'{oldp["pts"]:.1f}', f'{p["pts"]:.1f}',
                    f"Scoring baseline changed {dpts:+.1f}. Review sample size.")
    for game_id, g in new.get("games", {}).items():
        last = old.get("games", {}).get(game_id)
        if not last:
            continue
        if last["state"] != g["state"]:
            add("Game status", g["name"], "", last["state"], g["state"],
                "Pregame forecasts are NOT live, score-conditional predictions.")
        if last["tipoff"] != g["tipoff"]:
            add("Tipoff changed", g["name"], "", last["tipoff"], g["tipoff"],
                "Start time changed on the schedule feed.")
    return events


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _save(path: Path, val: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def record(snapshot: dict, root: Path = DEFAULT_DIR) -> list[dict]:
    """Diff and save only verified source keys; keep last valid state on feed failure."""
    state_file, event_file = root / "sync_state.json", root / "sync_events.json"
    prev = _load(state_file, {})
    found = changes(prev, snapshot)
    merged = {**prev, "checked_at": snapshot.get("checked_at", _utc())}
    for category in ("rosters", "injuries", "baselines", "games"):
        merged[category] = {**prev.get(category, {}), **snapshot.get(category, {})}
    _save(state_file, merged)
    if found:
        _save(event_file, (_load(event_file, []) + found)[-300:])
    return found


def history(root: Path = DEFAULT_DIR, limit: int = 75) -> list[dict]:
    return list(reversed(_load(root / "sync_events.json", [])[-limit:]))


def last_check(root: Path = DEFAULT_DIR) -> str | None:
    return _load(root / "sync_state.json", {}).get("checked_at")


def reconcile_editor(previous, roster, statuses: dict[str, str]):
    """Refresh model inputs while preserving only explicitly locked manual overrides."""
    import pandas as pd
    old = {}
    if previous is not None and not previous.empty:
        old = {(str(r["Team"]), normalize_name(r["Player"])): r
               for _, r in previous.iterrows()}
    rows = []
    for _, p in roster.iterrows():
        name, team = str(p["PLAYER_NAME"]), str(p["TEAM_ABBREVIATION"])
        key = (team, normalize_name(name))
        prev = old.get(key)
        locked = bool(prev.get("Manual lock", False)) if prev is not None else False
        rows.append({"Player": name, "Team": team, "Base min": round(float(p["MIN"]), 1),
                     "Availability": str(prev["Availability"]) if locked else statuses.get(name, "Healthy"),
                     "Minute cap": float(prev.get("Minute cap", 48.)) if prev is not None else 48.,
                     "Usage x": float(prev.get("Usage x", 1.)) if prev is not None else 1.,
                     "Manual lock": locked})
    return pd.DataFrame(rows)
