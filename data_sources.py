"""Live NBA schedule, baseline stats, and transparent injury feeds.

ESPN endpoints are unofficial and can change, NBA.com Stats can throttle.
Every call raises on failure; the dashboard never substitutes fabricated live data.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from generalization import blend_rates
import requests
import pandas as pd
import numpy as np

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 CourtsideIQ/1.0", "Accept": "application/json"}


def get_json(url: str, params: dict | None = None) -> dict:
    res = requests.get(url, params=params, headers=HEADERS, timeout=18)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected sports data response")
    return data


def schedule(day: date) -> list[dict]:
    payload = get_json(f"{ESPN}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": "50"})
    out = []
    for e in payload.get("events", []):
        comps = e.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        def pick(x):
            team = x.get("team", {})
            return {"abbr": team.get("abbreviation", ""), "name": team.get("displayName", ""),
                    "id": str(team.get("id", "")), "score": x.get("score", "")}
        status = e.get("status", {}).get("type", {})
        out.append({"id": str(e.get("id", "")), "home": pick(home), "away": pick(away),
                    "tipoff": e.get("date", ""), "status": status.get("name", ""),
                    "state": status.get("state", "pre")})
    return out


def injuries(team_id: str) -> tuple[list[dict], str]:
    """ESPN injuries are NOT the official NBA game injury report. May be missing records."""
    payload = get_json(f"{ESPN}/teams/{team_id}/injuries")
    items = payload.get("injuries", [])
    if isinstance(items, dict):
        items = items.get("items", [])
    parsed = []
    for i in items or []:
        athlete = i.get("athlete", {})
        if isinstance(athlete, list):
            athlete = athlete[0] if athlete else {}
        name = (athlete.get("displayName") or athlete.get("fullName") or
                i.get("name") or i.get("displayName") or "")
        status = i.get("status", "")
        if isinstance(status, dict):
            status = status.get("name") or status.get("type") or status.get("description", "")
        desc = i.get("shortComment") or i.get("longComment") or i.get("description") or ""
        if name:
            parsed.append({"player": str(name), "status": str(status), "note": str(desc)[:160]})
    return parsed, "ESPN injuries (unofficial; not a confirmed starting lineup)"


def _league_stat(endpoint, season: str, last_n: int) -> pd.DataFrame:
    return endpoint(
        season=season, season_type_all_star="Regular Season", per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base", last_n_games=last_n,
        league_id_nullable="00", timeout=35,
    ).get_data_frames()[0]


def _combine(base: pd.DataFrame, recent: pd.DataFrame, key: str, columns: list[str], weight: float) -> pd.DataFrame:
    out = base.copy()
    if recent.empty:
        return out
    rec = recent.set_index(key)
    for col in columns:
        if col not in out.columns or col not in rec.columns:
            continue
        mapped = out[key].map(rec[col])
        out[col] = (1 - weight) * pd.to_numeric(out[col], errors="coerce") + weight * pd.to_numeric(mapped, errors="coerce").fillna(pd.to_numeric(out[col], errors="coerce"))
    return out



def reconcile_shooting_lines(players: pd.DataFrame) -> pd.DataFrame:
    """Enforce basketball identities after independently blending season rates.

    Historical priors must not create three-point makes exceeding total FGM,
    FGM exceeding attempts, or makes exceeding attempts. Never invent points
    or field-goal makes when source observations are absent: corrections are
    capped by the observed attempt counts, and a diagnostic flag is retained.
    """
    p = players.copy()
    needed = ('FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM')
    if not all(col in p for col in needed):
        return p
    before = p[list(needed)].apply(pd.to_numeric, errors='coerce')
    fixed = before.clip(lower=0)
    fixed['FG3A'] = fixed['FG3A'].clip(upper=fixed['FGA'])
    fixed['FG3M'] = fixed['FG3M'].clip(upper=fixed['FG3A'])
    fixed['FGM'] = fixed['FGM'].clip(lower=fixed['FG3M'], upper=fixed['FGA'])
    fixed['FTM'] = fixed['FTM'].clip(upper=fixed['FTA'])
    changed = (before - fixed).abs().gt(1e-6).any(axis=1)
    for col in needed:
        p[col] = fixed[col].where(fixed[col].notna(), before[col])
    p['SHOOTING_INPUT_RECONCILED'] = changed
    if 'FG_PCT' in p:
        p['FG_PCT'] = (p['FGM'] / p['FGA'].replace(0, np.nan)).fillna(0).clip(0, 1)
    return p


def season_stats(season: str, recent_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-season + recent form with pace/defense from NBA advanced team stats.

    Current season samples are shrunk toward the previous season when available.
    Current-season roster membership remains authoritative (previous-season
    athletes are never blindly placed on their old team's current roster).
    """
    from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
    t = _league_stat(leaguedashteamstats.LeagueDashTeamStats, season, 0)
    p = _league_stat(leaguedashplayerstats.LeagueDashPlayerStats, season, 0)
    if t.empty or p.empty:
        raise ValueError("NBA Stats returned empty current-season team or player tables")
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star="Regular Season",
        per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced",
        league_id_nullable="00", last_n_games=0, timeout=35
    ).get_data_frames()[0]
    if not adv.empty:
        useful = [x for x in ("TEAM_ID", "PACE", "DEF_RATING") if x in adv]
        t = t.merge(adv[useful], on="TEAM_ID", how="left")
        # Estimate opponent points per game from points allowed per 100 possessions.
        if "DEF_RATING" in t and "PACE" in t:
            t["OPP_PTS"] = pd.to_numeric(t["DEF_RATING"], errors="coerce") * pd.to_numeric(t["PACE"], errors="coerce") / 100
    # Preserve actual recent minute observations BEFORE blending rate columns.
    # These are separate evidence columns, not fabricated recent-game priors.
    p['GAMES_PLAYED'] = pd.to_numeric(p.get('GP', 0), errors='coerce').fillna(0)
    for _window in (5, 10):
        try:
            _recent = _league_stat(leaguedashplayerstats.LeagueDashPlayerStats, season, _window)
            if not _recent.empty and {'PLAYER_ID', 'MIN'}.issubset(_recent.columns):
                _by_id = _recent.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID')['MIN']
                p[f'MIN_L{_window}'] = pd.to_numeric(p['PLAYER_ID'].map(_by_id), errors='coerce')
        except Exception:
            # Missing observations stay missing, never replaced with made-up minutes.
            p[f'MIN_L{_window}'] = np.nan
    if recent_n:
        try:
            tr = _league_stat(leaguedashteamstats.LeagueDashTeamStats, season, recent_n)
            pr = _league_stat(leaguedashplayerstats.LeagueDashPlayerStats, season, recent_n)
            t = _combine(t, tr, "TEAM_ID", ["PTS", "REB", "AST"], .20)
            p = _combine(p, pr, "PLAYER_ID", ["PTS", "REB", "AST", "MIN", "FG3M", "FGA", "FGM", "FG3A", "FTA", "FTM", "STL", "BLK", "TOV"], .27)
        except Exception:
            pass  # The dashboard still reports the currently available dataset.
    # Past seasons are explicit evidence, never silently assumed to be present.
    p['HISTORICAL_PRIOR_USED'] = False
    p['HISTORICAL_PRIOR_SEASON'] = ''
    p['HISTORICAL_CURRENT_WEIGHT'] = 1.0
    p['HISTORICAL_MINUTES_WEIGHT'] = 1.0
    t['HISTORICAL_PRIOR_USED'] = False
    t['HISTORICAL_PRIOR_SEASON'] = ''
    historical_errors = []
    try:
        year = int(season[:4])
        prior_season = f"{year-1}-{str(year)[-2:]}"
        tp = _league_stat(leaguedashteamstats.LeagueDashTeamStats, prior_season, 0)
        pp = _league_stat(leaguedashplayerstats.LeagueDashPlayerStats, prior_season, 0)
        if not tp.empty:
            t['HISTORICAL_PRIOR_USED'] = t['TEAM_ID'].isin(tp['TEAM_ID'])
            t['HISTORICAL_PRIOR_SEASON'] = prior_season
            # Early-season team defense is volatile. Blend the opponent scoring
            # baseline toward last year's TEAM defense, not toward old player
            # memberships. If the optional advanced feed fails, keep current
            # values rather than manufacturing a prior defensive rating.
            try:
                old_adv = leaguedashteamstats.LeagueDashTeamStats(
                    season=prior_season, season_type_all_star="Regular Season",
                    per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced",
                    league_id_nullable="00", last_n_games=0, timeout=35
                ).get_data_frames()[0]
                if not old_adv.empty and {'TEAM_ID', 'PACE', 'DEF_RATING'}.issubset(old_adv.columns):
                    old_adv = old_adv.drop_duplicates('TEAM_ID').set_index('TEAM_ID')
                    old_pace = t['TEAM_ID'].map(old_adv['PACE'])
                    old_opp = t['TEAM_ID'].map(old_adv['DEF_RATING'] * old_adv['PACE'] / 100.)
                    current_games = pd.to_numeric(t['GP'], errors='coerce').fillna(0).to_numpy(float)
                    for field, prior in [('PACE', old_pace), ('OPP_PTS', old_opp)]:
                        if field in t:
                            t[field] = blend_rates(pd.to_numeric(t[field], errors='coerce').to_numpy(float),
                                                   pd.to_numeric(prior, errors='coerce').to_numpy(float),
                                                   current_games, prior_games=12.)
            except Exception:
                pass
            by_id = tp.set_index("TEAM_ID")
            for c in ("PTS", "REB", "AST"):
                old = t["TEAM_ID"].map(by_id[c])
                games = pd.to_numeric(t["GP"], errors="coerce").fillna(0)
                current = pd.to_numeric(t[c], errors="coerce")
                t[c] = blend_rates(current.to_numpy(float), old.to_numpy(float), games.to_numpy(float), prior_games=12.)
        if not pp.empty:
            by_id = pp.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")
            matched = p['PLAYER_ID'].isin(by_id.index)
            gp = pd.to_numeric(p['GP'], errors='coerce').fillna(0).clip(lower=0)
            p['HISTORICAL_PRIOR_USED'] = matched
            p['HISTORICAL_PRIOR_SEASON'] = np.where(matched, prior_season, '')
            p['HISTORICAL_CURRENT_WEIGHT'] = np.where(matched, gp / (gp + 18.), 1.)
            p['HISTORICAL_MINUTES_WEIGHT'] = np.where(matched, gp / (gp + 5.), 1.)
            # Preserve current-team roster membership. Blend playing time too:
            # two games should not overwrite a full previous season, but changes
            # to coaching and roles matter faster than shooting efficiency.
            old_min = pd.to_numeric(p['PLAYER_ID'].map(by_id['MIN']), errors='coerce')
            new_min = pd.to_numeric(p['MIN'], errors='coerce')
            old_team = p['PLAYER_ID'].map(by_id['TEAM_ABBREVIATION']) if 'TEAM_ABBREVIATION' in by_id else pd.Series('', index=p.index)
            changed_team = old_team.notna() & (old_team.astype(str) != p['TEAM_ABBREVIATION'].astype(str))
            # A player's prior-team minutes are a poor estimate of his role
            # after a trade. Trust the observed new-team minutes instead.
            good = matched & ~changed_team & (old_min > 0) & (new_min > 0)
            minute_w = gp / (gp + 5.)
            p['MIN'] = np.where(good, np.clip(minute_w * new_min + (1-minute_w) * old_min, 0, 42), new_min)
            for c in ("PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGA", "FGM", "FG3A", "FTA", "FTM"):
                if c not in pp or c not in p:
                    continue
                old = p["PLAYER_ID"].map(by_id[c])
                games = pd.to_numeric(p["GP"], errors="coerce").fillna(0)
                current = pd.to_numeric(p[c], errors="coerce")
                p[c] = blend_rates(current.to_numpy(float), old.to_numpy(float), games.to_numpy(float), prior_games=18.)
    except Exception as exc:
        historical_errors.append(f'{type(exc).__name__}: {str(exc)[:160]}')
    p = reconcile_shooting_lines(p)
    if 'FG3A' in p and 'FG3M' in p and 'FG3_PCT' in p:
        attempts = pd.to_numeric(p['FG3A'], errors='coerce').fillna(0)
        made = pd.to_numeric(p['FG3M'], errors='coerce').fillna(0)
        p['FG3_PCT'] = np.where(attempts > 0, np.clip(made / attempts.clip(lower=1e-8), 0, 1), 0)
    p['HISTORICAL_WARNING'] = '; '.join(historical_errors)
    t['HISTORICAL_WARNING'] = '; '.join(historical_errors)
    return t, p


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {"TEAM_NAME": ["TEAM_NAME", "team", "TeamName"],
               "TEAM_ABBREVIATION": ["TEAM_ABBREVIATION", "abbr", "TEAM_ABBREV"],
               "PTS": ["PTS", "PPG"], "OPP_PTS": ["OPP_PTS", "OPP_PPG"],
               "PACE": ["PACE", "E_PACE"]}
    for col, choices in aliases.items():
        if col not in df:
            for c in choices:
                if c in df:
                    df[col] = df[c]
                    break
    if "TEAM_ABBREVIATION" not in df and "TEAM_ID" in df:
        # LeagueDashTeamStats usually returns TEAM_ID and TEAM_NAME, not abbreviations.
        # Use nba_api's shipped static team dictionary; avoid guessing by initial letters.
        try:
            from nba_api.stats.static import teams as nba_teams
            abbr_by_id = {int(x["id"]): x["abbreviation"] for x in nba_teams.get_teams()}
            df["TEAM_ABBREVIATION"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").map(abbr_by_id)
        except (ImportError, ValueError, KeyError):
            pass
    if "TEAM_ABBREVIATION" not in df or "TEAM_NAME" not in df:
        raise ValueError("Team data requires TEAM_NAME and TEAM_ABBREVIATION (or NBA TEAM_ID)")
    for col, default in [("PTS", 112), ("OPP_PTS", 112), ("PACE", 99)]:
        if col not in df:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    for col in ["TEAM_NAME", "TEAM_ABBREVIATION"]:
        df[col] = df[col].astype(str)
    return df.drop_duplicates("TEAM_ABBREVIATION").reset_index(drop=True)


def normalize_players(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {"PLAYER_NAME": ["PLAYER_NAME", "player", "Player"],
               "TEAM_ABBREVIATION": ["TEAM_ABBREVIATION", "TEAM_ABBREV", "team"],
               "PTS": ["PTS", "PPG"], "REB": ["REB", "RPG"], "AST": ["AST", "APG"], "MIN": ["MIN", "MPG"]}
    for col, choices in aliases.items():
        if col not in df:
            for c in choices:
                if c in df:
                    df[col] = df[c]
                    break
    if "PLAYER_NAME" not in df or "TEAM_ABBREVIATION" not in df:
        raise ValueError("Player data requires PLAYER_NAME and TEAM_ABBREVIATION")
    defaults = {"PTS": 6.0, "REB": 2.4, "AST": 1.4, "MIN": 15.0, "FG3M": .6,
                "STL": .6, "BLK": .35, "TOV": 1.0, "FGA": 5.0, "FG3A": 1.8, "FTA": 1.6, "FTM": 1.23, "GP": 5}
    for c, default in defaults.items():
        if c not in df:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default).clip(lower=0)
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str).str.upper()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str)
    return df.drop_duplicates(["TEAM_ABBREVIATION", "PLAYER_NAME"]).reset_index(drop=True)


def demo_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic test data. Never described as current NBA athletes or used as fallback silently."""
    teams = [("Boston Celtics", "BOS", 117, 110, 99), ("New York Knicks", "NYK", 114, 111, 97),
             ("Oklahoma City Thunder", "OKC", 120, 108, 100), ("Denver Nuggets", "DEN", 116, 113, 97)]
    td = pd.DataFrame(teams, columns=["TEAM_NAME", "TEAM_ABBREVIATION", "PTS", "OPP_PTS", "PACE"])
    rows = []
    positions = [("Lead guard", 26, 5, 8, 35, 3.1), ("Primary wing", 22, 6, 4, 34, 2.5),
                 ("Center", 18, 11, 3, 32, .4), ("Starter", 14, 4, 3, 29, 1.8),
                 ("Starter", 12, 5, 2, 28, 1.3), ("Sixth", 11, 3, 3, 25, 1.5),
                 ("Reserve", 7, 3, 1.5, 19, .7), ("Reserve", 5, 2, 1, 14, .5),
                 ("Depth", 4, 2, 1, 12, .3), ("Depth", 2, 1, .5, 8, .2)]
    for _, abbr, _, _, _ in teams:
        for idx, (_, pts, reb, ast, mins, fg3m) in enumerate(positions, 1):
            rows.append({"PLAYER_NAME": f"{abbr} Sample {idx}", "TEAM_ABBREVIATION": abbr,
                         "PTS": pts, "REB": reb, "AST": ast, "MIN": mins, "FG3M": fg3m,
                         "STL": .7, "BLK": .4, "TOV": 1.4, "FGA": pts / 2.1,
                         "FG3A": fg3m * 2.7, "FTA": pts / 4.2, "GP": 30})
    return td, pd.DataFrame(rows)


def normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", name.replace("jr", "").replace("iii", ""))


def injury_status(text: str) -> str:
    s = str(text).lower()
    if any(x in s for x in ("out", "suspended", "injured reserve")):
        return "Out"
    if "doubt" in s:
        return "Doubtful"
    if "question" in s or "day-to-day" in s or "day to day" in s:
        return "Questionable"
    if "probable" in s:
        return "Probable"
    return "Unclear"


def actual_boxscore(game_id: str) -> pd.DataFrame:
    payload = get_json(f"{ESPN}/summary", {"event": str(game_id)})
    rows = []
    for team in payload.get("boxscore", {}).get("players", []):
        abbr = team.get("team", {}).get("abbreviation", "")
        for category in team.get("statistics", []):
            labels = [str(x).upper() for x in category.get("labels", [])]
            for athlete in category.get("athletes", []):
                a = athlete.get("athlete", {})
                stats = athlete.get("stats", [])
                if not isinstance(stats, list) or not stats:
                    continue
                row = {"PLAYER_NAME": a.get("displayName", ""), "TEAM_ABBREVIATION": abbr}
                for key in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "3PM"):
                    src_key = {"TOV": "TO", "3PM": "3PT"}.get(key, key)
                    if src_key not in labels and key == "3PM":
                        src_key = "3PM"
                    if src_key in labels:
                        ix = labels.index(src_key)
                        if ix < len(stats):
                            try:
                                raw = str(stats[ix]); raw = raw.split(":")[0] if key == "MIN" else raw
                                if key == "3PM" and "-" in raw: raw = raw.split("-")[0]
                                row[key] = float(raw)
                            except (TypeError, ValueError):
                                pass
                if "PTS" in row:
                    rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(["PLAYER_NAME", "TEAM_ABBREVIATION"]) if rows else pd.DataFrame()


def current_roster(team_id: str) -> list[dict]:
    """ESPN's current team roster; includes players with 0 current-season NBA games.

    Injury fields are advisory, NOT equivalent to a confirmed NBA availability report.
    """
    payload = get_json(f"{ESPN}/teams/{team_id}/roster")
    out = []
    for a in payload.get("athletes", []):
        name = a.get("displayName") or a.get("fullName")
        if not name:
            continue
        statuses = []
        for injury in a.get("injuries", []):
            if isinstance(injury, dict) and injury.get("status"):
                statuses.append(str(injury["status"]))
        out.append({"player": name, "id": str(a.get("id", "")),
                    "position": a.get("position", {}).get("abbreviation", ""),
                    "injury": statuses[0] if statuses else "",
                    "status": a.get("status", {}).get("name", "")})
    if not out:
        raise ValueError("ESPN returned no current roster athletes")
    return out


def previous_season_players(season: str) -> pd.DataFrame:
    """Prior-year player per-game stats for injured players with zero current GP."""
    from nba_api.stats.endpoints import leaguedashplayerstats
    return _league_stat(leaguedashplayerstats.LeagueDashPlayerStats, season, 0)


def reconcile_matchup_rosters(players: pd.DataFrame, roster_by_team: dict[str, list[dict]],
                              past: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Use current ESPN membership, current stats first, previous-season fallback.

    Does not invent estimates for rookies whose current AND previous NBA data are
    both absent. Reports excluded players instead.
    """
    main = normalize_players(players)
    prior = normalize_players(past) if past is not None and not past.empty else pd.DataFrame()
    current_lookup = {normalize_name(n): r for n, r in main.set_index("PLAYER_NAME").iterrows()}
    prior_lookup = ({normalize_name(n): r for n, r in prior.set_index("PLAYER_NAME").iterrows()}
                    if len(prior) else {})
    reconciled = []
    missing = []
    for team, roster in roster_by_team.items():
        for info in roster:
            name = info["player"]
            name_key = normalize_name(name)
            match = current_lookup.get(name_key)
            origin = "Current NBA statistics"
            if match is None:
                match = prior_lookup.get(name_key)
                origin = "Previous season baseline — uncertain role"
            if match is None:
                missing.append(f"{name} ({team})")
                continue
            row = match.to_dict()
            row["PLAYER_NAME"] = name
            row["TEAM_ABBREVIATION"] = team
            row["BASELINE_SOURCE"] = origin
            reconciled.append(row)
    if not reconciled:
        raise ValueError("No ESPN roster players matched historical NBA statistics")
    out = pd.DataFrame(reconciled)
    # The simulator simulates the highest-minute 13 eligible rostered players.
    return normalize_players(out), missing
