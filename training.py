"""COURTSIDE IQ walk-forward NBA forecasting engine.

- Historical NBA LeagueGameLog records ONLY; never fabricate game logs.
- Rolling features always use preceding games; future results never leak in.
- Time-ordered out-of-sample validation, with an equal-information rolling baseline.
- Model used only if its own held-out MAE beats the baseline by >= 1%.
- Every model stores its performance, cutoff, and data vintage in manifest.json.

This is an experimental, auditable ML projection system, not Vegas parity.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import normalize_name

STATS = ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M")
MAXIMUMS = {"MIN": 45, "PTS": 70, "REB": 30, "AST": 25, "STL": 12,
            "BLK": 12, "TOV": 14, "FG3M": 15}
DEFAULT_FOLDER = Path.home() / ".courtsideiq" / "trained"
YEARS = ("2023-24", "2024-25", "2025-26")


def fetch_historical(seasons: tuple[str, ...], cache_dir: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Fetch and disk-cache complete regular-season player game logs, sequentially.

    If NBA.com blocks a request, use cached files or import exported historical
    player game logs. Never invent historical observations.
    """
    from nba_api.stats.endpoints import leaguegamelog
    cache_dir = Path(cache_dir or DEFAULT_FOLDER / "game_logs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for season in seasons:
        dest = cache_dir / f"{season.replace('-', '_')}_player_games.csv"
        if dest.exists() and not refresh:
            data = pd.read_csv(dest, dtype={"GAME_ID": str})
        else:
            data = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation="P", league_id="00", timeout=45,
            ).get_data_frames()[0]
            if data.empty:
                raise ValueError(f"NBA.com returned no player game logs for {season}")
            data.to_csv(dest, index=False)
        data["SEASON"] = season
        records.append(data)
    return pd.concat(records, ignore_index=True)


def _numeric_min(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # Accept '34:21' while handling numeric NBA.com minutes.
    minutes = pd.to_numeric(s.str.split(':').str[0], errors='coerce')
    return minutes


def standardize_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize NBA game-log table; duplicates fail closed."""
    df = logs.copy()
    if "PLAYER_ID" not in df and "Player_ID" in df:
        df = df.rename(columns={"Player_ID": "PLAYER_ID"})
    must = {"GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
            "MATCHUP", *STATS}
    if "PLAYER_NAME" not in df and "PLAYER" in df:
        df = df.rename(columns={"PLAYER": "PLAYER_NAME"})
    missing = must.difference(df.columns)
    if missing:
        raise ValueError("Historical game logs missing columns: " + ", ".join(sorted(missing)))
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce", utc=True)
    df = df.dropna(subset=["GAME_DATE", "PLAYER_ID", "GAME_ID"])
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str).str.upper()
    df["HOME"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(float)
    df["OPPONENT"] = df["MATCHUP"].astype(str).str.extract(r"(?:vs\.\s*|@\s*)([A-Z]{2,3})", expand=False)
    for stat in STATS:
        df[stat] = _numeric_min(df[stat]) if stat == "MIN" else pd.to_numeric(df[stat], errors="coerce")
    df = df.dropna(subset=["MIN", "PTS"])
    if df.duplicated(["PLAYER_ID", "GAME_ID"]).any():
        raise ValueError("Duplicate player/game entries in historical logs")
    df = df[df["MIN"] > 0].sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No historical player games with nonzero minutes")
    return df


def _build_team_environment(logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Team totals are computed from historical box scores; historical defense is
    # opponent points conceded before today's game, not today's final score.
    tg = (logs.groupby(["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"], as_index=False)
          .agg(PTS=("PTS", "sum"), FGA=("FGA", "sum") if "FGA" in logs else ("PTS", "sum"),
               OREB=("OREB", "sum") if "OREB" in logs else ("REB", "sum"),
               FTA=("FTA", "sum") if "FTA" in logs else ("PTS", "sum"),
               TOV=("TOV", "sum")))
    tg["POSS"] = tg["FGA"] + .44 * tg["FTA"] - tg["OREB"] + tg["TOV"]
    # Join opposing score from same game to calculate historical points allowed.
    pair = tg[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP_TEAM", "PTS": "OPP_PTS"})
    tg = tg.merge(pair, on="GAME_ID", how="left")
    tg = tg[tg["TEAM_ABBREVIATION"] != tg["OPP_TEAM"]].drop_duplicates(
        ["GAME_ID", "TEAM_ABBREVIATION"])
    tg = tg.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
    for raw, feature in (("PTS", "OFF_P10"), ("OPP_PTS", "DEF_P10"), ("POSS", "PACE_P10")):
        tg[feature] = tg.groupby("TEAM_ABBREVIATION", sort=False)[raw].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    return tg, tg[["GAME_ID", "TEAM_ABBREVIATION", "OFF_P10", "DEF_P10", "PACE_P10"]].copy()


def build_features(logs: pd.DataFrame, *, future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build pregame feature rows, incl. future games, without peeking at game result.

    Returned frame carries target columns for historical games; future targets NaN.
    """
    base = standardize_logs(logs)
    team_history, team_features = _build_team_environment(base)
    if future is not None and not future.empty:
        ff = future.copy()
        must = {"GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "HOME"}
        if missing := must.difference(ff.columns):
            raise ValueError("Future records missing: " + ", ".join(sorted(missing)))
        ff["GAME_DATE"] = pd.to_datetime(ff["GAME_DATE"], utc=True)
        ff["GAME_ID"] = ff.get("GAME_ID", pd.Series(["FUTURE"]*len(ff), index=ff.index)).astype(str)
        ff["MATCHUP"] = ff["TEAM_ABBREVIATION"].astype(str) + " vs. " + ff["OPPONENT"].astype(str)
        for c in STATS:
            ff[c] = np.nan
        ff["_future"] = True
        base["_future"] = False
        joined = pd.concat([base, ff], ignore_index=True)
    else:
        joined = base.copy()
        joined["_future"] = False
    joined = joined.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"], kind="stable").reset_index(drop=True)
    joined["PLAYER_ID"] = joined["PLAYER_ID"].astype(str)
    groups = joined.groupby("PLAYER_ID", sort=False)
    joined["GAMES_BEFORE"] = groups.cumcount().astype(float)
    last_date = groups["GAME_DATE"].shift(1)
    joined["DAYS_REST"] = ((joined["GAME_DATE"] - last_date).dt.total_seconds() / 86400).clip(0, 40).fillna(7)
    joined["B2B"] = (joined["DAYS_REST"] <= 1.4).astype(float)
    joined["MONTH"] = joined["GAME_DATE"].dt.month.astype(float)
    feature_cols = ["GAMES_BEFORE", "DAYS_REST", "B2B", "MONTH", "HOME"]
    for stat in STATS:
        for window in (3, 10, 20):
            cname = f"{stat}_L{window}"
            joined[cname] = groups[stat].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean())
            feature_cols.append(cname)
        cname = f"{stat}_STD10"
        joined[cname] = groups[stat].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())
        feature_cols.append(cname)
    # Current game's team and opponent rolling ratings from past completed games.
    # The combined source table stores P10 *at* a historical game, excluding it.
    joined = joined.merge(team_features.rename(columns={"TEAM_ABBREVIATION": "TEAM_KEY"}),
                          left_on=["GAME_ID", "TEAM_ABBREVIATION"],
                          right_on=["GAME_ID", "TEAM_KEY"], how="left")
    opp_features = team_features.rename(columns={"TEAM_ABBREVIATION": "OPP_KEY",
        "OFF_P10": "OPP_OFF_P10", "DEF_P10": "OPP_DEF_P10", "PACE_P10": "OPP_PACE_P10"})
    joined = joined.merge(opp_features, left_on=["GAME_ID", "OPPONENT"],
                          right_on=["GAME_ID", "OPP_KEY"], how="left")
    if future is not None and not future.empty:
        recent = team_history.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").tail(10)
        latest = (recent.groupby("TEAM_ABBREVIATION", as_index=True)
                  .agg(OFF_P10=("PTS", "mean"), DEF_P10=("OPP_PTS", "mean"),
                       PACE_P10=("POSS", "mean")))
        future_mask = joined["_future"]
        for key, team_col, field in (("OFF_P10", "TEAM_ABBREVIATION", "OFF_P10"),
                                      ("DEF_P10", "TEAM_ABBREVIATION", "DEF_P10"),
                                      ("PACE_P10", "TEAM_ABBREVIATION", "PACE_P10"),
                                      ("OPP_OFF_P10", "OPPONENT", "OFF_P10"),
                                      ("OPP_DEF_P10", "OPPONENT", "DEF_P10"),
                                      ("OPP_PACE_P10", "OPPONENT", "PACE_P10")):
            joined.loc[future_mask, key] = joined.loc[future_mask, team_col].map(latest[field])
    feature_cols += ["OFF_P10", "DEF_P10", "PACE_P10", "OPP_OFF_P10", "OPP_DEF_P10", "OPP_PACE_P10"]
    joined.attrs["features"] = feature_cols
    return joined


def _metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return {"mae": round(float(np.mean(np.abs(actual - predicted))), 4),
            "rmse": round(float(np.sqrt(np.mean((actual - predicted)**2))), 4)}


def _fantasy(df: pd.DataFrame) -> np.ndarray:
    return (df["PTS"] + 2*df["REB"] + 2*df["AST"] + 3*df["STL"] + 3*df["BLK"] - df["TOV"]).to_numpy(float)


def train(logs: pd.DataFrame, folder: Path = DEFAULT_FOLDER, min_games: int = 600,
          max_rows: int | None = None) -> dict:
    """Train strictly chronological holdout models, refit after evaluation, store metadata.

    No random train/test split. One model/stat. Avoid saving a model if a
    holdout has too few games or insufficient historical sample for players.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    folder = Path(folder)
    data = build_features(logs)
    features = data.attrs["features"]
    data = data[(data["GAMES_BEFORE"] >= 3) & data["PTS_L10"].notna()]
    if max_rows and len(data) > max_rows:
        data = data.sort_values("GAME_DATE").tail(max_rows)
    if len(data) < min_games:
        raise ValueError(f"Need at least {min_games} usable historical games; found {len(data)}")
    unique_days = np.array(sorted(data["GAME_DATE"].unique()))
    if len(unique_days) < 25:
        raise ValueError("Training needs >=25 distinct game dates for a chronological test")
    # Hold out the final 20% of *dates*, not a random 20% of players.
    cutoff = pd.Timestamp(unique_days[int(len(unique_days) * .8)])
    historical = data[data["GAME_DATE"] < cutoff]
    holdout = data[data["GAME_DATE"] >= cutoff]
    if len(historical) < 400 or len(holdout) < 100:
        raise ValueError("Insufficient earlier and held-out games; fetch additional seasons")
    models = {}
    baseline_preds = {}
    model_preds = {}
    report = []
    for stat in STATS:
        train_set = historical.dropna(subset=[stat]); test_set = holdout.dropna(subset=[stat])
        if len(train_set) < 250 or len(test_set) < 50:
            continue
        mdl = HistGradientBoostingRegressor(max_iter=135, learning_rate=.055,
            max_leaf_nodes=16, min_samples_leaf=45, l2_regularization=7,
            early_stopping=False, random_state=17)
        mdl.fit(train_set[features], train_set[stat])
        pred = np.clip(mdl.predict(test_set[features]), 0, MAXIMUMS[stat])
        prior = np.clip(test_set[f"{stat}_L10"].fillna(train_set[stat].mean()).to_numpy(float), 0, MAXIMUMS[stat])
        actual = test_set[stat].to_numpy(float)
        pm, bm = _metrics(actual, pred), _metrics(actual, prior)
        improvement = (bm["mae"] - pm["mae"]) / max(bm["mae"], .0001)
        # Promote only truly out-of-sample improvements. Otherwise the rolling
        # baseline remains the safer live projection for that stat.
        blend = .65 if improvement >= .01 else 0.
        model_preds[stat] = pd.Series(pred, index=test_set.index)
        baseline_preds[stat] = pd.Series(prior, index=test_set.index)
        report.append({"stat": stat, "model_mae": pm["mae"], "baseline_mae": bm["mae"],
                       "model_rmse": pm["rmse"], "baseline_rmse": bm["rmse"],
                       "improvement_pct": round(100*improvement, 2),
                       "live_model_weight": blend, "holdout_n": len(test_set),
                       "residual_sd": round(float(np.std(actual - (blend*pred + (1-blend)*prior))), 3)})
        # Fit production model on all known games AFTER holdout statistics
        # are frozen. Holdout report remains strictly untrained-on outcomes.
        mdl.fit(data[features], data[stat])
        models[stat] = mdl
    if not models:
        raise ValueError("Historical dataset too sparse to train any models")
    report_df = pd.DataFrame(report)
    fantasy_test = holdout.copy()
    # All evaluated rows must be represented for the aggregate fantasy metric.
    for stat in STATS:
        if stat in model_preds:
            r = report_df.set_index("stat").loc[stat]
            weighted = (r["live_model_weight"] * model_preds[stat] +
                        (1-r["live_model_weight"]) * baseline_preds[stat])
            fantasy_test[f"PRED_{stat}"] = weighted
            fantasy_test[f"BASE_{stat}"] = baseline_preds[stat]
    fp_stats = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
    fantasy_metrics = {}
    if all(f"PRED_{s}" in fantasy_test for s in fp_stats):
        actual_df = fantasy_test[list(fp_stats)].dropna()
        indexes = actual_df.index
        pred_df = fantasy_test.loc[indexes, [f"PRED_{s}" for s in fp_stats]].copy()
        pred_df.columns = list(fp_stats)
        base_df = fantasy_test.loc[indexes, [f"BASE_{s}" for s in fp_stats]].copy()
        base_df.columns = list(fp_stats)
        fantasy_metrics = {"model": _metrics(_fantasy(actual_df), _fantasy(pred_df)),
                           "baseline": _metrics(_fantasy(actual_df), _fantasy(base_df)),
                           "observations": len(indexes)}
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, folder / "models.joblib")
    # Save complete historical logs to support player-specific game-day predictions.
    canonical = standardize_logs(logs)
    canonical.to_csv(folder / "history.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "training_games": len(historical), "holdout_games": len(holdout),
                "all_games": len(data), "holdout_from": str(cutoff.date()),
                "latest_game": str(canonical["GAME_DATE"].max().date()),
                "features": features, "results": report, "fantasy": fantasy_metrics,
                "method": "time-ordered holdout; pregame rolling-only features; independent MAE guardrail",
                "seasons": sorted(canonical["SEASON"].dropna().astype(str).unique().tolist()) if "SEASON" in canonical else []}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(folder: Path = DEFAULT_FOLDER) -> tuple[dict, dict, pd.DataFrame]:
    import joblib
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    models = joblib.load(folder / "models.joblib")
    logs = pd.read_csv(folder / "history.csv", dtype={"GAME_ID": str, "PLAYER_ID": str})
    return manifest, models, logs


def forecast(players: pd.DataFrame, home: str, away: str, tipoff: str,
             folder: Path = DEFAULT_FOLDER) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply holdout-guarded ML to current roster rows without future leakage.

    Current roster (not last year's historical team) dictates team membership.
    Athletes not matched to any historical ID retain the season rate baseline.
    Only overrides fields when player has >=3 historical games.
    """
    manifest, models, logs = load(folder)
    base = standardize_logs(logs)
    use = players.copy()
    for stat in STATS:
        if stat in use.columns:
            use[stat] = pd.to_numeric(use[stat], errors="coerce").astype(float)
    date = pd.to_datetime(tipoff, utc=True)
    if date <= base["GAME_DATE"].max():
        # Avoid entering a historical game with information from its future.
        raise ValueError("Trained history contains games on/after this matchup. Retrain with only earlier games for a historical replay.")
    # Validate roster membership from the selected fixture and current feed.
    latest = base.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")
    latest["name_key"] = latest["PLAYER_NAME"].map(normalize_name)
    lookup = latest.drop_duplicates("name_key", keep="last").set_index("name_key")
    upcoming = []
    for _, p in use[use["TEAM_ABBREVIATION"].isin([home, away])].iterrows():
        key = normalize_name(p["PLAYER_NAME"])
        if key not in lookup.index:
            continue
        t = str(p["TEAM_ABBREVIATION"])
        upcoming.append({"GAME_ID": f"FUTURE_{key}", "GAME_DATE": date,
                         "PLAYER_ID": str(lookup.loc[key, "PLAYER_ID"]),
                         "PLAYER_NAME": p["PLAYER_NAME"], "TEAM_ABBREVIATION": t,
                         "OPPONENT": away if t == home else home, "HOME": 1. if t == home else 0.})
    if not upcoming:
        raise ValueError("No rostered players matched archived player game logs")
    upcoming_df = pd.DataFrame(upcoming)
    features = build_features(logs, future=upcoming_df)
    future_rows = features[features["_future"]].copy()
    future_rows = future_rows.set_index("PLAYER_NAME")
    indexes = []
    audits = []
    stat_report = {r["stat"]: r for r in manifest["results"]}
    for idx, p in use.iterrows():
        name = str(p["PLAYER_NAME"])
        if name not in future_rows.index:
            audits.append({"Player": name, "Team": p["TEAM_ABBREVIATION"], "Source": "Season rate fallback", "History games": 0, "Reason": "No historical ID match"})
            continue
        fr = future_rows.loc[name]
        if isinstance(fr, pd.DataFrame):
            fr = fr.iloc[-1]
        games = int(fr["GAMES_BEFORE"])
        if games < 3:
            audits.append({"Player": name, "Team": p["TEAM_ABBREVIATION"], "Source": "Season rate fallback", "History games": games, "Reason": "Insufficient player history"})
            continue
        row = fr[manifest["features"]].to_frame().T
        used = []
        for stat in STATS:
            if stat not in models:
                continue
            report = stat_report[stat]
            base_stat = float(p.get(stat, 0.))
            if stat == "FG3M":
                base_stat = float(p.get("FG3M", .5))
            raw_ml = float(models[stat].predict(row)[0])
            rolling = fr.get(f"{stat}_L10", np.nan)
            # Weighted ML / historical recent form, then mild blend with current
            # league season-rate baseline for data timeliness and role changes.
            prior = base_stat if pd.isna(rolling) else float(rolling)
            w = float(report["live_model_weight"])
            trained_value = w*raw_ml + (1-w)*prior
            blended = .72*trained_value + .28*base_stat
            result = float(np.clip(blended, 0, MAXIMUMS[stat]))
            use.loc[idx, stat] = result
            if stat == "MIN":
                use.loc[idx, "MIN"] = result
            if w > 0:
                used.append(stat)
        audits.append({"Player": name, "Team": p["TEAM_ABBREVIATION"],
                       "Source": "Historical rolling + holdout-guarded ML" if used else "Historical rolling (ML not validated)",
                       "History games": games, "Reason": "Learned stats: " + ", ".join(used) if used else "Baseline performed as well or better"})
        indexes.append(idx)
    return use, pd.DataFrame(audits), manifest
