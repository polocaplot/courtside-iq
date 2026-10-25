"""Custom fantasy points scoring, using the *same* simulated box scores as matchup pages."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FantasyScoring:
    PTS: float = 1.0
    REB: float = 2.0
    AST: float = 2.0
    STL: float = 3.0
    BLK: float = 3.0
    TOV: float = -1.0
    THREE_PM: float = 0.0

    def weights(self) -> dict[str, float]:
        data = asdict(self)
        data["3PM"] = data.pop("THREE_PM")
        return data


def fantasy_values(stat_samples: dict[str, np.ndarray], scoring: FantasyScoring) -> np.ndarray:
    scores = np.zeros_like(stat_samples["PTS"], dtype=np.float32)
    for stat, weight in scoring.weights().items():
        scores += stat_samples[stat] * weight
    return scores


def leaderboard(result: dict, scoring: FantasyScoring, threshold: float = 40.0) -> pd.DataFrame:
    rows = []
    for _, row in result["box"].iterrows():
        sim = fantasy_values(result["samples"][row["ID"]], scoring)
        rows.append({
            "Player": row["Player"], "Team": row["Team"], "Status": row["Status"],
            "Fantasy AVG": round(float(sim.mean()), 2),
            "Fantasy P10": round(float(np.quantile(sim, .1)), 2),
            "Fantasy P50": round(float(np.quantile(sim, .5)), 2),
            "Fantasy P90": round(float(np.quantile(sim, .9)), 2),
            f"P(≥{threshold:g})": round(float(np.mean(sim >= threshold)) * 100, 1),
            "Boom %": round(float(np.mean(sim >= 1.25 * max(8., sim.mean()))) * 100, 1),
            "Bust %": round(float(np.mean(sim <= .70 * max(8., sim.mean()))) * 100, 1),
            "MIN": row["MIN"], "PTS": row["PTS"], "REB": row["REB"],
            "AST": row["AST"], "STL": row["STL"], "BLK": row["BLK"], "TOV": row["TOV"],
            "ID": row["ID"],
        })
    return pd.DataFrame(rows).sort_values("Fantasy AVG", ascending=False).reset_index(drop=True)


def score_boxscore(df: pd.DataFrame, scoring: FantasyScoring) -> pd.Series:
    return sum(pd.to_numeric(df.get(s, pd.Series(0, index=df.index)), errors="coerce").fillna(0) * w
               for s, w in scoring.weights().items())


def explain_fantasy_line(row: pd.Series, scoring: FantasyScoring) -> pd.DataFrame:
    return pd.DataFrame([{"Stat": stat, "Projected": float(row[stat]), "Multiplier": weight,
                          "Contribution": round(float(row[stat]) * weight, 2)}
                         for stat, weight in scoring.weights().items() if abs(weight) > 1e-12])
