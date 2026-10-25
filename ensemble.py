"""Optional, interpretable pregame prior + role forecast.

Does not replace the original possession or statistical simulator. The layer only
adjusts *pregame inputs*. It never uses actual results from the target game.
Weights are heuristics pending walk-forward validation, NOT trained sportsbook weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from data_sources import normalize_name, normalize_players

RATE_COLS = ('PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'FGA', 'FG3A', 'FTA', 'FTM')
OUT = {'Out', 'Doubtful'}


def prepare_prior(prior: pd.DataFrame | None) -> dict[str, dict]:
    if prior is None or prior.empty:
        return {}
    clean = normalize_players(prior)
    return {normalize_name(r['PLAYER_NAME']): r.to_dict() for _, r in clean.iterrows()}


def blended_inputs(current: pd.DataFrame, prior: pd.DataFrame | None = None,
                   *, prior_games: float = 16., minutes_prior_games: float = 10.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shrink current-season per-minute production toward prior-year player rates.

    Returns adjusted player inputs and an audit. Unknown players are left unchanged.
    A previous-season-only fallback remains previous-season data and is flagged.
    """
    if prior_games <= 0 or minutes_prior_games <= 0:
        raise ValueError('Prior sample weights must be positive')
    df = normalize_players(current)
    lookup = prepare_prior(prior)
    for col in ("MIN", *RATE_COLS):
        df[col] = df[col].astype(float)
    audit = []
    for i, row in df.iterrows():
        source = str(row.get('BASELINE_SOURCE', 'Current NBA statistics'))
        old = lookup.get(normalize_name(row['PLAYER_NAME']))
        gp = float(np.clip(row.get('GP', 0), 0, 82))
        if old is None or 'Previous season' in source:
            audit.append({'Player': row['PLAYER_NAME'], 'Team': row['TEAM_ABBREVIATION'],
                          'Current GP': gp, 'Prior matched': bool(old), 'Current weight': 1.,
                          'Note': 'Previous-year-only baseline' if 'Previous season' in source else 'No prior season match'})
            continue
        old_min = float(old['MIN'])
        now_min = float(row['MIN'])
        if old_min <= 0 or now_min <= 0:
            audit.append({'Player': row['PLAYER_NAME'], 'Team': row['TEAM_ABBREVIATION'],
                          'Current GP': gp, 'Prior matched': True, 'Current weight': 1.,
                          'Note': 'Invalid minutes: no blend'})
            continue
        w = gp / (gp + prior_games)
        mw = gp / (gp + minutes_prior_games)
        # Recent team and coaching usage changes deserve more weight in minutes
        # than in per-minute shooting/production rates.
        new_min = float(np.clip(mw * now_min + (1 - mw) * old_min, 0, 42))
        df.at[i, 'MIN'] = new_min
        for stat in RATE_COLS:
            curr_rate = float(row[stat]) / now_min
            old_rate = float(old[stat]) / old_min
            rate = w * curr_rate + (1 - w) * old_rate
            df.at[i, stat] = max(0., rate * new_min)
        audit.append({'Player': row['PLAYER_NAME'], 'Team': row['TEAM_ABBREVIATION'],
                      'Current GP': gp, 'Prior matched': True, 'Current weight': round(w, 3),
                      'Note': 'Per-minute prior blend + minutes blend'})
    return df, pd.DataFrame(audit)


def role_usage_adjustments(players: pd.DataFrame, injuries: dict[str, str],
                           existing: dict[str, float] | None = None,
                           *, maximum_bonus: float = .20) -> tuple[dict[str, float], pd.DataFrame]:
    """Conservative missing-teammate opportunity redistribution.

    Assumes a limited share of missing shot creation can be absorbed by active
    players; all other effects remain with the simulator's existing engine.
    Manual usage overrides are NEVER replaced.
    """
    if not 0 <= maximum_bonus <= .4:
        raise ValueError('maximum_bonus out of range')
    df = normalize_players(players)
    overrides = dict(existing or {})
    rows = []
    for team, roster in df.groupby('TEAM_ABBREVIATION'):
        available = roster[~roster['PLAYER_NAME'].map(lambda n: injuries.get(n, 'Healthy') in OUT)]
        missing = roster[roster['PLAYER_NAME'].map(lambda n: injuries.get(n, 'Healthy') in OUT)]
        lost_creation = float((missing['PTS'] + 1.5 * missing['AST']).sum())
        active_creation = float((available['PTS'] + 1.5 * available['AST']).sum())
        if active_creation <= 0 or lost_creation <= 0:
            continue
        bonus = float(np.clip(.18 * lost_creation / active_creation, 0, maximum_bonus))
        for _, p in available.iterrows():
            name = str(p['PLAYER_NAME'])
            if name in overrides:
                note = 'Manual override preserved'
            else:
                overrides[name] = 1. + bonus
                note = 'Missing-teammate opportunity proxy'
            rows.append({'Player': name, 'Team': team, 'Usage x': overrides[name], 'Reason': note})
    return overrides, pd.DataFrame(rows)


def prediction_deltas(base: dict, candidate: dict, scoring) -> pd.DataFrame:
    """Compare two independently simulated results without mixing their samples."""
    from fantasy import fantasy_values
    left = base['box'].set_index('ID')
    right = candidate['box'].set_index('ID')
    out = []
    for player_id in sorted(set(left.index) & set(right.index)):
        a, b = left.loc[player_id], right.loc[player_id]
        out.append({'Player': b['Player'], 'Team': b['Team'],
                    'Original MIN': float(a['MIN']), 'Smart MIN': float(b['MIN']),
                    'Original PTS': float(a['PTS']), 'Smart PTS': float(b['PTS']),
                    'Original fantasy': float(fantasy_values(base['samples'][player_id], scoring).mean()),
                    'Smart fantasy': float(fantasy_values(candidate['samples'][player_id], scoring).mean())})
    frame = pd.DataFrame(out)
    if not frame.empty:
        frame['Fantasy change'] = frame['Smart fantasy'] - frame['Original fantasy']
        frame = frame.sort_values('Fantasy change', key=lambda s: s.abs(), ascending=False)
    return frame
