"""Optional defender-by-zone observational data; not optical player tracking.

Expected shot-make modifiers are shrunk to conservative league zone priors.
Defender assignments in simulation are approximations; exact primary matchups,
shot contests and distance cannot be inferred from aggregate tables.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from shot_intelligence import ZONES, PRIOR_MAKE

COLS = ('TEAM_ABBREVIATION', 'DEFENDER_NAME', 'ZONE', 'FGA_CONTESTED', 'FGM_ALLOWED')


def validate_defenders(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError('Defender CSV missing: ' + ', '.join(sorted(missing)))
    d = df[list(COLS)].copy()
    for col in ('TEAM_ABBREVIATION', 'DEFENDER_NAME', 'ZONE'):
        d[col] = d[col].astype(str).str.strip()
    d['TEAM_ABBREVIATION'] = d['TEAM_ABBREVIATION'].str.upper()
    if not d['ZONE'].isin(ZONES).all():
        raise ValueError('Unknown defensive shot zone')
    for col in ('FGA_CONTESTED', 'FGM_ALLOWED'):
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not np.isfinite(d[col]).all() or (d[col] < 0).any():
            raise ValueError('Invalid defensive shot counts')
    if (d['FGM_ALLOWED'] > d['FGA_CONTESTED']).any():
        raise ValueError('Makes cannot exceed contested attempts')
    return d.groupby(['TEAM_ABBREVIATION', 'DEFENDER_NAME', 'ZONE'], as_index=False)[['FGA_CONTESTED', 'FGM_ALLOWED']].sum()


def defender_zone_adjustments(roster: pd.DataFrame, team: str,
                              observations: pd.DataFrame | None) -> np.ndarray:
    """Return [players, zones] additive make-probability modifiers.

    Conservative Bayesian shrinkage: 200 pseudo-attempts per zone and maximum
    4 percentage points in either direction. Missing observations yield zero.
    """
    out = np.zeros((len(roster), len(ZONES)), dtype=float)
    if observations is None or observations.empty:
        return out
    d = observations[observations['TEAM_ABBREVIATION'] == team]
    for i, name in enumerate(roster['PLAYER_NAME'].astype(str)):
        player = d[d['DEFENDER_NAME'].str.casefold() == name.casefold()]
        for j, zone in enumerate(ZONES):
            r = player[player['ZONE'] == zone]
            if r.empty:
                continue
            n = float(r['FGA_CONTESTED'].sum())
            m = float(r['FGM_ALLOWED'].sum())
            prior = PRIOR_MAKE[zone]
            rate = (m + 200 * prior) / (n + 200)
            out[i, j] = np.clip(rate - prior, -.04, .04)
    return out
