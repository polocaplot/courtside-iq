"""Shot-zone data ingest, empirical-Bayes smoothing and opponent adjustments.

Data is aggregated shot locations, not defender coordinates or possession tracking.
No provider claims are made when sources are unavailable.
"""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd

ZONES = ('Rim', 'Paint', 'Midrange', 'Corner 3', 'Above-break 3')
PRIOR_MAKE = {'Rim': .65, 'Paint': .43, 'Midrange': .42, 'Corner 3': .38, 'Above-break 3': .35}
PRIOR_SHARE = np.array([.31, .18, .13, .09, .29])


def zone_from_nba(basic: str, area: str = '', shot_type: str = '') -> str:
    basic, area, shot_type = str(basic).lower(), str(area).lower(), str(shot_type).lower()
    if 'corner 3' in basic or ('3pt' in shot_type and 'corner' in area):
        return 'Corner 3'
    if 'above the break' in basic or 'backcourt' in basic or '3pt' in shot_type:
        return 'Above-break 3'
    if 'restricted area' in basic:
        return 'Rim'
    if 'paint' in basic or 'non-ra' in basic:
        return 'Paint'
    return 'Midrange'


def aggregate_nba_shots(shots: pd.DataFrame, team_abbreviation: str) -> pd.DataFrame:
    required = {'PLAYER_NAME', 'SHOT_ZONE_BASIC', 'SHOT_MADE_FLAG'}
    if not required.issubset(shots.columns):
        raise ValueError('Shot chart is missing columns: ' + ', '.join(sorted(required - set(shots.columns))))
    d = shots.copy()
    d['ZONE'] = [zone_from_nba(r['SHOT_ZONE_BASIC'], r.get('SHOT_ZONE_AREA', ''), r.get('SHOT_TYPE', ''))
                 for _, r in d.iterrows()]
    d['FGM'] = pd.to_numeric(d['SHOT_MADE_FLAG'], errors='coerce').fillna(0).clip(0, 1)
    d['FGA'] = 1
    d['TEAM_ABBREVIATION'] = team_abbreviation.upper()
    return d.groupby(['TEAM_ABBREVIATION', 'PLAYER_NAME', 'ZONE'], as_index=False)[['FGA', 'FGM']].sum()


def validate_profiles(df: pd.DataFrame) -> pd.DataFrame:
    required = {'TEAM_ABBREVIATION', 'PLAYER_NAME', 'ZONE', 'FGA', 'FGM'}
    if not required.issubset(df.columns):
        raise ValueError('Shot profile CSV requires TEAM_ABBREVIATION, PLAYER_NAME, ZONE, FGA, FGM')
    d = df.copy()
    d['TEAM_ABBREVIATION'] = d['TEAM_ABBREVIATION'].astype(str).str.upper().str.strip()
    d['PLAYER_NAME'] = d['PLAYER_NAME'].astype(str).str.strip()
    d['ZONE'] = d['ZONE'].astype(str).str.strip()
    if not d['ZONE'].isin(ZONES).all():
        raise ValueError('Invalid zone; use ' + ', '.join(ZONES))
    for col in ['FGA', 'FGM']:
        d[col] = pd.to_numeric(d[col], errors='raise')
    if (d['FGA'] < 0).any() or (d['FGM'] < 0).any() or (d['FGM'] > d['FGA']).any():
        raise ValueError('Shot counts must satisfy 0 <= FGM <= FGA')
    return d.groupby(['TEAM_ABBREVIATION', 'PLAYER_NAME', 'ZONE'], as_index=False)[['FGA', 'FGM']].sum()


def validate_defense(df: pd.DataFrame) -> pd.DataFrame:
    required = {'TEAM_ABBREVIATION', 'ZONE', 'FGA_ALLOWED', 'FGM_ALLOWED'}
    if not required.issubset(df.columns):
        raise ValueError('Defense CSV requires TEAM_ABBREVIATION, ZONE, FGA_ALLOWED, FGM_ALLOWED')
    d = df.copy()
    d['TEAM_ABBREVIATION'] = d['TEAM_ABBREVIATION'].astype(str).str.upper().str.strip()
    d['ZONE'] = d['ZONE'].astype(str).str.strip()
    if not d['ZONE'].isin(ZONES).all():
        raise ValueError('Invalid defensive zone')
    for c in ['FGA_ALLOWED', 'FGM_ALLOWED']:
        d[c] = pd.to_numeric(d[c], errors='raise')
    if (d['FGA_ALLOWED'] < 0).any() or (d['FGM_ALLOWED'] < 0).any() or (d['FGM_ALLOWED'] > d['FGA_ALLOWED']).any():
        raise ValueError('Invalid defensive attempts/makes')
    return d.groupby(['TEAM_ABBREVIATION', 'ZONE'], as_index=False)[['FGA_ALLOWED', 'FGM_ALLOWED']].sum()


def player_zone_model(player: str, team: str, profiles: pd.DataFrame | None,
                      defense: pd.DataFrame | None = None, opponent: str = '',
                      strength: float = 1., player_baseline: dict | None = None) -> dict:
    """Shrink noisy observed FG% toward zone priors; defense toward zone average.

    This is an interpretable estimator, not a trained tracking-based xFG model.
    """
    if profiles is None or profiles.empty:
        p = pd.DataFrame(columns=['ZONE', 'FGA', 'FGM'])
    else:
        p = profiles[(profiles['TEAM_ABBREVIATION'] == team) &
                     (profiles['PLAYER_NAME'].str.casefold() == player.casefold())]
    attempts = np.array([float(p.loc[p['ZONE'] == z, 'FGA'].sum()) for z in ZONES])
    makes = np.array([float(p.loc[p['ZONE'] == z, 'FGM'].sum()) for z in ZONES])
    # More shrinkage for low sample sizes; no spurious 100% shooting estimates.
    probs = (makes + 35 * np.array([PRIOR_MAKE[z] for z in ZONES])) / (attempts + 35)
    shares = (attempts + 35 * PRIOR_SHARE) / (attempts.sum() + 35)
    source = 'shot chart' if attempts.sum() else 'league priors'
    # When NBA shot charts are missing, use the player's observed box-score
    # 3PA/FGA and 3PM/3PA instead of giving every player the same 3PT profile.
    # This does not infer rim-vs-midrange behavior: those remain league priors.
    if attempts.sum() == 0 and player_baseline is not None:
        b = player_baseline
        fga = max(0., float(b.get('FGA', 0.) or 0.))
        tpa = min(fga, max(0., float(b.get('FG3A', 0.) or 0.)))
        tpm = min(tpa, max(0., float(b.get('FG3M', 0.) or 0.)))
        games = max(1., float(b.get('GP', 1.) or 1.))
        if fga > .1:
            observed_share = tpa / fga
            # Regularize with the league-average 3PA share, but preserve large
            # differences between player shooting styles.
            three_share = (tpa * games + 12. * PRIOR_SHARE[3:].sum()) / (fga * games + 12.)
            shares[:3] = PRIOR_SHARE[:3] / PRIOR_SHARE[:3].sum() * (1. - three_share)
            shares[3:] = PRIOR_SHARE[3:] / PRIOR_SHARE[3:].sum() * three_share
            if tpa > .05:
                p3 = (tpm * games + 18. * .355) / (tpa * games + 18.)
                probs[3:] = np.clip(p3, .18, .55)
            source = 'player box-score 3PA/3PM + estimated 2PT zones'
    # Current shot charts can contain only 1-2 games. Anchor their 3PA mix to
    # the historical/current per-game player box-score profile even when charts
    # exist. This avoids early-season noise turning an interior player into a
    # high-volume perimeter shooter. Keep real corner/above-break proportions.
    if player_baseline is not None:
        b = player_baseline
        fga = max(0., float(b.get('FGA', 0.) or 0.))
        tpa = min(fga, max(0., float(b.get('FG3A', 0.) or 0.)))
        tpm = min(tpa, max(0., float(b.get('FG3M', 0.) or 0.)))
        gp = max(1., float(b.get('GP', 1.) or 1.))
        if fga > .2:
            # The box score is season+historical shrunk upstream. The per-game
            # shooting style is more stable than a handful of chart attempts.
            base_three = tpa / fga
            chart_three = float(shares[3:].sum())
            chart_weight = min(.45, attempts.sum() / (attempts.sum() + 150.))
            target_three = float(np.clip((1-chart_weight)*base_three + chart_weight*chart_three, 0., .90))
            two = max(float(shares[:3].sum()), 1e-9)
            three = max(float(shares[3:].sum()), 1e-9)
            shares[:3] *= (1-target_three)/two
            shares[3:] *= target_three/three
            # Use the observed season+historical 3PM rate to regularize a noisy
            # shot-chart 3P%. Real zone makes still contribute at high sample size.
            if tpa > .05:
                base_pct = (tpm*gp + .355*20) / (tpa*gp + 20)
                chart_pct = (makes[3:].sum()+.355*35)/(attempts[3:].sum()+35)
                pct_weight = min(.60, attempts[3:].sum()/(attempts[3:].sum()+100.))
                target_pct = float(np.clip((1-pct_weight)*base_pct+pct_weight*chart_pct, .10, .65))
                probs[3:] = np.clip(probs[3:] * target_pct / max(float(np.average(probs[3:], weights=shares[3:])), .05), .10, .65)
            source += ' + historical 3PA/3PM anchor'
    shares /= shares.sum()
    adjustments = np.zeros(len(ZONES))
    if defense is not None and not defense.empty:
        q = defense[defense['TEAM_ABBREVIATION'] == opponent]
        for i, z in enumerate(ZONES):
            row = q[q['ZONE'] == z]
            if not row.empty:
                n = float(row['FGA_ALLOWED'].sum())
                made = float(row['FGM_ALLOWED'].sum())
                # Defensive shooting percentage is a noisy proxy, not a measured matchup effect.
                defensive_p = (made + 150 * PRIOR_MAKE[z]) / (n + 150)
                adjustments[i] = float(np.clip((defensive_p - PRIOR_MAKE[z]) * strength, -.06, .06))
    probs = np.clip(probs + adjustments, .15, .85)
    expected = float(sum(shares[i] * probs[i] * (3 if i >= 3 else 2) for i in range(5)))
    return {'zones': ZONES, 'attempts': attempts, 'makes': makes, 'shares': shares,
            'make_prob': probs, 'defense_adjustment': adjustments, 'points_per_attempt': expected,
            'sample_attempts': float(attempts.sum()), 'source': source}


def fetch_team_shots(team_id: int, season: str, team_abbr: str) -> pd.DataFrame:
    """User-initiated NBA.com fetch. Requests may be blocked or throttled."""
    from nba_api.stats.endpoints import shotchartdetail
    raw = shotchartdetail.ShotChartDetail(
        team_id=int(team_id), player_id=0, season_nullable=season,
        season_type_all_star='Regular Season', context_measure_simple='FGA',
        timeout=35).get_data_frames()[0]
    if raw.empty:
        raise ValueError('NBA.com returned no shot attempts; check season or connection')
    return aggregate_nba_shots(raw, team_abbr)


# Historical shot locations are collected by stable NBA PLAYER_ID rather than
# old team membership; downloaded files are cached independently per season.
def historical_seasons(season: str, count: int = 2) -> tuple[str, ...]:
    year = int(str(season)[:4])
    return tuple(f"{year-i}-{str(year-i+1)[-2:]}" for i in range(1, count+1))


def fetch_player_shots(player_id: int, season: str, current_team: str,
                       current_name: str) -> pd.DataFrame:
    """Real observed shots for one NBA player in one season, including trades.

    Never use the historical TEAM_ID as the current roster assignment.
    Empty tables are a legitimate no-attempt observation.
    """
    from nba_api.stats.endpoints import shotchartdetail
    raw = shotchartdetail.ShotChartDetail(
        team_id=0, player_id=int(player_id), season_nullable=season,
        season_type_all_star='Regular Season', context_measure_simple='FGA',
        timeout=35).get_data_frames()[0]
    if raw.empty:
        return pd.DataFrame(columns=['TEAM_ABBREVIATION','PLAYER_NAME','ZONE','FGA','FGM'])
    raw = raw.copy()
    raw['PLAYER_NAME'] = str(current_name)
    return aggregate_nba_shots(raw, current_team)


def merge_historical_profiles(current: pd.DataFrame, historical: pd.DataFrame,
                              *, weight: float = .4) -> pd.DataFrame:
    """Add weighted older shot evidence without falsely labeling it current-season.

    Current roster is the authority; historical observations are downweighted
    rather than replacing real current attempts.
    """
    if historical is None or historical.empty:
        return validate_profiles(current)
    older = historical.copy()
    older[['FGA','FGM']] = older[['FGA','FGM']].astype(float) * float(weight)
    return validate_profiles(pd.concat([current, older], ignore_index=True))
