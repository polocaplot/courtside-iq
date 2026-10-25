"""Conservative lineup-conditioned defensive event modifiers.

No optical tracking or true matchup labels. Rates are shrinkage-adjusted proxies
from player box scores and optionally defended-shot observations. Team defense
already contributes to the baseline score; use limited adjustments to avoid
large double counting. Never use information after tipoff for a pregame forecast.
"""
from __future__ import annotations
import numpy as np
from shot_intelligence import ZONES


def defensive_rates(roster, minutes, strength=1.0):
    """Season steal/block/rebound rate proxies with strong shrinkage.

    Returns per-player effects bounded to plausible ranges. STL/BLK/REB are
    season per-game averages, not individual matchup tracking or event rates.
    """
    strength = float(np.clip(strength, 0, 1.5))
    m = np.maximum(roster['MIN'].to_numpy(float), 1)
    def effect(stat, prior, prior_minutes, cap):
        per36 = np.nan_to_num(roster[stat].to_numpy(float), nan=prior, posinf=prior, neginf=prior) / m * 36
        # Effective evidence grows with season minutes; per-game columns do
        # not expose exact season sample counts, so use conservative shrinkage.
        # Per-game minutes alone cannot establish reliability: a player who
        # played 35 minutes in one game is still a tiny defensive sample.
        gp = (roster['GP'].to_numpy(float) if 'GP' in roster else
              np.full(len(roster), 8.))
        gp = np.maximum(np.nan_to_num(gp, nan=0., posinf=0., neginf=0.), 0.)
        minutes_evidence = m * gp
        reliability = minutes_evidence / (minutes_evidence + prior_minutes * 25.)
        return np.clip((per36 - prior) * reliability * strength, -cap, cap)
    return {'steal': effect('STL', 1.25, 32, .9),
            'block': effect('BLK', .8, 36, 1.0),
            'rebound': effect('REB', 5.5, 45, 2.0)}


def lineup_defense(on_court, rates):
    """Lineup strength per simulated game. Exactly five defenders expected."""
    if on_court.ndim != 2 or np.any(on_court.sum(axis=1) != 5):
        raise ValueError('Defense requires exactly five players on court')
    steal = np.sum(on_court * rates['steal'][None, :], axis=1) / 5
    block = np.sum(on_court * rates['block'][None, :], axis=1) / 5
    rebound = np.sum(on_court * rates['rebound'][None, :], axis=1) / 5
    return {'turnover_delta': np.clip(steal * .012, -.015, .015),
            'block_delta': np.clip(block * .022, -.025, .025),
            'oreb_delta': np.clip(-rebound * .008, -.018, .018),
            'rim_deterrence': np.clip(block * .035, -.035, .035),
            'contest_delta': np.clip(-block * .008 - steal * .003, -.012, .012)}


def zone_shares_after_defense(shares, deterrence):
    """Small rim-attempt redistribution toward non-rim zones; sums to one."""
    shares = np.asarray(shares, dtype=float)
    deterrence = np.asarray(deterrence, dtype=float)
    out = shares.copy()
    change = np.clip(out[:, 0] * deterrence, -.025, .025)
    out[:, 0] -= change
    rest = out[:, 1:]
    norm = np.maximum(rest.sum(axis=1), 1e-9)
    out[:, 1:] += change[:, None] * rest / norm[:, None]
    out = np.maximum(out, 0)
    return out / out.sum(axis=1, keepdims=True)


def lineup_contest_modifier(defender_modifiers, on_court, zones, rates=None):
    """Conservative zone-aware *probabilistic* contest effect.

    Observed defended-shot aggregates inform player/zone effects, but no actual
    shooter-to-defender assignments are available. We weight eligible defenders
    by a shrunk box-score proxy (blocks near the rim, steals on the perimeter),
    rather than treating all five players as equally likely contesters. The
    weighted effect is centered against the five-man mean to avoid exaggerating
    noisy player aggregates, and remains capped at 3 percentage points.
    Missing tracking observations always yield exactly zero.
    """
    zones = np.asarray(zones, dtype=int)
    court = np.asarray(on_court, dtype=bool)
    if court.ndim != 2 or len(court) != len(zones):
        raise ValueError('Expected [possessions, players] on-court matrix')
    if np.any(court.sum(axis=1) != 5):
        raise ValueError('Each possession needs exactly five defenders')
    if np.any((zones < 0) | (zones >= len(ZONES))):
        raise ValueError('Unknown shot zone index')
    if defender_modifiers is None:
        return np.zeros(len(zones))
    mods = np.asarray(defender_modifiers, dtype=float)
    if mods.shape != (court.shape[1], len(ZONES)):
        raise ValueError('Defender modifiers must match roster and zones')
    mods = np.clip(np.nan_to_num(mods, nan=0., posinf=0., neginf=0.), -.04, .04)
    if not np.any(mods):
        return np.zeros(len(zones))
    # Defensive proxies are centered around zero; shrink the weighting to
    # avoid assigning a defender with certainty based on box-score statistics.
    weights = np.ones_like(court, dtype=float)
    if rates is not None:
        block = np.asarray(rates['block'], dtype=float)
        steal = np.asarray(rates['steal'], dtype=float)
        if block.shape != (court.shape[1],) or steal.shape != (court.shape[1],):
            raise ValueError('Defensive rate shape mismatch')
        near_rim = (zones <= 1)[:, None]
        influence = np.where(near_rim, block[None, :], steal[None, :])
        weights = np.clip(1. + .35 * influence, .6, 1.5)
    weights *= court
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    selected = mods[:, zones].T
    observed = np.sum(selected * weights, axis=1)
    # This is a lineup-level expected contest effect, not identified primary
    # defender data. Retain only 75% to reduce overlap with team baselines.
    return np.clip(.75 * observed, -.03, .03)
