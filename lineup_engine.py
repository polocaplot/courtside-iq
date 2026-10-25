"""Approximate on-court lineup sampler for simulated possessions.

Samples exactly five players per side per rotation segment. A supplied observed
lineup-frequency CSV can influence lineup selection, but is NOT a reconstruction
of a coach's substitutions, tracking data, or a real game timeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from generalization import lineup_evidence_weight

LINEUP_COLUMNS = ('TEAM_ABBREVIATION', 'PLAYER_1', 'PLAYER_2', 'PLAYER_3', 'PLAYER_4', 'PLAYER_5', 'POSSESSIONS')


def validate_lineups(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(LINEUP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError('Lineup CSV missing: ' + ', '.join(sorted(missing)))
    out = df[list(LINEUP_COLUMNS)].copy()
    out['TEAM_ABBREVIATION'] = out['TEAM_ABBREVIATION'].astype(str).str.upper().str.strip()
    out['POSSESSIONS'] = pd.to_numeric(out['POSSESSIONS'], errors='raise')
    if not np.isfinite(out['POSSESSIONS']).all() or (out['POSSESSIONS'] <= 0).any():
        raise ValueError('Lineup possessions must be positive and finite')
    for i in range(1, 6):
        out[f'PLAYER_{i}'] = out[f'PLAYER_{i}'].astype(str).str.strip()
    if any(len(set(row)) != 5 for row in out[[f'PLAYER_{i}' for i in range(1, 6)]].itertuples(index=False, name=None)):
        raise ValueError('Every lineup must have five distinct players')
    return out


def choose_lineups(rng, minutes: np.ndarray, names: list[str], team: str,
                   priors: pd.DataFrame | None = None) -> np.ndarray:
    """Return [games, players] boolean on-court matrix, exactly five per game.

    Baseline: sample 5 without replacement weighted by projected playing time.
    Optional lineup priors: sample observed five-man units by their possession
    counts, conditional on all five having nonzero projected minutes. The priors
    are approximate; if a game has no valid unit, it uses baseline sampling.
    """
    n, p = minutes.shape
    if p < 5 or np.any((minutes > 0).sum(axis=1) < 5):
        raise ValueError('Each game needs at least five eligible players')
    w = np.maximum(minutes, 0.)
    # Gumbel-top-k weighted sampling without replacement. Zero-minute players
    # receive -inf keys and cannot be chosen.
    u = np.maximum(rng.random((n, p)), 1e-12)
    keys = np.log(np.maximum(w, 1e-12)) - np.log(-np.log(u))
    keys[w <= 0] = -np.inf
    ix = np.argpartition(keys, -5, axis=1)[:, -5:]
    selected = np.zeros((n, p), dtype=bool)
    selected[np.arange(n)[:, None], ix] = True
    if priors is None or priors.empty:
        return selected
    table = priors[priors['TEAM_ABBREVIATION'] == team]
    if table.empty:
        return selected
    name_map = {str(name).casefold(): i for i, name in enumerate(names)}
    units, counts = [], []
    for row in table.itertuples(index=False):
        members = [name_map.get(str(getattr(row, f'PLAYER_{i}')).casefold()) for i in range(1, 6)]
        if None not in members and len(set(members)) == 5:
            units.append(members)
            counts.append(float(row.POSSESSIONS))
    if not units:
        return selected
    units = np.asarray(units, dtype=int)
    counts = np.asarray(counts, dtype=float)
    # Weights account for observed frequency AND current eligibility; this
    # prevents historical units with injured players being forced onto court.
    valid = np.all(w[:, units] > 0, axis=2)
    choices = np.where(valid, counts[None, :], 0.)
    totals = choices.sum(axis=1)
    usable = totals > 0
    if usable.any():
        # Blend observed units with the broad minute-weighted model. Even an
        # enormous dataset cannot force a single historical lineup into every
        # possession; unfamiliar units remain eligible after trades/injuries.
        trust = lineup_evidence_weight(counts, valid)
        use_observed = usable & (rng.random(n) < trust)
        if use_observed.any():
            r = rng.random(use_observed.sum()) * totals[use_observed]
            sampled = np.sum(r[:, None] > np.cumsum(choices[use_observed], axis=1), axis=1)
            selected[use_observed] = False
            selected[np.where(use_observed)[0][:, None], units[sampled]] = True
    return selected


def choose_target_balanced_lineups(rng, target_minutes: np.ndarray,
                                   played_slots: np.ndarray, total_slots: np.ndarray,
                                   current_step: int, interval: int,
                                   previous: np.ndarray | None,
                                   eligible_weights: np.ndarray,
                                   names: list[str], team: str,
                                   priors: pd.DataFrame | None = None) -> np.ndarray:
    """Choose five players while actively tracking remaining minute budgets.

    Unlike independent weighted draws, this attempts to match per-game minute
    allocations over the full game. The model is still a coarse rotation
    approximation, not a verified coaching substitution timeline. The
    caller can suppress foul-troubled players via eligible_weights.
    """
    n, p = target_minutes.shape
    if n == 0:
        return np.zeros((0, p), dtype=bool)
    # Played slots are possession-time equivalents, each lasting 48 / total.
    desired_slots = target_minutes * (total_slots[:, None] / 48.)
    deficit = desired_slots - played_slots
    remaining = np.maximum(total_slots - current_step, 1)
    # A player who needs 18 more slots in 20 remaining slots is urgent;
    # a player with only 2 remaining slots is not. This makes it possible
    # to recover from previous substitutions instead of repeating their bias.
    urgency = deficit / remaining[:, None]
    available = eligible_weights > 0
    # Honor true ineligibility, but allow a low-priority player to fill a unit
    # if all remaining budgets are nearly exhausted.
    urgency = np.where(available, urgency, -1e6)
    # A modest preference to keep a unit together, preventing wholesale
    # substitution every rotation interval. Random jitter adds plausible
    # game-to-game variation without overwhelming minute constraints.
    if previous is not None:
        urgency += .065 * previous
    urgency += rng.normal(0., .018, size=(n, p))
    # Fouls suppress the priority, but never disqualify a team with <5 healthy
    # alternatives; the caller already handles eligibility fallback.
    ratio = eligible_weights / np.maximum(deficit, .25)
    urgency -= np.where(available, np.clip(1. - ratio, 0., 1.) * .18, 0.)
    top = np.argpartition(urgency, -5, axis=1)[:, -5:]
    selected = np.zeros((n, p), dtype=bool)
    selected[np.arange(n)[:, None], top] = True
    return selected
