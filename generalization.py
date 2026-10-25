"""Cross-season evidence weighting and lineup uncertainty.

Observed five-man units are *one* source of evidence, never the entire rotation
model. Current eligibility and minutes remain authoritative. No historical lineup
is considered confirmed for an upcoming game.
"""
from __future__ import annotations
import numpy as np


def lineup_evidence_weight(counts, valid, *, pseudo_possessions=500., max_weight=.65):
    """Probability of using an observed lineup rather than the broad minute model.

    Counts may be possessions or minute-based proxy weights. Low coverage, small
    samples and concentrated one-unit datasets are deliberately discounted.
    Returns one probability per simulated game, in [0, max_weight].
    """
    c = np.asarray(counts, dtype=float)
    v = np.asarray(valid, dtype=bool)
    if c.ndim != 1 or v.ndim != 2 or v.shape[1] != len(c):
        raise ValueError('Invalid observed lineup evidence shape')
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Lineup weights must be nonnegative and finite')
    total = c.sum()
    if total <= 0:
        return np.zeros(len(v))
    available = (v * c[None, :]).sum(axis=1)
    coverage = available / total
    # Effective number of distinct units, bounded by 5 to avoid overconfidence
    # when just one or two lineups have ever played together.
    diversity = total * total / max(np.dot(c, c), 1e-12)
    breadth = min(1., diversity / 5.)
    evidence = available / (available + float(pseudo_possessions))
    return np.clip(float(max_weight) * evidence * np.sqrt(coverage) * (.35 + .65 * breadth), 0., max_weight)


def sample_reliability(games, prior_games=18.):
    """Shrink current-season estimates toward longer-run priors early in season."""
    gp = np.asarray(games, dtype=float)
    return np.clip(np.maximum(np.nan_to_num(gp, nan=0.), 0.) /
                   (np.maximum(np.nan_to_num(gp, nan=0.), 0.) + prior_games), 0., 1.)


def blend_rates(current, prior, current_games, *, prior_games=18.):
    """Blend only matching-player rates, not historical team membership."""
    current = np.asarray(current, dtype=float)
    prior = np.asarray(prior, dtype=float)
    weight = sample_reliability(current_games, prior_games)
    return np.where(np.isfinite(prior), weight * current + (1 - weight) * prior, current)
