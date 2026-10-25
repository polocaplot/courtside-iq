"""Explicit, cached bootstrap of real historical ML; no synthetic records."""
from __future__ import annotations
from pathlib import Path
from typing import Callable

DEFAULT_SEASONS = ('2023-24', '2024-25', '2025-26')


def bootstrap_model(folder: Path, seasons=DEFAULT_SEASONS, *, fetch=None, train=None):
    """Fetch official game logs, validate with the existing chronological trainer.

    Invoked only by an explicit user action. An unavailable NBA feed raises and
    leaves any existing trained model unchanged (training.py handles atomic writes).
    """
    if fetch is None:
        from training import fetch_historical as fetch
    if train is None:
        from training import train as train
    folder = Path(folder)
    seasons = tuple(seasons)
    if not seasons or len(set(seasons)) != len(seasons):
        raise ValueError('Select at least one unique historical season')
    logs = fetch(seasons, folder / 'game_logs')
    if logs is None or logs.empty:
        raise ValueError('No historical player game logs returned; model not trained')
    result = train(logs, folder=folder)
    if not (folder / 'manifest.json').is_file():
        raise RuntimeError('Training returned without a validated model manifest')
    return result
