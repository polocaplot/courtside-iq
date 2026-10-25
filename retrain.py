"""Incremental once-per-day retraining while the local dashboard is running."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
from training import DEFAULT_FOLDER, fetch_historical, load, train, standardize_logs


def update_models(folder: Path = DEFAULT_FOLDER, season: str = '2026-27',
                  minimum_new_rows: int = 50) -> dict:
    """Fetch current season once (or from disk), refresh when >=minimum_new_rows.

    Requires previously trained archive. No third-party data mutations.
    """
    folder = Path(folder)
    manifest, _models, history = load(folder)
    current = fetch_historical((season,), folder / 'game_logs', refresh=True)
    new = standardize_logs(current)
    old = standardize_logs(history)
    ids = set(zip(old['PLAYER_ID'], old['GAME_ID']))
    added = new[[ (x,y) not in ids for x,y in zip(new['PLAYER_ID'], new['GAME_ID']) ]]
    if len(added) < minimum_new_rows:
        return {'status':'unchanged', 'new_games': len(added), 'message':'No model update required'}
    combined = pd.concat([old, added], ignore_index=True)
    fresh = train(combined, folder=folder)
    return {'status':'retrained', 'new_games':len(added),
            'message':f"Updated on {fresh['created_utc']}", 'manifest':fresh}
