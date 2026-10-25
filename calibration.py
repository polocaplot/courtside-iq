"""Strict chronological calibration audit for saved *pregame* game predictions.

Input must be actual pregame forecasts and final results, one row per game.
This module never creates fake observations or claims sportsbook-level accuracy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

REQUIRED = {'GAME_DATE', 'HOME_WIN_PROB', 'HOME_WON'}


def validate_results(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError('Calibration CSV missing: ' + ', '.join(sorted(missing)))
    d = data.copy()
    d['GAME_DATE'] = pd.to_datetime(d['GAME_DATE'], errors='raise', utc=True)
    d['HOME_WIN_PROB'] = pd.to_numeric(d['HOME_WIN_PROB'], errors='raise')
    d['HOME_WON'] = pd.to_numeric(d['HOME_WON'], errors='raise')
    if not np.isfinite(d['HOME_WIN_PROB']).all() or not d['HOME_WIN_PROB'].between(0, 1).all():
        raise ValueError('Win probabilities must be finite and between 0 and 1')
    if not d['HOME_WON'].isin([0, 1]).all():
        raise ValueError('HOME_WON must be 0 or 1')
    if 'GAME_ID' in d and d['GAME_ID'].duplicated().any():
        raise ValueError('Duplicate game IDs in calibration input')
    if 'FORECAST_SAVED_AT' in d:
        saved = pd.to_datetime(d['FORECAST_SAVED_AT'], errors='raise', utc=True)
        if (saved >= d['GAME_DATE']).any():
            raise ValueError('Some forecasts were saved after tipoff; refusing leakage')
    return d.sort_values('GAME_DATE').reset_index(drop=True)


def evaluate_calibration(data: pd.DataFrame, min_games: int = 80) -> dict:
    d = validate_results(data)
    if len(d) < min_games:
        raise ValueError(f'Need at least {min_games} completed, genuine pregame games; received {len(d)}')
    # Holdout on unique calendar dates; no same-date data can enter training.
    dates = d['GAME_DATE'].dt.normalize().unique()
    if len(dates) < 15:
        raise ValueError('Need forecasts from at least 15 different game dates')
    cutoff = dates[int(len(dates) * .8)]
    train = d[d['GAME_DATE'].dt.normalize() < cutoff]
    test = d[d['GAME_DATE'].dt.normalize() >= cutoff]
    if len(train) < 50 or len(test) < 15:
        raise ValueError('Need at least 50 training games and 15 held-out games')
    calibrator = IsotonicRegression(y_min=.01, y_max=.99, out_of_bounds='clip')
    calibrator.fit(train['HOME_WIN_PROB'], train['HOME_WON'])
    raw = np.clip(test['HOME_WIN_PROB'].to_numpy(float), .001, .999)
    corrected = np.clip(calibrator.predict(raw), .001, .999)
    actual = test['HOME_WON'].to_numpy(int)
    brier_raw = float(brier_score_loss(actual, raw))
    brier_cal = float(brier_score_loss(actual, corrected))
    log_raw = float(log_loss(actual, raw, labels=[0, 1]))
    log_cal = float(log_loss(actual, corrected, labels=[0, 1]))
    accepted = brier_cal < brier_raw and log_cal <= log_raw
    # Do not auto-activate unless both held-out proper scoring rules improve.
    table = test[['GAME_DATE', 'HOME_WIN_PROB', 'HOME_WON']].copy()
    table['CALIBRATED'] = corrected
    return {'train_games': len(train), 'holdout_games': len(test),
            'holdout_from': str(pd.Timestamp(cutoff).date()),
            'raw_brier': brier_raw, 'calibrated_brier': brier_cal,
            'raw_log_loss': log_raw, 'calibrated_log_loss': log_cal,
            'passed_holdout': bool(accepted), 'holdout': table, 'calibrator': calibrator}
