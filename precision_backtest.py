"""Leakage-safe rolling historical accuracy audit using real NBA player game logs.

Run: python precision_backtest.py /path/to/NBA_LeagueGameLog.csv --output audit.json
This audits minutes, usage proxies and box-score stats. It does not claim to
backtest the full possession simulator or reconstruct historic injury reports.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training import standardize_logs


def rolling_audit(raw: pd.DataFrame, min_prior_games: int = 5) -> dict:
    df = standardize_logs(raw).sort_values(['PLAYER_ID', 'GAME_DATE', 'GAME_ID'])
    stats = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
    groups = df.groupby('PLAYER_ID', sort=False)
    audit = df[['PLAYER_ID', 'GAME_DATE']].copy()
    audit['N_PRIOR'] = groups.cumcount()
    for stat in stats:
        # shift(1) is essential: current game outcomes must never be features.
        audit[f'PRED_{stat}'] = groups[stat].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        audit[f'ACT_{stat}'] = df[stat].to_numpy(float)
    eligible = audit[audit['N_PRIOR'] >= min_prior_games].copy()
    if eligible.empty:
        raise ValueError('No eligible games; import more genuine historical observations')
    dates = sorted(eligible['GAME_DATE'].unique())
    if len(dates) < 10:
        raise ValueError('At least 10 distinct eligible dates required')
    cutoff = dates[int(len(dates) * .8)]
    holdout = eligible[eligible['GAME_DATE'] >= cutoff]
    if holdout.empty:
        raise ValueError('Empty holdout period')
    metrics = {}
    for stat in stats:
        actual = holdout[f'ACT_{stat}'].to_numpy(float)
        pred = holdout[f'PRED_{stat}'].to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(pred)
        if not valid.any():
            continue
        diff = pred[valid] - actual[valid]
        metrics[stat] = {'n': int(valid.sum()), 'mae': round(float(np.mean(abs(diff))), 3),
                         'bias': round(float(np.mean(diff)), 3),
                         'rmse': round(float(np.sqrt(np.mean(diff**2))), 3)}
    return {'method': 'chronological last-20%-of-dates holdout, prior-10-game rolling baseline',
            'note': 'Box-score baseline only; not full simulator validation. Historical injuries not reconstructed.',
            'holdout_start': str(pd.Timestamp(cutoff).date()), 'eligible_rows': len(eligible),
            'holdout_rows': len(holdout), 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('precision_audit.json'))
    args = parser.parse_args()
    report = rolling_audit(pd.read_csv(args.csv, dtype={'GAME_ID': str, 'PLAYER_ID': str}))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
