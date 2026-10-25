"""Opt-in, local-only automatic forecast recording and completed-game audits.

All snapshots must be made before real tipoff; final outcomes are fetched from
an actual game feed, never inferred from the projection. A game is graded once.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
import json
import os
import pandas as pd
from tracking import DEFAULT_PATH, load_history, store_snapshot, grade_snapshot

GRADE_PATH = Path.home() / '.courtsideiq' / 'grades.json'


def save_if_due(result, game, scoring, source, path: Path = DEFAULT_PATH, hours: int = 4):
    """Save at most once per game per interval. No synthetic/past games."""
    if game is None or game.get('state') != 'pre':
        return None
    existing = [r for r in load_history(path) if str(r.get('game_id')) == str(game.get('id'))]
    if existing:
        latest = max(datetime.fromisoformat(r['saved_at']) for r in existing)
        if datetime.now(timezone.utc) - latest < timedelta(hours=hours):
            return None
    return store_snapshot(result, game, scoring, source, path=path)


def _read(path):
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, dict) else {}
    except (ValueError, OSError):
        return {}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def grade_completed(schedule_lookup, actual_lookup, history_path: Path = DEFAULT_PATH,
                    grades_path: Path = GRADE_PATH, max_games: int = 4):
    """Audit pending snapshots, only after the scoreboard reports final.

    External functions are injected for offline testing. One final box score
    per game is fetched and used for all its previously saved snapshots.
    """
    records = load_history(history_path)
    grades = _read(grades_path)
    checked = 0
    errors = []
    by_game = {}
    now = datetime.now(timezone.utc)
    for record in records:
        if record['id'] in grades:
            continue
        try:
            tip = datetime.fromisoformat(record['tipoff'])
        except (ValueError, KeyError):
            continue
        if tip >= now:
            continue
        by_game.setdefault(str(record['game_id']), []).append(record)
    for gid, snaps in sorted(by_game.items(), key=lambda kv: kv[1][0]['tipoff']):
        if checked >= max_games:
            break
        checked += 1
        try:
            game_day = date.fromisoformat(snaps[0]['tipoff'][:10])
            games = schedule_lookup(game_day.isoformat())
            game = next((g for g in games if str(g['id']) == gid), None)
            if game is None or game.get('state') != 'post':
                continue
            actual = actual_lookup(gid)
            if actual.empty:
                raise ValueError('Final player box score is empty')
            for snap in snaps:
                grade = grade_snapshot(snap, actual)
                if grade.empty:
                    continue
                hscore = float(game['home']['score'])
                ascore = float(game['away']['score'])
                grades[snap['id']] = {
                    'game_id': gid, 'matchup': snap['matchup'],
                    'saved_at': snap['saved_at'], 'tipoff': snap['tipoff'], 'graded_at': now.isoformat(),
                    'fantasy_mae': round(float(grade['Abs. Error'].mean()), 3),
                    'players_graded': len(grade),
                    'home_won': int(hscore > ascore),
                    'home_score': hscore, 'away_score': ascore,
                    'home_win_prob': snap.get('home_win_prob'),
                    'home_projected': snap.get('home_projected'),
                    'away_projected': snap.get('away_projected'),
                    'home_score_abs_error': round(abs(hscore - float(snap.get('home_projected', 0))), 2),
                    'away_score_abs_error': round(abs(ascore - float(snap.get('away_projected', 0))), 2),
                    'player_grades': grade.to_dict('records'),
                }
        except Exception as exc:
            errors.append(f'{gid}: {type(exc).__name__}: {str(exc)[:120]}')
    if grades:
        _write(grades_path, grades)
    return grades, errors


def grade_summary(grades: dict):
    if not grades:
        return pd.DataFrame()
    return pd.DataFrame([{k: v.get(k) for k in (
        'matchup', 'saved_at', 'fantasy_mae', 'players_graded', 'home_win_prob',
        'home_won', 'home_score_abs_error', 'away_score_abs_error')}
        for v in grades.values()])


def calibration_frame(grades: dict):
    """Real, saved-before-tipoff game probabilities only; first forecast per game."""
    by_game = {}
    for v in grades.values():
        if v.get('home_win_prob') is None:
            continue
        gid = v['game_id']
        if gid not in by_game or v['saved_at'] < by_game[gid]['saved_at']:
            by_game[gid] = v
    return pd.DataFrame([{'GAME_DATE': v['tipoff'], 'FORECAST_SAVED_AT': v['saved_at'],
                          'GAME_ID': v['game_id'], 'HOME_WIN_PROB': v['home_win_prob'],
                          'HOME_WON': v['home_won']}
                         for v in by_game.values()])
