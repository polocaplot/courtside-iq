"""Bounded, copyable simulation diagnostics. No network or Streamlit dependency."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd


def _effective_minutes(row):
    """Single-player reproduction of the simulator's minute prior."""
    import pandas as pd
    from model import _minute_prior
    try:
        return round(float(_minute_prior(pd.DataFrame([row]))[0]), 3)
    except (ValueError, TypeError, KeyError):
        return None


def _safe(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (datetime,)): return value.isoformat()
    if isinstance(value, (np.ndarray,)): return value.tolist()
    if isinstance(value, (float,)) and not np.isfinite(value): return None
    return str(value)


def make_report(result, *, game, season, data_mode, source, injury_issues=(),
                shot_profiles=None, lineup_rows=0, defender_rows=0,
                player_inputs=None, team_inputs=None, model_used=False,
                model_issue=None, shot_source=None, defender_feed_meta=None):
    """Build a reproducible summary of one completed run (not a raw 150K-play trace).

    Excludes private credentials and only exports players for the selected game.
    """
    if not result: return None
    home, away = str(result.get('home', '')), str(result.get('away', ''))
    hs, vs = np.asarray(result.get('home_score', [])), np.asarray(result.get('away_score', []))
    cfg = result.get('config')
    box = result.get('box')
    box_rows = box.to_dict('records') if isinstance(box, pd.DataFrame) else []
    covered = set()
    if isinstance(shot_profiles, pd.DataFrame) and not shot_profiles.empty:
        if {'TEAM_ABBREVIATION', 'PLAYER_NAME'}.issubset(shot_profiles.columns):
            covered = {(str(t), str(p).casefold()) for t, p in zip(shot_profiles['TEAM_ABBREVIATION'], shot_profiles['PLAYER_NAME'])}
    missing = [f"{r.get('Team')} | {r.get('Player')}" for r in box_rows
               if (str(r.get('Team')), str(r.get('Player')).casefold()) not in covered]
    sample = [{'home': int(h), 'away': int(a)} for h, a in zip(hs[:20], vs[:20])]
    audits = {}
    for team, events in (result.get('possession_audits') or {}).items():
        audits[str(team)] = {str(k): {
            'count': len(v), 'mean': round(float(np.mean(v)), 3) if len(v) else None,
            'first_5': np.asarray(v[:5]).tolist()
        } for k, v in events.items()}
    def select_rows(frame, cols=None):
        if not isinstance(frame, pd.DataFrame): return []
        if 'TEAM_ABBREVIATION' in frame.columns:
            frame = frame[frame['TEAM_ABBREVIATION'].isin([home, away])]
        if cols: frame = frame[[c for c in cols if c in frame.columns]]
        return frame.head(50).to_dict('records')
    historical = select_rows(player_inputs, ['TEAM_ABBREVIATION','PLAYER_NAME','GP','HISTORICAL_PRIOR_USED',
                                              'HISTORICAL_PRIOR_SEASON','HISTORICAL_CURRENT_WEIGHT',
                                              'HISTORICAL_MINUTES_WEIGHT','HISTORICAL_WARNING','BASELINE_SOURCE',
                                              'MIN','MIN_L5','MIN_L10','GAMES_PLAYED'])
    matched = sum(bool(x.get('HISTORICAL_PRIOR_USED')) for x in historical)
    history_warnings = sorted({str(x.get('HISTORICAL_WARNING')).strip() for x in historical
                               if pd.notna(x.get('HISTORICAL_WARNING'))
                               and str(x.get('HISTORICAL_WARNING')).strip()})
    # Structural realism checks are computed from simulated events, not from
    # actual NBA outcomes. They must not be described as backtest accuracy.
    realism = {'verified_out_of_sample': False,
               'rotation_method': 'minute-deficit-aware five-player lineup scheduler',
               'player_usage_method': 'per-minute on-court event rates',
               'overtime_method': 'additional possession events (foul state restarts at OT)',
               'teams': {}}
    for team in (home, away):
        block = (result.get('team_arrays') or {}).get(team, {})
        if not block or 'PTS' not in block:
            continue
        points = np.asarray(block['PTS'], dtype=np.int64)
        fg = np.asarray(block.get('FGM', np.zeros_like(points)), dtype=np.int64)
        three = np.asarray(block.get('3PM', np.zeros_like(points)), dtype=np.int64)
        ft = np.asarray(block.get('FTM', np.zeros_like(points)), dtype=np.int64)
        minutes = np.asarray(block.get('MIN', np.zeros_like(points)), dtype=float)
        team_minutes = minutes.sum(axis=1)
        team_points = points.sum(axis=1)
        event_points = (2 * fg + three + ft).sum(axis=1)
        roster = [r for r in box_rows if r.get('Team') == team]
        minutes_gaps = []
        injury_exclusions = []
        for row in roster:
            if str(row.get('Status', '')).strip().casefold() == 'out':
                injury_exclusions.append({'player': row.get('Player'),
                                          'status': 'Out',
                                          'projected_minutes': row.get('MIN', 0)})
                continue
            inp = next((x for x in historical if x.get('TEAM_ABBREVIATION') == team
                        and x.get('PLAYER_NAME') == row.get('Player')), None)
            if inp is not None:
                prior = _effective_minutes(inp)
                if prior is not None:
                    minutes_gaps.append({'player': row.get('Player'),
                                         'projected_minutes': row.get('MIN'),
                                         'effective_prior': prior,
                                         'difference': round(float(row.get('MIN', 0)) - prior, 2)})
        realism['teams'][team] = {
            'event_scoring_identity_all_games': bool(np.all(team_points == event_points)),
            'mean_regulation_or_overtime_minutes': round(float(np.mean(team_minutes)), 3),
            'minutes_conservation_all_games': bool(np.all(
                np.isclose((team_minutes - 240.) / 25.,
                           np.rint((team_minutes - 240.) / 25.), atol=.02) &
                (team_minutes >= 239.98))),
            'overtime_games': int(np.count_nonzero(team_minutes > 252.)),
            'minute_prior_deviations': minutes_gaps,
            'injury_exclusions_not_counted_as_rotation_errors': injury_exclusions,
        }
    report = {
        'report_type': 'COURTSIDE IQ single-paste diagnostic',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'matchup': game, 'season': season, 'data_mode': data_mode,
        'source_provenance': source,
        'simulation': {
            'engine': result.get('engine_verified'),
            'requested': int(getattr(cfg, 'iterations', 0)),
            'completed': int(result.get('completed_draws', len(hs))),
            'home_score_array_length': len(hs), 'away_score_array_length': len(vs),
            'score_arrays_match_request': bool(len(hs) == len(vs) == int(getattr(cfg, 'iterations', 0))),
            'runtime_seconds': result.get('elapsed_seconds'),
            'random_seed': getattr(cfg, 'seed', None),
            'possession_mode': bool(getattr(cfg, 'possession_mode', False)),
            'shot_mode': bool(getattr(cfg, 'shot_mode', False)),
            'defense_strength': getattr(cfg, 'defense_strength', None),
            'home_mean': round(float(np.mean(hs)), 3) if len(hs) else None,
            'away_mean': round(float(np.mean(vs)), 3) if len(vs) else None,
        },
        'data_quality': {
            'shot_chart_source': result.get('shot_data_source_at_run', shot_source),
            'shot_fetch_check': result.get('shot_data_check'),
            'historical_shot_downloads_pending': result.get('shot_history_pending', 0),
            'players_with_shot_charts': len(box_rows) - len(missing),
            'players_in_box_score': len(box_rows),
            'players_missing_shot_charts': missing,
            'injury_warnings': list(map(str, injury_issues)),
            'lineup_rows': int(lineup_rows), 'defender_rows': int(defender_rows),
            'historical_ml_used': bool(model_used), 'ml_warning': model_issue or ('Trained model not activated; check ML Studio and pregame settings.' if not model_used else ''),
            'recent_minutes_players': sum(any(pd.notna(x.get(k)) for k in ('MIN_L5','MIN_L10')) for x in historical),
            'recent_minutes_inputs': len(historical),
            'reconciled_shooting_players': [str(x.get('PLAYER_NAME')) for x in select_rows(player_inputs, ['PLAYER_NAME','SHOOTING_INPUT_RECONCILED']) if x.get('SHOOTING_INPUT_RECONCILED')],
            'minute_input_audit': [{
                'team': x.get('TEAM_ABBREVIATION'), 'player': x.get('PLAYER_NAME'),
                'season_min': x.get('MIN'), 'last5_min': x.get('MIN_L5'),
                'last10_min': x.get('MIN_L10'), 'games_played': x.get('GAMES_PLAYED', x.get('GP')),
                'effective_minute_prior': _effective_minutes(x),
                'historical_minute_weight': x.get('HISTORICAL_MINUTES_WEIGHT'),
            } for x in historical],
            'defender_data_warning': ('Zero observed defended-shot aggregates: all defensive matchups are estimates; inspect Shot Lab feed errors.' if int(defender_rows) == 0 else ''),
            'defender_feed_errors': list((defender_feed_meta or {}).get('errors', [])),
            'defender_feed_last_attempt': (defender_feed_meta or {}).get('checked_at'),
            'defender_data_kind': (defender_feed_meta or {}).get('defender_data_kind', 'unavailable_or_unknown'),
            'defender_categories_returned': (defender_feed_meta or {}).get('defender_categories', {}),
            'historical_ml_bootstrap_available': True,
            'historical_prior_player_matches': matched,
            'historical_prior_player_inputs': len(historical),
            'historical_prior_warnings': history_warnings,
            'historical_priors_are_not_trained_ml': True,
            'rotation_policy': 'adaptive 9-11 man rotation targets with deficit-aware five-player scheduling, foul-trouble and blowout adjustments',
            'integrated_game_state': 'Approximate coupled lineup/defense/foul/bonus/minutes/shot events; NOT verified play-by-play or tracked defender matchups',
        },
        'realism_audit': realism,
        'shot_calibration': result.get('shot_calibration') or result.get('possession_calibration') or {},
        'scoring_calibration_audit': {
            t: {'pregame_target': c.get('pregame_target'),
                'simulated_mean': round(float(np.mean(hs if t == home else vs)), 3),
                'residual_points': round(float(np.mean(hs if t == home else vs)) - float(c.get('pregame_target')), 3),
                'is_verified_out_of_sample': False}
            for t, c in (result.get('possession_calibration') or {}).items()
            if c.get('pregame_target') is not None and len(hs) and len(vs)
        },
        'possession_calibration': result.get('possession_calibration') or {},
        'quarter_summary': ({'source': result['quarter_summary']['source'],
                             'includes_overtime_in_quarters': False,
                             'teams': {team: {
                                 'quarter_mean_points': q['quarter_mean_points'],
                                 'end_of_quarter_mean_score': q['end_of_quarter_mean_score'],
                                 'quarter_event_means': q['quarter_event_means'],
                                 'overtime_mean_points_all_games': q['overtime_mean_points_all_games'],
                             } for team, q in result['quarter_summary']['teams'].items()}}
                            if result.get('quarter_summary') else None),
        'event_audit': audits,
        'sample_scores': sample,
        'player_projections': box_rows,
        'historical_player_evidence': historical,
        'pregame_player_inputs': select_rows(player_inputs, ['TEAM_ABBREVIATION','PLAYER_NAME','MIN','MIN_L5','MIN_L10','GAMES_PLAYED','GP','PTS','REB','AST','FGA','FGM','FG3A','FG3M','FG3_PCT','FTA','FTM','STL','BLK','TOV','HISTORICAL_PRIOR_USED','HISTORICAL_CURRENT_WEIGHT','HISTORICAL_MINUTES_WEIGHT','SHOOTING_INPUT_RECONCILED']),
        'pregame_team_inputs': select_rows(team_inputs),
        'limitations': 'Approximate paired possession slots, approximate quarter and bonus timing, no exact shot clock, no verified primary-defender matchups. Overtime restarts foul state. Minute and event accounting checks are structural only, not historical predictive accuracy. Reports aggregated events, not 150,000 raw traces.'
    }
    return json.dumps(report, indent=2, default=_safe, ensure_ascii=False)
