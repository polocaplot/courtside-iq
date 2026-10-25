"""Best-effort public NBA.com aggregate lineup and defender feeds.

No positional tracking, individual matchup attribution or official real-time guarantee.
Failed requests never replace the last successful cache. Fetches only when explicitly
requested or when the dashboard's opted-in periodic refresh is due.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import os
import re
import unicodedata
import pandas as pd
from lineup_engine import validate_lineups, LINEUP_COLUMNS
from defender_intelligence import validate_defenders, COLS

CACHE_DIR = Path.home() / '.courtsideiq' / 'advanced_feeds'
FEED_SCHEMA_VERSION = 6  # bump when parsing changes; stale failed results must be retried


def _paths(season: str, root: Path):
    safe = season.replace('/', '_')
    return root / f'lineups_{safe}.csv', root / f'defenders_{safe}.csv', root / f'feeds_{safe}.json'


def read_cache(season: str, root: Path = CACHE_DIR):
    lp, dp, mp = _paths(season, root)
    meta = {}
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except (ValueError, OSError):
            pass
    def read(path, validator, columns):
        if not path.exists():
            return pd.DataFrame(columns=list(columns))
        try:
            return validator(pd.read_csv(path))
        except (OSError, ValueError, KeyError):
            return pd.DataFrame(columns=list(columns))
    return read(lp, validate_lineups, LINEUP_COLUMNS), read(dp, validate_defenders, COLS), meta


def should_refresh(season: str, hours: int = 24, root: Path = CACHE_DIR):
    lineups, defenders, meta = read_cache(season, root)
    try:
        if defenders.empty and meta.get('feed_schema_version', 0) < FEED_SCHEMA_VERSION:
            return True
        then = datetime.fromisoformat(meta['checked_at'])
        # Missing defended-shot data should not be suppressed for 24 hours
        # just because a lineup request succeeded or a failed fetch was logged.
        retry_hours = min(hours, 1) if defenders.empty or lineups.empty else hours
        return datetime.now(timezone.utc) - then >= timedelta(hours=retry_hours)
    except (KeyError, ValueError, TypeError):
        return True


def parse_lineup_table(raw: pd.DataFrame, player_names: dict[str, str], team_names: dict[str, str]):
    """Resolve lineup member IDs from NBA.com GROUP_ID, avoiding fragile name splits."""
    if raw.empty:
        return pd.DataFrame(columns=list(LINEUP_COLUMNS))
    result = []
    for _, row in raw.iterrows():
        group_id = str(row.get('GROUP_ID', '')).strip('-')
        ids = [part for part in group_id.split('-') if part]
        if len(ids) != 5 or len(set(ids)) != 5:
            continue
        players = [player_names.get(i) for i in ids]
        if any(not p for p in players):
            continue
        tid = str(row.get('TEAM_ID', ''))
        team = str(row.get('TEAM_ABBREVIATION', '') or team_names.get(tid, '')).strip().upper()
        if team == 'NAN' or not team:
            team = team_names.get(tid, '')
        if not team:
            continue
        # MIN is an observed playing-time proxy, not possession counts.
        # Label as such in feed metadata; the lineup sampler only needs weights.
        weight = pd.to_numeric(row.get('POSS', row.get('MIN', 0)), errors='coerce')
        if pd.isna(weight) or float(weight) <= 0:
            continue
        result.append([team, *players, float(weight)])
    if not result:
        return pd.DataFrame(columns=list(LINEUP_COLUMNS))
    return validate_lineups(pd.DataFrame(result, columns=list(LINEUP_COLUMNS)))


def _normalized_player_name(value: object) -> str:
    """Match NBA.com spelling variants without guessing a player's identity."""
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]', '', value.casefold())


def _player_id(value: object) -> str:
    try:
        return str(int(float(value)))
    except (ValueError, TypeError, OverflowError):
        return ''


def _roster_identity(player_stats: pd.DataFrame):
    """Construct both ID and unambiguous exact-name current-roster identities."""
    by_id, by_name, duplicates = {}, {}, set()
    for _, row in player_stats.iterrows():
        name = str(row.get('PLAYER_NAME', '')).strip()
        team = str(row.get('TEAM_ABBREVIATION', '')).strip().upper()
        if not name or name.casefold() == 'nan' or not team or team == 'NAN':
            continue
        pid = _player_id(row.get('PLAYER_ID'))
        if pid:
            by_id[pid] = (team, name)
        key = _normalized_player_name(name)
        if key in by_name and by_name[key][0] != team:
            duplicates.add(key)
        else:
            by_name[key] = (team, name)
    for key in duplicates:
        by_name.pop(key, None)
    return by_id, by_name


def parse_defender_tables(rim: pd.DataFrame, three: pd.DataFrame,
                          team_names: dict[str, str],
                          roster_teams: dict[str, str] | None = None,
                          season_weight: float = 1.0,
                          roster_names: dict[str, tuple[str, str]] | None = None,
                          allow_official_team_fallback: bool = False):
    """Resolve official LeagueDashPtDefend fields to current roster teams.

    NBA.com uses PLAYER_LAST_TEAM_ABBREVIATION and CLOSE_DEF_PERSON_ID.
    Historical records are remapped by player ID to the CURRENT roster.
    The NBA '3 Pointers' feed does not distinguish corners from above-break:
    store the aggregate in one proxy bucket, NEVER duplicate its attempts.
    Counts are scaled by season_weight before Bayesian shrinkage.
    """
    rows = []
    roster_teams = roster_teams or {}
    roster_names = roster_names or {}
    for frame, zones, attempt_columns, make_columns in [
        (rim, ['Rim'], ('FGA_LT_06', 'D_FGA', 'DEFENDED_FGA'),
         ('FGM_LT_06', 'D_FGM', 'DEFENDED_FGM')),
        (three, ['Above-break 3'], ('FG3A', 'D_FGA', 'DEFENDED_FGA'),
         ('FG3M', 'D_FGM', 'DEFENDED_FGM'))]:
        for _, row in frame.iterrows():
            name = str(row.get('PLAYER_NAME', '')).strip()
            player_id = _player_id(row.get('CLOSE_DEF_PERSON_ID', row.get('PLAYER_ID')))
            identity = None
            if player_id in roster_teams:
                identity = (roster_teams[player_id], name)
            elif roster_names:
                # Some NBA.com data products return IDs that cannot be matched to
                # the supplied season table. Only use a UNIQUE exact name match.
                identity = roster_names.get(_normalized_player_name(name))
            if identity:
                team, name = identity
            elif roster_teams or roster_names:
                if not allow_official_team_fallback:
                    continue  # Historical rows must map to current team.
                # Current-season official defender rows are still genuine observations,
                # even if the separate roster endpoint omits a player. Never use this
                # fallback for prior seasons (a trade would misattribute the player).
                team = str(row.get('PLAYER_LAST_TEAM_ABBREVIATION', row.get('TEAM_ABBREVIATION', ''))).strip().upper()
                if not team or team == 'NAN':
                    team = team_names.get(_player_id(row.get('PLAYER_LAST_TEAM_ID', row.get('TEAM_ID'))), '')
                if team not in set(team_names.values()):
                    continue
            else:
                team = str(row.get('PLAYER_LAST_TEAM_ABBREVIATION', row.get('TEAM_ABBREVIATION', ''))).strip().upper()
                if not team or team == 'NAN':
                    team = team_names.get(_player_id(row.get('PLAYER_LAST_TEAM_ID', row.get('TEAM_ID'))), '')
            # NBA.com's LeagueDashPtDefend uses category-specific defended-shot
            # columns (FGA_LT_06/FGM_LT_06 and FG3A/FG3M). These are
            # observed shots allowed, NOT the defender's offensive shooting.
            # Older API schemas use D_FGA/D_FGM or DEFENDED_* aliases.
            def first_numeric(columns):
                for column in columns:
                    if column in row.index:
                        value = pd.to_numeric(row[column], errors='coerce')
                        if pd.notna(value):
                            return value
                return float('nan')
            attempts = first_numeric(attempt_columns)
            makes = first_numeric(make_columns)
            if not team or not name or pd.isna(attempts) or pd.isna(makes) or attempts < 0 or makes < 0 or makes > attempts:
                continue
            for zone in zones:
                rows.append([team, name, zone, float(attempts) * season_weight,
                             float(makes) * season_weight])
    if not rows:
        return pd.DataFrame(columns=list(COLS))
    return validate_defenders(pd.DataFrame(rows, columns=list(COLS)))


def previous_season(season: str) -> str:
    year = int(season.split('-')[0]) - 1
    return f'{year}-{(year + 1) % 100:02d}'


def historical_defender_weight(frames: list[pd.DataFrame], prior_games: float = 12.0) -> float:
    """Scale full previous-season counts to a 12-game equivalent prior.

    This prevents last season's 82-game total from overwhelming this season,
    while keeping early-season estimates stable. GP is from NBA's defender feed.
    """
    gps = []
    for frame in frames:
        if 'GP' in frame:
            vals = pd.to_numeric(frame['GP'], errors='coerce').dropna()
            gps.extend(float(v) for v in vals if v > 0)
    if not gps:
        return 0.0
    return min(1.0, prior_games / max(1.0, float(pd.Series(gps).median())))


def fetch_public_feeds(season: str, player_stats: pd.DataFrame, team_stats: pd.DataFrame,
                       root: Path = CACHE_DIR) -> dict:
    """Try independent endpoints; persist each only on a validated successful result."""
    from nba_api.stats.endpoints import leaguedashlineups, leaguedashptdefend
    root.mkdir(parents=True, exist_ok=True)
    lp, dp, mp = _paths(season, root)
    player_names = {_player_id(r.get('PLAYER_ID')): str(r['PLAYER_NAME']) for _, r in player_stats.iterrows()
                    if _player_id(r.get('PLAYER_ID')) and pd.notna(r.get('PLAYER_NAME'))}
    team_names = {str(int(r['TEAM_ID'])): str(r['TEAM_ABBREVIATION']) for _, r in team_stats.iterrows()
                  if pd.notna(r.get('TEAM_ID')) and pd.notna(r.get('TEAM_ABBREVIATION'))}
    status = {'checked_at': datetime.now(timezone.utc).isoformat(), 'season': season,
              'feed_schema_version': FEED_SCHEMA_VERSION,
              'lineup_source': 'NBA.com LeagueDashLineups, possessions or minutes as frequency weights',
              'defender_source': 'NBA.com LeagueDashPtDefend, <6 ft and 3-point buckets (zone proxies)',
              'errors': []}
    try:
        raw = leaguedashlineups.LeagueDashLineups(season=season, group_quantity=5,
            season_type_all_star='Regular Season', measure_type_detailed_defense='Base',
            per_mode_detailed='Totals', timeout=35).get_data_frames()[0]
        parsed = parse_lineup_table(raw, player_names, team_names)
        if parsed.empty:
            raise ValueError('No resolvable five-player lineups returned')
        parsed.to_csv(lp, index=False)
        status['lineups'] = len(parsed)
    except Exception as exc:
        status['errors'].append(f'Lineups: {type(exc).__name__}: {str(exc)[:140]}')
    try:
        roster_by_id, roster_by_name = _roster_identity(player_stats)
        roster_teams = {pid: team for pid, (team, _) in roster_by_id.items()}
        status['defender_roster_ids'] = len(roster_by_id)
        status['defender_roster_names'] = len(roster_by_name)
        # Query independently: a failed 3PT endpoint must not discard rim data.
        # Do not substitute invented tracking records for unavailable NBA feeds.
        def get_defended(year: str):
            frames = {}
            for category, label in [('Less Than 6Ft', 'rim'), ('3 Pointers', 'three')]:
                try:
                    frame = leaguedashptdefend.LeagueDashPtDefend(
                        season=year, season_type_all_star='Regular Season',
                        defense_category=category, league_id='00',
                        per_mode_simple='Totals', timeout=35).get_data_frames()[0]
                    if not frame.empty:
                        frames[label] = frame
                    else:
                        status['errors'].append(f'{year} {label} defended shots: NBA.com returned an empty table')
                except Exception as exc:
                    status['errors'].append(f'{year} {label} defended shots: '
                        f'{type(exc).__name__}: {str(exc)[:110]}')
            return frames
        current = get_defended(season)
        prior_year = previous_season(season)
        prior = get_defended(prior_year)
        empty = pd.DataFrame()
        parts = []
        status['defender_raw_columns'] = {
            'current': {key: list(frame.columns) for key, frame in current.items()},
            'prior': {key: list(frame.columns) for key, frame in prior.items()},
        }
        status['defender_raw_rows'] = {
            'current': {key: len(frame) for key, frame in current.items()},
            'prior': {key: len(frame) for key, frame in prior.items()},
        }
        # Explicitly report when the feed returns players but none match the
        # supplied roster. This is not the same as a blocked/empty endpoint.
        status['defender_identity_matches'] = {}
        for label, frames in [('current', current), ('prior', prior)]:
            status['defender_identity_matches'][label] = {}
            for category, frame in frames.items():
                ids = sum(_player_id(row.get('CLOSE_DEF_PERSON_ID')) in roster_teams
                          for _, row in frame.iterrows())
                names = sum(_normalized_player_name(row.get('PLAYER_NAME')) in roster_by_name
                            for _, row in frame.iterrows())
                status['defender_identity_matches'][label][category] = {'by_id': ids, 'by_name': names}
        if current:
            parts.append(parse_defender_tables(current.get('rim', empty),
                current.get('three', empty), team_names, roster_teams=roster_teams,
                roster_names=roster_by_name, allow_official_team_fallback=True))
        if prior:
            weight = historical_defender_weight(list(prior.values()))
            status['defender_prior_weight'] = weight
            status['defender_prior_season'] = prior_year
            if weight:
                parts.append(parse_defender_tables(prior.get('rim', empty),
                    prior.get('three', empty), team_names,
                    roster_teams=roster_teams, roster_names=roster_by_name,
                    season_weight=weight))
        status['defender_categories'] = {
            'current': sorted(current), 'prior': sorted(prior)}
        parts = [part for part in parts if not part.empty]
        if not parts:
            counts = status['defender_raw_rows']
            matches = status['defender_identity_matches']
            schemas = status['defender_raw_columns']
            # Include representative counts only; player names and IDs are unnecessary.
            samples = {}
            for label, frames in [('current', current), ('prior', prior)]:
                samples[label] = {}
                for category, frame in frames.items():
                    if not frame.empty:
                        row = frame.iloc[0]
                        samples[label][category] = {
                            col: str(row.get(col))[:40] for col in
                            ('FGA_LT_06', 'FGM_LT_06', 'FG3A', 'FG3M', 'D_FGA', 'D_FGM', 'DEFENDED_FGA', 'DEFENDED_FGM')
                            if col in frame.columns}
            status['defender_count_samples'] = samples
            raise ValueError(f'No usable defended-shot aggregates. Raw rows: {counts}; '
                             f'roster matches: {matches}; columns: {schemas}; '
                             f'count samples: {samples}.')
        parsed = validate_defenders(pd.concat(parts, ignore_index=True))
        parsed.to_csv(dp, index=False)
        status['defenders'] = len(parsed)
        status['defender_parsed_teams'] = parsed['TEAM_ABBREVIATION'].value_counts().to_dict()
        status['defender_data_kind'] = 'observed_aggregate_zone_proxy_not_matchup_tracking'
        status['defender_source'] = ('NBA.com LeagueDashPtDefend, current and '
            'previous-season 12-game-equivalent Bayesian prior; <6ft and aggregate 3PT proxies (no corner split)')
        status['defender_historical_fallback'] = bool(prior and not current)
    except Exception as exc:
        status['defenders'] = 0
        status['defender_data_kind'] = 'unavailable_no_observed_defender_data'
        status['errors'].append(f'Defense: {type(exc).__name__}: {str(exc)[:2400]}')
    # Save the attempted check, but keep successful data files even when another feed fails.
    tmp = mp.with_suffix('.tmp')
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, mp)
    return status
