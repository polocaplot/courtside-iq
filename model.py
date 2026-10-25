"""Fast, seeded, correlated NBA box score Monte Carlo engine.

This is a statistical estimator, NOT a trained proprietary sportsbook model.
The baseline mode simulates pace/score environments and coherent player allocations.
The optional possession-event mode simulates approximate possession events, not
exact lineup/defender/shot-clock tracked play-by-play. Use walk-forward validation before
trusting edge estimates. All inputs must be tagged as live, CSV, or synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from data_sources import normalize_players, normalize_teams, normalize_name
from shot_intelligence import player_zone_model
from possession_engine import simulate_chunk

STATS = ("MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV")
SHOT_STATS = ("FGA", "FGM", "3PA", "FTA", "FTM", "OREB", "DREB", "PF")
STATUS_PROB = {"Healthy": 1., "Available": 1., "Probable": .92, "Questionable": .55,
               "Doubtful": .14, "Out": 0., "Unclear": .75}


@dataclass
class Config:
    iterations: int = 150_000
    seed: int = 42
    home_advantage: float = 2.2
    pace_adjustment: float = 0.
    volatility: float = 1.
    score_sd: float = 11.7
    minutes_sd: float = 2.7
    injury_overrides: dict[str, str] = field(default_factory=dict)
    minutes_overrides: dict[str, float] = field(default_factory=dict)
    usage_overrides: dict[str, float] = field(default_factory=dict)
    possession_mode: bool = False
    lineup_priors: pd.DataFrame | None = None
    defender_profiles: pd.DataFrame | None = None
    rotation_interval: int = 8
    calibrate_possession: bool = True
    shot_mode: bool = False
    shot_profiles: pd.DataFrame | None = None
    defense_profiles: pd.DataFrame | None = None
    defense_strength: float = 1.0
    defensive_events_strength: float = 1.0
    # Optional observed pregame inputs; do not infer confirmed starters from season rank.
    confirmed_starters: dict[str, list[str]] = field(default_factory=dict)
    strict_input_validation: bool = True


def _minute_prior(roster: pd.DataFrame) -> np.ndarray:
    """Use observed recent games if supplied, with shrinkage for small samples.

    Supported optional columns: MIN_L5, MIN_L10, GAMES_PLAYED, MINUTES_LIMIT.
    Never invent recent-game observations when the data feed omits them.
    """
    season = np.clip(pd.to_numeric(roster['MIN'], errors='coerce').to_numpy(float), 0, 48)
    result = season.copy()
    games = (pd.to_numeric(roster['GAMES_PLAYED'], errors='coerce').fillna(0).to_numpy(float)
             if 'GAMES_PLAYED' in roster else np.zeros(len(roster)))
    # L5 is a subset of L10. Blend their combined signal once rather than
    # treating the two overlapping samples as independent evidence.
    last5 = (pd.to_numeric(roster['MIN_L5'], errors='coerce').to_numpy(float)
             if 'MIN_L5' in roster else np.full(len(roster), np.nan))
    last10 = (pd.to_numeric(roster['MIN_L10'], errors='coerce').to_numpy(float)
              if 'MIN_L10' in roster else np.full(len(roster), np.nan))
    good5 = np.isfinite(last5) & (last5 >= 0) & (last5 <= 48)
    good10 = np.isfinite(last10) & (last10 >= 0) & (last10 <= 48)
    recent = np.where(good5 & good10, .65*last5 + .35*last10,
                      np.where(good5, last5, last10))
    valid = good5 | good10
    # New-team rotations can change immediately. Give real current minutes
    # meaningful weight even after 1-2 games, while retaining a prior.
    alpha = np.where(good5 & good10,
                     np.minimum(.65, .22 + .48 * games / (games + 5.)),
                     np.minimum(.40, .15 + .35 * games / (games + 5.))) * valid
    result = np.where(valid, (1-alpha)*season + alpha*recent, season)
    return result


def _input_warnings(rosters: dict, cfg: Config) -> list[str]:
    warnings = []
    for team, roster in rosters.items():
        for col in ('MIN', 'FGA', 'PTS', 'REB', 'AST'):
            if col not in roster:
                warnings.append(f'{team}: missing {col}')
                continue
            values = pd.to_numeric(roster[col], errors='coerce').to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                warnings.append(f'{team}: invalid {col} inputs')
        if not any(col in roster and pd.to_numeric(roster[col], errors='coerce').notna().any() for col in ('MIN_L5', 'MIN_L10')):
            warnings.append(f'{team}: recent player minutes unavailable; using season baseline')
        if 'GAMES_PLAYED' in roster and pd.to_numeric(roster['GAMES_PLAYED'], errors='coerce').fillna(0).max() < 5:
            warnings.append(f'{team}: fewer than five current-season games; rotation/usage estimates are uncertain')
        starters = cfg.confirmed_starters.get(team, [])
        if not starters:
            warnings.append(f'{team}: starters not confirmed; inferred from minutes')
        elif len(starters) != 5 or any(x not in set(roster['PLAYER_NAME']) for x in starters):
            warnings.append(f'{team}: confirmed starter list invalid or incomplete')
        if sum(float(x) for x in roster['MIN'].to_numpy(float)) > 285:
            warnings.append(f'{team}: roster minutes exceed 285; rotation normalization required')
    return warnings


def _player_list(players: pd.DataFrame, team: str) -> pd.DataFrame:
    roster = players[players["TEAM_ABBREVIATION"] == team].copy()
    if roster.empty:
        raise ValueError(f"No player data for {team}. Select different teams or load the current season.")
    # Rank by the effective current-role estimate, not stale season minutes.
    # Retain enough players to represent a genuine 11-man rotation while
    # keeping the possession engine computationally bounded.
    roster['_ROLE_MIN'] = _minute_prior(roster)
    roster = (roster.sort_values('_ROLE_MIN', ascending=False)
              .head(15).drop(columns='_ROLE_MIN').reset_index(drop=True))
    if len(roster) < 5:
        raise ValueError(f"{team} has only {len(roster)} players in the dataset; at least five are required.")
    return roster


def _availability(roster: pd.DataFrame, cfg: Config) -> np.ndarray:
    return np.array([STATUS_PROB.get(cfg.injury_overrides.get(str(p), "Healthy"), .75)
                     for p in roster["PLAYER_NAME"]], dtype=float)


def _caps(roster: pd.DataFrame, cfg: Config) -> np.ndarray:
    return np.array([min(48., max(0., float(cfg.minutes_overrides.get(str(p), 48.))))
                     for p in roster["PLAYER_NAME"]], dtype=float)


def _minutes(rng, size, roster, cfg, lead_margin) -> tuple[np.ndarray, np.ndarray]:
    names = roster["PLAYER_NAME"].tolist()
    baseline = _minute_prior(roster)
    chances = _availability(roster, cfg)
    caps = _caps(roster, cfg)
    if np.sum(caps[chances > 0.0]) < 240 - 1e-8:
        raise ValueError("Fewer than 240 potential active player-minutes; reduce minute caps or injury exclusions.")
    active = rng.random((size, len(roster))) < chances[None, :]
    # If doubtful/questionable players happen to leave fewer than 5 active, temporarily
    # activate highest-probability eligible player to keep a legal rotation.
    low = np.flatnonzero(active.sum(axis=1) < 5)
    if len(low):
        for idx in np.argsort(-chances):
            if chances[idx] == 0:
                continue
            need = low[active[low].sum(axis=1) < 5]
            if not len(need):
                break
            active[need, idx] = True
    if np.any((active * caps[None, :]).sum(axis=1) < 240 - 1e-7):
        # Rare combined questionable-out outcomes: replace with the most likely
        # non-'Out' availability in the minimal subset necessary for 240 mins.
        bad = np.flatnonzero((active * caps).sum(axis=1) < 240 - 1e-7)
        for idx in np.argsort(-chances):
            if chances[idx] == 0:
                continue
            active[bad, idx] = True
            bad = bad[(active[bad] * caps).sum(axis=1) < 240 - 1e-7]
            if not len(bad):
                break
    if np.any((active * caps).sum(axis=1) < 240 - 1e-7):
        raise ValueError("This scenario cannot fill a legal 240 player-minutes. Relax restrictions.")
    # An NBA roster can have 13 eligible players but a typical game uses a
    # substantially shorter regular rotation. Giving every healthy reserve
    # positive minutes diluted starters (e.g. 33 baseline -> 25 projected).
    # Select a game-specific 9-10 man rotation using *current roster* minutes,
    # with uncertainty for the fringe. This is NOT a confirmed lineup feed.
    # Availability and explicit minute caps remain authoritative.
    # Some teams legitimately use 11 meaningful players. A fixed nine-man
    # rotation was zeroing out healthy players with 9-12 minute priors.
    # Expand the rotation when at least 11 *eligible* players have meaningful
    # minute evidence; still allow a shorter rotation game-to-game.
    meaningful = int(np.sum((baseline >= 8.) & (chances > 0.) & (caps >= 8.)))
    if 11 <= meaningful <= 12:
        rotation_size = rng.choice((10, 11), size=size, p=(.30, .70))
    else:
        rotation_size = rng.choice((9, 10), size=size, p=(.65, .35))
    priority = baseline[None, :] + rng.normal(0., 2.5, (size, len(roster)))
    # A confirmed starting five outranks bench players for rotation selection.
    # This does not guarantee starter minutes: injury/cap rules still apply.
    team_code = str(roster['TEAM_ABBREVIATION'].iloc[0]) if 'TEAM_ABBREVIATION' in roster else ''
    confirmed = set(cfg.confirmed_starters.get(team_code, []))
    if len(confirmed) == 5:
        priority += np.array([45. if x in confirmed else 0. for x in names])[None, :]
    priority = np.where(active & (caps[None, :] > 0), priority, -1e9)
    ordering = np.argsort(-priority, axis=1)
    rank = np.empty_like(ordering)
    rank[np.arange(size)[:, None], ordering] = np.arange(len(roster))[None, :]
    rotation = active & (rank < rotation_size[:, None]) & (caps[None, :] > 0)
    # If unusual injuries/caps make the shortened rotation infeasible, bring
    # the next eligible bench players in until exactly 240 minutes are possible.
    available_capacity = (rotation * caps[None, :]).sum(axis=1)
    for slot in range(10, len(roster) + 1):
        need = available_capacity < 240 - 1e-7
        if not np.any(need):
            break
        extra = need[:, None] & active & (rank == slot - 1) & (caps[None, :] > 0)
        rotation |= extra
        available_capacity = (rotation * caps[None, :]).sum(axis=1)
    if np.any(available_capacity < 240 - 1e-7):
        raise ValueError("Insufficient active rotation minutes after injury and cap adjustments")
    active = rotation
    # First five are rotation starters by season minutes, not confirmed starting lineup.
    margin_loss = np.clip(-lead_margin - 14., 0, 23) / 23
    baseline_2d = baseline[None, :] + rng.normal(0, cfg.minutes_sd * cfg.volatility,
                                                  (size, len(roster)))
    starter_mask = np.array([x in confirmed for x in names], dtype=bool) if len(confirmed) == 5 else np.arange(len(roster)) < 5
    baseline_2d[:, starter_mask] -= margin_loss[:, None] * 3.
    baseline_2d[:, ~starter_mask] += margin_loss[:, None] * 1.5
    # Player-specific minute overrides are CAPS, not target minutes.
    desired = np.clip(baseline_2d, .15, None) * active
    # Constrained additive allocation rather than proportional scaling.
    # Proportional normalization (e.g. 33 * 240 / 310 = 25.5) systematically
    # underprojects starters when season rosters include too many bench minutes.
    # Preserve high-minute roles, absorbing surplus primarily at the bench.
    # Each simulated team has exactly 240 regulation player-minutes.
    room = caps[None, :] * active
    desired = np.minimum(desired, room)
    baseline_sum = desired.sum(axis=1)
    over = baseline_sum > 240.
    # Bench players have greater flexibility when minutes must be removed;
    # established high-minute players absorb more of a shortage.
    flex_down = np.clip((37. - baseline)[None, :] / 13., .18, 2.2) * active
    flex_up = np.clip(baseline[None, :] / 25., .45, 1.65) * active
    flex = np.where(over[:, None], flex_down, flex_up)
    # Solve sum(clip(desired + lambda * flex, 0, cap)) = 240
    # by monotone bisection. A feasible roster is checked above.
    lo = np.full(size, -240., dtype=float)
    hi = np.full(size, 240., dtype=float)
    for _ in range(38):
        mid = (lo + hi) * .5
        allocation = np.clip(desired + mid[:, None] * flex, 0., room)
        too_many = allocation.sum(axis=1) > 240.
        hi = np.where(too_many, mid, hi)
        lo = np.where(too_many, lo, mid)
    result = np.clip(desired + ((lo + hi) * .5)[:, None] * flex, 0., room)
    # Numerical residual only (typically < 1e-8).
    residual = 240. - result.sum(axis=1)
    if np.max(np.abs(residual)) > .001:
        raise ValueError('Minute allocation failed to conserve team minutes')
    return result, active


def _allocate(rng: np.random.Generator, total: np.ndarray, weights: np.ndarray,
              concentration: float = 1.) -> np.ndarray:
    """Gamma-dirichlet heterogeneity + sequential binomial allocates EXACT total."""
    b, n = weights.shape
    shape = np.maximum(weights * concentration, .03)
    noise = rng.gamma(shape, 1., size=(b, n))
    noise = np.where(weights > 0, noise, 0.)
    denom = noise.sum(axis=1, keepdims=True)
    prob = np.divide(noise, denom, out=np.zeros_like(noise), where=denom > 0)
    # If no scoring weights, fallback equal among eligible players.
    bad = (denom[:, 0] < 1e-12)
    if bad.any():
        valid = weights[bad] > 0
        prob[bad] = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1)
    allocated = np.zeros((b, n), dtype=np.int16)
    rem = total.astype(np.int32).copy()
    rem_prob = np.ones(b, dtype=float)
    for j in range(n - 1):
        chance = np.clip(np.divide(prob[:, j], rem_prob,
                                   out=np.zeros(b), where=rem_prob > 1e-10), 0, 1)
        val = rng.binomial(rem, chance).astype(np.int16)
        allocated[:, j] = val
        rem -= val
        rem_prob -= prob[:, j]
    allocated[:, -1] = rem
    return allocated


def _team_chunk(rng, n, roster, cfg, score, margin, env, opponent_points, team: str, opponent: str) -> tuple[dict, np.ndarray]:
    minutes, active = _minutes(rng, n, roster, cfg, margin)
    base_min = np.maximum(1., roster["MIN"].to_numpy(float))
    ratio = minutes / base_min[None, :]
    pg = roster["PTS"].to_numpy(float)
    usage = np.array([float(cfg.usage_overrides.get(name, 1.))
                      for name in roster["PLAYER_NAME"].tolist()])
    usage = np.clip(usage, .25, 2.5)
    injury_mass = (pg[None, :] * (1 - active)).sum(axis=1)
    scale = 1 + np.clip(injury_mass / max(pg.sum(), 1) * .22, 0, .15)
    score_weights = np.maximum(.001, pg[None, :] * ratio * usage[None, :] * scale[:, None]) * active
    # Rotational offense penalty, relative to baseline; this also influences game scores.
    baseline = float(np.sum(pg))
    projected = score_weights.sum(axis=1)
    offense_shift = np.clip(.22 * (projected - baseline), -9, 7).astype(float)
    new_score = np.clip(np.rint(score + offense_shift), 70, 180).astype(np.int32)
    scores = _allocate(rng, new_score, score_weights, concentration=.72)
    shot_threes = None
    if cfg.shot_mode:
        # Vectorized per-player, per-zone shot attempts and binomial makes.
        # Shot attempts/makes are simulated, but not ordered possession by possession.
        shot_points = np.zeros((n, len(roster)), dtype=np.int32)
        shot_threes = np.zeros_like(shot_points)
        for j, (_, player) in enumerate(roster.iterrows()):
            model = player_zone_model(str(player['PLAYER_NAME']), team, cfg.shot_profiles,
                                      cfg.defense_profiles, opponent, cfg.defense_strength)
            # Actual season FGA/FTA drive opportunity volume; shot charts drive location mix.
            fga = float(player['FGA'])
            fta = float(player['FTA'])
            if model['sample_attempts'] < 30:
                # Missing chart data: a conservative, explicitly approximate
                # calibration to the player's measured scoring baseline.
                inferred_fga = max(0., (float(player['PTS']) - .77 * fta) /
                                   max(.5, model['points_per_attempt']))
                fga = max(fga, min(inferred_fga, fga * 1.35))
            attempt_rate = np.clip(fga * ratio[:, j] * usage[j] * env, 0, 42)
            for z in range(5):
                attempts = rng.poisson(attempt_rate * model['shares'][z])
                made = rng.binomial(attempts, float(model['make_prob'][z]))
                shot_points[:, j] += made * (3 if z >= 3 else 2)
                if z >= 3:
                    shot_threes[:, j] += made
            ft_attempts = rng.poisson(np.clip(fta * ratio[:, j] * usage[j] * env, 0, 25))
            # 0.77 is a prior, not player-specific FT% without validated FTM/FTA data.
            shot_points[:, j] += rng.binomial(ft_attempts, .77)
        scores = np.minimum(shot_points, 300).astype(np.int16)
        new_score = scores.astype(np.int32).sum(axis=1)

    # Real possessions generally create rebound opportunities from missed FG. Approximate
    # opponent missed-shot environment via score + pace; cap team totals realistically.
    reb_mean = np.clip(43.5 * env + (114 - opponent_points) * .045, 29, 59)
    rebounds = rng.poisson(reb_mean).clip(25, 65).astype(np.int32)
    assist_mean = np.clip(25.5 * (new_score / 114.) + .8 * (env - 1), 12, 40)
    assists = rng.poisson(assist_mean).clip(9, 45).astype(np.int32)
    # Assists <= approximate team made field goals.
    assists = np.minimum(assists, np.floor(new_score / 2.05).astype(np.int32))
    out = {"MIN": minutes.astype(np.float32), "PTS": scores}
    for stat, totals, conc in [("REB", rebounds, 1.6), ("AST", assists, 1.35),
                               ("STL", rng.poisson(8.0 * env).astype(np.int32), 1.6),
                               ("BLK", rng.poisson(4.9 * env).astype(np.int32), 1.5),
                               ("TOV", rng.poisson(13.8 * env).astype(np.int32), 2.0)]:
        weights = np.maximum(.02, roster[stat].to_numpy(float)[None, :] * ratio) * active
        out[stat] = _allocate(rng, totals, weights, conc)
    # Threes are constrained by individual points, and moderately correlated with points.
    # Not a reconstructed play-by-play shooting line.
    threes_weights = np.maximum(.01, roster["FG3M"].to_numpy(float)[None, :] * ratio) * active
    expected_threes = threes_weights.sum(axis=1)
    team_3pm = rng.poisson(np.clip(expected_threes * env, .01, 29)).astype(np.int32)
    team_3pm = np.minimum(team_3pm, new_score // 3)
    if shot_threes is None:
        candidate = _allocate(rng, team_3pm, threes_weights, 1.15)
        out["3PM"] = np.minimum(candidate, scores // 3).astype(np.int16)
    else:
        out["3PM"] = shot_threes.astype(np.int16)
    return out, new_score


def run(team_data: pd.DataFrame, player_data: pd.DataFrame,
        home: str, away: str, cfg: Config) -> dict:
    if cfg.iterations < 100 or cfg.iterations > 300_000:
        raise ValueError("Iterations must be between 100 and 300,000.")
    if home == away:
        raise ValueError("Home and away teams must be different.")
    tdf, pdf = normalize_teams(team_data), normalize_players(player_data)
    teams = tdf.set_index("TEAM_ABBREVIATION")
    if home not in teams.index or away not in teams.index:
        raise ValueError("The selected teams are missing from the loaded dataset.")
    rosters = {home: _player_list(pdf, home), away: _player_list(pdf, away)}
    input_warnings = _input_warnings(rosters, cfg)
    for team, roster in rosters.items():
        if "SHOOTING_INPUT_RECONCILED" in roster and roster["SHOOTING_INPUT_RECONCILED"].fillna(False).any():
            input_warnings.append(f"{team}: independently blended shooting rates were reconciled to valid box-score constraints")
    if cfg.possession_mode and (cfg.defender_profiles is None or cfg.defender_profiles.empty):
        input_warnings.append('Defender-tracking data unavailable; matchup effects are statistical estimates')
    if cfg.possession_mode and (cfg.shot_profiles is None or cfg.shot_profiles.empty):
        input_warnings.append('Shot-profile observations unavailable; using shot-zone priors')
    if cfg.strict_input_validation and any('invalid ' in w or 'missing ' in w for w in input_warnings):
        raise ValueError('Unreliable simulation inputs: ' + '; '.join(input_warnings))
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.iterations)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    all_stats = STATS + SHOT_STATS if cfg.possession_mode else STATS
    for team in [home, away]:
        arrays[team] = {s: np.empty((n, len(rosters[team])), dtype=np.float32 if s == "MIN" else np.int16)
                        for s in all_stats}
    home_score = np.empty(n, dtype=np.int16)
    away_score = np.empty(n, dtype=np.int16)
    pace = np.empty(n, dtype=np.float32)
    possession_audits = {home: {}, away: {}} if cfg.possession_mode else None
    quarter_points = {home: np.zeros((n, 4), dtype=np.int16),
                      away: np.zeros((n, 4), dtype=np.int16)} if cfg.possession_mode else None
    quarter_player_sums = ({team: {stat: np.zeros((4, len(rosters[team])), dtype=np.float64)
                                    for stat in all_stats}
                            for team in (home, away)} if cfg.possession_mode else None)
    quarter_event_sums = {home: {}, away: {}} if cfg.possession_mode else None
    ph, pa = float(teams.loc[home, "PACE"]), float(teams.loc[away, "PACE"])
    pace_base = max(85., (ph + pa) / 2 + cfg.pace_adjustment)
    hp = float(teams.loc[home, "PTS"])
    ap = float(teams.loc[away, "PTS"])
    ho = float(teams.loc[home, "OPP_PTS"])
    ao = float(teams.loc[away, "OPP_PTS"])
    hmean = (hp + ao) * .5 + cfg.home_advantage
    amean = (ap + ho) * .5
    efficiency_adjustments = {}
    calibration_report = None
    if cfg.possession_mode and cfg.calibrate_possession:
        # Pilot-based matchup calibration. Uses only pregame team scoring baselines;
        # NOT trained xFG or fitted to the actual game outcome. Its purpose is
        # preventing an event-prior mismatch from swamping team offense/defense.
        pilot_rng = np.random.default_rng(int(cfg.seed) + 2_000_001)
        pilot_n = 1200
        pilot_hm, _ = _minutes(pilot_rng, pilot_n, rosters[home], cfg, np.zeros(pilot_n))
        pilot_am, _ = _minutes(pilot_rng, pilot_n, rosters[away], cfg, np.zeros(pilot_n))
        pilot_poss = np.clip(pilot_rng.normal(pace_base, 2.5, pilot_n).round(), 78, 122).astype(int)
        pilot_blocks, pilot_audit = simulate_chunk(pilot_rng, pilot_n, rosters[home], rosters[away],
                                                    pilot_hm, pilot_am, home, away, cfg, pilot_poss)
        # A second independent pilot measures the *actual* nonlinear response
        # to adjusted shot probabilities. The previous FGA*2.25 shortcut
        # substantially underestimated the residual for some matchups.
        initial_means = {home: float(pilot_blocks[0]['PTS'].sum(axis=1).mean()),
                         away: float(pilot_blocks[1]['PTS'].sum(axis=1).mean())}
        targets = {home: hmean, away: amean}
        for team in (home, away):
            fga = float(pilot_audit[0 if team == home else 1]['FGA'].mean())
            efficiency_adjustments[team] = float(np.clip(
                (targets[team] - initial_means[team]) / max(fga * 2.0, 1.), -.06, .06))
        # Use common random numbers to stabilize the measured response.
        check_rng = np.random.default_rng(int(cfg.seed) + 2_000_002)
        check_blocks, _ = simulate_chunk(check_rng, pilot_n, rosters[home], rosters[away],
                                         pilot_hm, pilot_am, home, away, cfg, pilot_poss,
                                         efficiency_adjustments)
        check_means = {home: float(check_blocks[0]['PTS'].sum(axis=1).mean()),
                       away: float(check_blocks[1]['PTS'].sum(axis=1).mean())}
        for team in (home, away):
            fga = float(pilot_audit[0 if team == home else 1]['FGA'].mean())
            residual = targets[team] - check_means[team]
            efficiency_adjustments[team] = float(np.clip(
                efficiency_adjustments[team] + residual / max(fga * 2.0, 1.), -.06, .06))
        # Verify the final adjustment with a separate sample, without
        # pretending a pregame team baseline is a guaranteed game result.
        verify_rng = np.random.default_rng(int(cfg.seed) + 2_000_003)
        verify_blocks, _ = simulate_chunk(verify_rng, pilot_n, rosters[home], rosters[away],
                                          pilot_hm, pilot_am, home, away, cfg, pilot_poss,
                                          efficiency_adjustments)
        calibration_report = {}
        for team, block in ((home, verify_blocks[0]), (away, verify_blocks[1])):
            verified = float(block['PTS'].sum(axis=1).mean())
            residual = float(targets[team] - verified)
            calibration_report[team] = {
                'pilot_pts': round(initial_means[team], 2),
                'pregame_target': round(float(targets[team]), 2),
                'shot_probability_shift': round(efficiency_adjustments[team], 4),
                'uncalibrated_engine_gap_points': round(float(targets[team] - initial_means[team]), 2),
                'large_shooting_correction_warning': bool(abs(efficiency_adjustments[team]) >= .03),
                'unmatched_target_points': round(residual, 2),
                'calibration_policy': 'Two-pass observed pilot; bounded ±6 percentage point shot adjustment; residual not forced',
                'source': 'pregame team PTS and OPP_PTS baselines',
                'calibration_gap_warning': bool(abs(residual) > 2.0)}
            if abs(efficiency_adjustments[team]) >= .03:
                input_warnings.append(f'{team}: shot calibration shifts make probability by {efficiency_adjustments[team]:+.1%}; possession event rates require historical validation')
            if abs(residual) > 2.0:
                input_warnings.append(f'{team}: possession scoring baseline remains {residual:+.1f} PPG from target after calibration; not validated')
    for start in range(0, n, 4000):
        end = min(start + 4000, n)
        b = end - start
        p = np.clip(rng.normal(pace_base, 3.1 * cfg.volatility, b), 82, 118)
        env = p / max((ph + pa) / 2, 85)
        common = rng.normal(0, 3.2 * cfg.volatility, b)
        h = np.clip(np.rint(hmean * env + common + rng.normal(0, cfg.score_sd * cfg.volatility, b)), 70, 180)
        a = np.clip(np.rint(amean * env + common + rng.normal(0, cfg.score_sd * cfg.volatility, b)), 70, 180)
        if cfg.possession_mode:
            hm, _ = _minutes(rng, b, rosters[home], cfg, h-a)
            am, _ = _minutes(rng, b, rosters[away], cfg, a-h)
            # Possessions are shared across opponents; scoring emerges from events.
            poss = np.clip(np.rint(p + rng.normal(0, 2.0, b)), 78, 122).astype(int)
            blocks, audits, periods = simulate_chunk(rng, b, rosters[home], rosters[away],
                                             hm, am, home, away, cfg, poss, efficiency_adjustments,
                                             return_quarters=True)
            hd, ad = blocks
            for side, team in enumerate((home, away)):
                pdata = periods[side]
                quarter_points[team][start:end] = np.rint(
                    pdata['players']['PTS'].sum(axis=2).T).astype(np.int16)
                for stat in all_stats:
                    quarter_player_sums[team][stat] += pdata['players'][stat].sum(axis=1)
                for stat, values in pdata['events'].items():
                    if stat not in quarter_event_sums[team]:
                        quarter_event_sums[team][stat] = np.zeros(4, dtype=np.float64)
                    quarter_event_sums[team][stat] += values.sum(axis=1)
            hs = hd['PTS'].sum(axis=1).astype(np.int32)
            ass = ad['PTS'].sum(axis=1).astype(np.int32)
            for team, audit in [(home, audits[0]), (away, audits[1])]:
                for key, vals in audit.items():
                    if key not in possession_audits[team]:
                        possession_audits[team][key] = np.empty(n, dtype=np.int16)
                    possession_audits[team][key][start:end] = vals
        else:
            hd, hs = _team_chunk(rng, b, rosters[home], cfg, h, h-a, env, a, home, away)
            ad, ass = _team_chunk(rng, b, rosters[away], cfg, a, a-h, env, h, away, home)
        # Possession-mode overtime must generate REAL box-score events. The old
        # shortcut fabricated extra points, minutes, and broke PTS=2*FGM+3PM+FTM.
        # Model each extra period as ~5/48 of a game, including actual field
        # goals, free throws, assists and rebounds. Current limitations: this
        # approximation restarts foul/lineup state at the OT boundary.
        if cfg.possession_mode:
            tied = np.flatnonzero(hs == ass)
            for ot_period in range(6):
                if not len(tied):
                    break
                ot_poss = np.clip(np.rint(rng.normal(pace_base * 5 / 48, 1.2, len(tied))),
                                  7, 16).astype(int)
                ot_blocks, ot_audits = simulate_chunk(
                    rng, len(tied), rosters[home], rosters[away],
                    hm[tied], am[tied], home, away, cfg, ot_poss,
                    efficiency_adjustments)
                for side, block, scores, overtime_block, audit in (
                    (home, hd, hs, ot_blocks[0], ot_audits[0]),
                    (away, ad, ass, ot_blocks[1], ot_audits[1])):
                    for stat in all_stats:
                        if stat == 'MIN':
                            # simulate_chunk reports its slot counts on a
                            # 48-minute scale; OT slots represent five minutes.
                            block[stat][tied] += overtime_block[stat] * (5. / 48.)
                        else:
                            block[stat][tied] += overtime_block[stat]
                    scores[tied] += overtime_block['PTS'].sum(axis=1).astype(np.int32)
                    for key, vals in audit.items():
                        possession_audits[side][key][start + tied] += vals
                tied = tied[hs[tied] == ass[tied]]
            if len(tied):
                input_warnings.append(f'{len(tied)} simulations remain tied after six simulated overtimes')
        else:
            ties = np.flatnonzero(hs == ass)
            if len(ties):
                home_wins = rng.random(len(ties)) < .5
                for side, data, scores, winners in [(home, hd, hs, home_wins),
                                                      (away, ad, ass, ~home_wins)]:
                    extra = (rng.poisson(4, len(ties)) + 2 + 2 * winners).astype(np.int32)
                    scoreweights = data["PTS"][ties].astype(float) + data["MIN"][ties] / 8
                    add = _allocate(rng, extra, scoreweights)
                    data["PTS"][ties] += add
                    scores[ties] += extra
                    data["MIN"][ties] *= 53 / 48
        home_score[start:end] = hs
        away_score[start:end] = ass
        pace[start:end] = p
        for side, block in [(home, hd), (away, ad)]:
            for stat in all_stats:
                arrays[side][stat][start:end] = block[stat]
    # Per-player summaries and samples, for props and export.
    rows = []
    samples = {}
    for abbr in [away, home]:
        roster = rosters[abbr]
        for j, name in enumerate(roster["PLAYER_NAME"].tolist()):
            pk = f"{abbr}|{name}"
            samples[pk] = {s: arrays[abbr][s][:, j] for s in all_stats}
            samples[pk]["PRA"] = (samples[pk]["PTS"] + samples[pk]["REB"] + samples[pk]["AST"]).astype(np.int16)
            row = {"Player": name, "Team": abbr, "ID": pk,
                   "Status": cfg.injury_overrides.get(name, "Healthy")}
            for s in all_stats:
                values = samples[pk][s]
                row[s] = round(float(np.mean(values)), 1)
                p10, p90 = _integer_percentiles(values)
                row[f"{s}_P10"] = round(float(p10), 1)
                row[f"{s}_P90"] = round(float(p90), 1)
            row["PRA"] = round(float(np.mean(samples[pk]["PRA"])), 1)
            rows.append(row)
    box = pd.DataFrame(rows)
    quarter_summary = None
    if cfg.possession_mode:
        quarter_summary = {'source': 'same possession-event draws as final box scores',
                           'includes_overtime_in_quarters': False,
                           'teams': {}, 'overtime_games': 0}
        ot_counts = []
        for team, scores in ((home, home_score), (away, away_score)):
            qpts = quarter_points[team]
            ot_pts = scores.astype(np.int32) - qpts.sum(axis=1).astype(np.int32)
            ot_counts.append(ot_pts)
            qplayer = quarter_player_sums[team]
            quarter_summary['teams'][team] = {
                'quarter_points': qpts,
                'quarter_mean_points': [round(float(x), 3) for x in qpts.mean(axis=0)],
                'end_of_quarter_mean_score': [round(float(x), 3) for x in
                                              np.cumsum(qpts, axis=1).mean(axis=0)],
                'quarter_event_means': {stat: [round(float(x), 3) for x in values / n]
                                        for stat, values in quarter_event_sums[team].items()},
                'quarter_player_means': {stat: (values / n).round(3)
                                         for stat, values in qplayer.items()},
                'player_names': rosters[team]['PLAYER_NAME'].astype(str).tolist(),
                'overtime_mean_points_all_games': round(float(ot_pts.mean()), 3),
            }
        # An overtime game has positive overtime scoring on at least one side.
        # Zero-zero overtime periods are rare; do not use this as a definitive
        # count of OT games (track that separately if exact OT frequency needed).
    return {
        "home": home, "away": away, "config": cfg, "box": box, "samples": samples,
        "home_score": home_score, "away_score": away_score,
        "pace": pace, "home_win": float(np.mean(home_score > away_score)),
        "away_win": float(np.mean(away_score > home_score)),
        "home_mean": float(home_score.mean()), "away_mean": float(away_score.mean()),
        "total": home_score.astype(np.int32) + away_score.astype(np.int32),
        "margin": home_score.astype(np.int32) - away_score.astype(np.int32),
        "team_arrays": arrays,
        "possession_audits": possession_audits,
        "quarter_summary": quarter_summary,
        "possession_calibration": calibration_report,
        "precision_warnings": input_warnings,
        "precision_status": "requires_validation" if input_warnings else "input_checks_passed_not_backtested",
    }


def _integer_percentiles(values, probs=(.10, .90)):
    """Exact linear-interpolated percentiles for bounded discrete box stats.

    Histogram order statistics avoid hundreds of repeated large array sorts
    during a 150,000-game run. MIN is continuous and uses NumPy instead.
    """
    a = np.asarray(values)
    if a.size == 0:
        return (float('nan'),) * len(probs)
    if not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 2000:
        return tuple(np.quantile(a, probs).tolist())
    counts = np.bincount(a, minlength=int(a.max()) + 1)
    cum = np.cumsum(counts)
    n = len(a)
    out = []
    for q in probs:
        at = (n - 1) * q
        lo, hi = int(np.floor(at)), int(np.ceil(at))
        v0 = int(np.searchsorted(cum, lo + 1, side='left'))
        v1 = int(np.searchsorted(cum, hi + 1, side='left'))
        out.append(v0 + (at - lo) * (v1 - v0))
    return tuple(out)


def prop_probability(result: dict, player_id: str, market: str, line: float) -> dict:
    if player_id not in result["samples"]:
        raise ValueError("Unknown player")
    if market not in result["samples"][player_id]:
        raise ValueError("Unknown prop market")
    values = result["samples"][player_id][market]
    return {"over": float((values > line).mean()), "under": float((values < line).mean()),
            "push": float((values == line).mean()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p10": float(np.quantile(values, .1)),
            "p90": float(np.quantile(values, .9))}
