"""Experimental, vectorized *coupled* possession-event Monte Carlo.

Event order is approximate and parameter priors are heuristic. This is NOT optical
tracking, calibrated xFG, exact substitutions, or an official play-by-play simulator.
All player points come from simulated made shots and free throws; steals, blocks,
assists and rebounds arise from corresponding events rather than independent totals.
"""
from __future__ import annotations
import numpy as np
from shot_intelligence import player_zone_model
from lineup_engine import choose_lineups, choose_target_balanced_lineups
from defender_intelligence import defender_zone_adjustments
from defense_engine import defensive_rates, lineup_defense, zone_shares_after_defense, lineup_contest_modifier


def _choose(rng, weights):
    """Choose one player per simulated game, robust to zero-mass rows."""
    c = np.cumsum(weights, axis=1)
    tot = c[:, -1]
    fallback = np.where(tot > 0, tot, 1.)
    ix = (rng.random(len(weights))[:, None] * fallback[:, None] > c).sum(axis=1)
    return np.minimum(ix, weights.shape[1] - 1)


def _add(arr, games, indices, amount=1):
    if len(games):
        np.add.at(arr, (games, indices), amount)


def simulate_chunk(rng, n, home_roster, away_roster, home_minutes, away_minutes,
                   home: str, away: str, cfg, possessions: np.ndarray,
                   efficiency_adjustments: dict | None = None, return_quarters: bool = False):
    """Return team stat arrays, team scores and event audits for n paired games.

    Each team has a possession count sampled from the same pace environment. A
    missed shot creates a rebound; one offensive rebound can produce a putback.
    Five-player units are sampled at rotation intervals from projected minutes or
    optional observed lineup frequencies. These are approximate, not real substitutions.
    """
    rosters = [home_roster, away_roster]
    mins = [home_minutes, away_minutes]
    names = [home, away]
    stats = [{s: np.zeros((n, len(r)), dtype=np.int16) for s in
              ('PTS', 'REB', 'AST', '3PM', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', '3PA', 'FTA', 'FTM', 'OREB', 'DREB')} for r in rosters]
    audits = [{k: np.zeros(n, dtype=np.int16) for k in
               ('FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB', 'POSS',
                'PERSONAL_FOULS', 'SHOOTING_FOULS', 'NONSHOOTING_FOULS',
                'OFFENSIVE_FOULS', 'BONUS_TRIPS', 'FOUL_OUTS', 'NONSHOOTING_CONTINUATIONS')} for _ in rosters]
    # Persistent per-game state. A foul changes the next lineup, which changes
    # defensive pressure, shot selection, offensive usage, and actual minutes.
    personal_fouls = [np.zeros((n, len(r)), dtype=np.int8) for r in rosters]
    quarter_team_fouls = [np.zeros(n, dtype=np.int8) for _ in rosters]
    on_court_counts = [np.zeros((n, len(r)), dtype=np.int16) for r in rosters]
    refresh_lineups = [np.ones(n, dtype=bool) for _ in rosters]
    profiles = []
    shooters = []
    assist_w = []
    reb_w = []
    steal_w = []
    block_w = []
    turnover_w = []
    ft_prob = []
    foul_weights = []
    defender_modifiers = []
    defensive_profiles = []
    for t, r in enumerate(rosters):
        # Per-game PF where available; otherwise a deliberately conservative
        # role-based proxy. This is not observed foul-by-matchup evidence.
        pf = r['PF'].to_numpy(float) if 'PF' in r else np.full(len(r), np.nan)
        # Archive foul rates from earlier games inform foul assignment,
        # linking historical fouls -> rotations -> defense -> possessions.
        if 'ARCHIVE_FOULS_PER_36' in r:
            archived = r['ARCHIVE_FOULS_PER_36'].to_numpy(float) * np.maximum(
                r['MIN'].to_numpy(float), 0.) / 36.
            pf = np.where(np.isfinite(pf), pf, archived)
        pf = np.nan_to_num(pf, nan=2.5, posinf=2.5, neginf=2.5)
        foul_weights.append(np.clip(pf / np.maximum(r['MIN'].to_numpy(float), 8.), .035, .18))
        defender_modifiers.append(defender_zone_adjustments(r, names[t], getattr(cfg, "defender_profiles", None)))
        defensive_profiles.append(defensive_rates(r, mins[t], getattr(cfg, "defensive_events_strength", 1.0)))
        # On-court event selection must use per-minute rates. Multiplying by
        # projected minutes here *again* double-counted playing time, making
        # long-minute starters disproportionately likely to shoot, pass, rebound.
        # Shrink tiny-minute samples toward a conservative role prior.
        baseline = np.maximum(r['MIN'].to_numpy(float), 0.)
        exposure = np.maximum(baseline, 8.)
        usage = np.array([cfg.usage_overrides.get(str(x), 1.) for x in r['PLAYER_NAME']], dtype=float)
        usage = np.clip(usage, .25, 2.5)
        shooter_w = ((np.maximum(0., r['FGA'].to_numpy(float)) + .6) / (exposure + 2.))[None, :] * usage
        shooters.append(np.broadcast_to(shooter_w, mins[t].shape))
        def weight(col):
            per_min = (np.maximum(0., r[col].to_numpy(float)) + .12) / (exposure + 2.)
            return np.broadcast_to(per_min[None, :], mins[t].shape)
        assist_w.append(weight('AST'))
        reb_w.append(weight('REB'))
        steal_w.append(weight('STL'))
        block_w.append(weight('BLK'))
        turnover_w.append(weight('TOV'))
        # Player-specific FT% with a beta prior. Imported CSVs without FTM use
        # a 77% prior, while real NBA.com per-game FTM/FTA provide player signal.
        fta = r['FTA'].to_numpy(float)
        ftm = r['FTM'].to_numpy(float) if 'FTM' in r else .77 * fta
        ft_prob.append(np.clip((ftm * 20 + 20 * .77) / (fta * 20 + 20), .50, .97))
        models = [player_zone_model(str(row['PLAYER_NAME']), names[t], cfg.shot_profiles,
                                    cfg.defense_profiles, names[1-t], cfg.defense_strength,
                                    player_baseline=row.to_dict())
                  for _, row in r.iterrows()]
        zone_probs = np.stack([m['make_prob'] for m in models])
        if efficiency_adjustments:
            zone_probs = np.clip(zone_probs + float(efficiency_adjustments.get(names[t], 0.)), .15, .88)
        profiles.append((np.stack([m['shares'] for m in models]), zone_probs))

    # Both teams are advanced through paired possession slots. Quarter boundary
    # is approximated from each game's own pace (not a shot clock reconstruction).
    on_court = [None, None]
    rotation_interval = max(2, int(getattr(cfg, 'rotation_interval', 8)))
    lineup_priors = getattr(cfg, 'lineup_priors', None)
    max_poss = int(possessions.max())
    all_games = np.arange(n)
    previous_quarter = np.full(n, -1, dtype=np.int8)
    # Capture cumulative event counts at each game's own quarter boundaries.
    # Differences give exact quarter-level player/team stats from the SAME
    # simulated possessions, rather than a separate approximate forecast.
    quarter_cumulative = None
    if return_quarters:
        quarter_cumulative = []
        for t in (0, 1):
            quarter_cumulative.append({
                'players': {k: np.zeros((4, n, len(rosters[t])), dtype=np.float32)
                            for k in (*stats[t].keys(), 'MIN', 'PF')},
                'events': {k: np.zeros((4, n), dtype=np.int32) for k in audits[t]},
            })

    def capture_quarter(games, quarter_index):
        if quarter_cumulative is None or not len(games):
            return
        for side in (0, 1):
            dest = quarter_cumulative[side]
            for stat, values in stats[side].items():
                dest['players'][stat][quarter_index, games] = values[games]
            dest['players']['MIN'][quarter_index, games] = on_court_counts[side][games]
            dest['players']['PF'][quarter_index, games] = personal_fouls[side][games]
            for stat, values in audits[side].items():
                dest['events'][stat][quarter_index, games] = values[games]
    for step in range(max_poss):
        quarter = np.minimum(3, (4 * step // np.maximum(possessions, 1))).astype(np.int8)
        changed = quarter != previous_quarter
        if np.any(changed):
            # The previous period is complete before the new period's events.
            if return_quarters:
                for q in (1, 2, 3):
                    games_at_boundary = np.flatnonzero(changed & (quarter == q))
                    capture_quarter(games_at_boundary, q - 1)
            for t in (0, 1):
                quarter_team_fouls[t][changed] = 0
            previous_quarter = quarter
        if step % rotation_interval == 0:
            for t in (0, 1):
                refresh_lineups[t][:] = True
        for t in (0, 1):
            if not refresh_lineups[t].any():
                continue
            # Foul trouble affects coaching substitution preferences; six fouls
            # disqualify a player when at least five other players are available.
            # Remaining target minutes adapt to the actual minutes already played.
            # This connects the pregame rotation forecast to live substitutions.
            remaining_target = np.maximum(
                mins[t] * (possessions / 48.)[:, None] - on_court_counts[t], 0.)
            weights = remaining_target * np.where(personal_fouls[t] >= 5, .015,
                             np.where(personal_fouls[t] == 4, .24,
                             np.where(personal_fouls[t] == 3, .55, 1.)))
            # Fourth-quarter blowouts change substitution choices for BOTH
            # teams, rather than just shifting a post-hoc minutes number.
            if step:
                current_score = [stats[z]['PTS'].sum(axis=1) for z in (0, 1)]
                blowout = (quarter >= 3) & (np.abs(current_score[0] - current_score[1]) >= 20)
                if np.any(blowout):
                    weights[blowout, :5] *= .40
                    weights[blowout, 5:] *= 1.65
            disqualified = personal_fouls[t] >= 6
            can_exclude = ((mins[t] > 0) & ~disqualified).sum(axis=1) >= 5
            weights = np.where(disqualified & can_exclude[:, None], 0., weights)
            # No eligible five-man unit after disqualifications is an explicitly
            # unsupported edge case; fallback to least-fouled eligible players.
            insufficient = (weights > 0).sum(axis=1) < 5
            if np.any(insufficient):
                weights[insufficient] = mins[t][insufficient] * np.where(
                    disqualified[insufficient], .001, 1.)
            needs = np.flatnonzero(refresh_lineups[t] & (possessions > step))
            if not len(needs):
                refresh_lineups[t][:] = False
                continue
            # Select the next five based on how far each player's *actual*
            # on-court minutes lag their game-specific minute budget. Historical
            # lineup priors remain available as a modest continuity preference,
            # but cannot override the player's remaining minute capacity.
            chosen = choose_target_balanced_lineups(
                rng, mins[t][needs], on_court_counts[t][needs],
                possessions[needs], step, rotation_interval,
                on_court[t][needs] if on_court[t] is not None else None,
                weights[needs],
                rosters[t]['PLAYER_NAME'].astype(str).tolist(), names[t], lineup_priors)
            if on_court[t] is None:
                on_court[t] = np.zeros((n, len(rosters[t])), dtype=bool)
            on_court[t][needs] = chosen
            refresh_lineups[t][:] = False
        active = all_games[possessions > step]
        if not len(active):
            break
        for t in (0, 1):
            on_court_counts[t][active] += on_court[t][active].astype(np.int16)
        for offense in (0, 1):
            defense = 1 - offense
            defensive = lineup_defense(on_court[defense][active], defensive_profiles[defense])
            o, d = stats[offense], stats[defense]
            oa, da = audits[offense], audits[defense]
            oa['POSS'][active] += 1
            sh = _choose(rng, shooters[offense][active] * on_court[offense][active])
            # TO probability is shared prior with modest player/team adjustment.
            turnover_base = np.clip(np.sum(rosters[offense]['TOV'].to_numpy(float)) /
                                    max(np.sum(rosters[offense]['FGA'].to_numpy(float)) + 15, 1), .09, .17)
            turnover = rng.random(len(active)) < np.clip(turnover_base + defensive["turnover_delta"], .06, .24)
            t_games, t_sh = active[turnover], sh[turnover]
            _add(o['TOV'], t_games, t_sh)
            # A subset of turnovers are offensive fouls. They count toward
            # the offensive player's six-foul limit and affect future rotations.
            offensive_foul = rng.random(len(t_games)) < .13
            ofg, ofs = t_games[offensive_foul], t_sh[offensive_foul]
            if len(ofg):
                _add(personal_fouls[offense], ofg, ofs)
                oa['OFFENSIVE_FOULS'][ofg] += 1
                oa['PERSONAL_FOULS'][ofg] += 1
                refresh_lineups[offense][ofg] |= personal_fouls[offense][ofg, ofs] >= 3
            steals = rng.random(len(t_games)) < .57
            if steals.any():
                sg = t_games[steals]
                defender = _choose(rng, steal_w[defense][sg] * on_court[defense][sg])
                _add(d['STL'], sg, defender)
            live = active[~turnover]
            shooters_live = sh[~turnover]
            if not len(live):
                continue
            # Coupled whistle -> assigned defender PF -> bonus/free throws ->
            # possible benching -> new on-court matchup on the next possession.
            foul = rng.random(len(live)) < np.clip(
                .19 + defensive['block_delta'][np.searchsorted(active, live)] * .30,
                .14, .24)
            fg, fs = live[foul], shooters_live[foul]
            if len(fg):
                dw = (on_court[defense][fg] * foul_weights[defense][None, :] *
                      np.where(personal_fouls[defense][fg] >= 5, .05, 1.))
                df = _choose(rng, dw)
                _add(personal_fouls[defense], fg, df)
                da['PERSONAL_FOULS'][fg] += 1
                quarter_team_fouls[defense][fg] += 1
                refresh_lineups[defense][fg] |= personal_fouls[defense][fg, df] >= 3
                # Shot-foul vs non-shooting; latter only awards FTs in bonus.
                shooting = rng.random(len(fg)) < .60
                da['SHOOTING_FOULS'][fg[shooting]] += 1
                da['NONSHOOTING_FOULS'][fg[~shooting]] += 1
                bonus = (~shooting) & (quarter_team_fouls[defense][fg] >= 5)
                da['BONUS_TRIPS'][fg[bonus]] += 1
                ft_games = fg[shooting | bonus]
                ft_shooters = fs[shooting | bonus]
                if len(ft_games):
                    # 3-point shooting fouls produce three FTs; other foul trips two.
                    share = profiles[offense][0][ft_shooters, 3:].sum(axis=1)
                    is_three = (rng.random(len(ft_games)) < share * .35) & shooting[shooting | bonus]
                    attempts = (2 + is_three.astype(np.int16)).astype(np.int16)
                    made = rng.binomial(attempts, ft_prob[offense][ft_shooters]).astype(np.int16)
                    _add(o['PTS'], ft_games, ft_shooters, made)
                    _add(o['FTA'], ft_games, ft_shooters, attempts)
                    _add(o['FTM'], ft_games, ft_shooters, made)
                    oa['FTA'][ft_games] += attempts
                    oa['FTM'][ft_games] += made
            # A non-shooting defensive foul outside the bonus does NOT end an
            # NBA possession: the offense inbounds and continues. Previously
            # every defensive foul terminated the possession, suppressing
            # field-goal attempts and team scoring by several points/game.
            # Shooting fouls and bonus trips still terminate this possession.
            keep_playing = ~foul
            if len(fg):
                continue_after_inbound = (~shooting) & (~bonus)
                oa['NONSHOOTING_CONTINUATIONS'][fg[continue_after_inbound]] += 1
                keep_playing[foul] = continue_after_inbound
            live, shooter = live[keep_playing], shooters_live[keep_playing]
            if not len(live):
                continue
            shares, make_probs = profiles[offense]
            # One initial shot, then at most one second-chance shot.
            for attempt in range(2):
                if not len(live):
                    break
                active_idx = np.searchsorted(active, live)
                adjusted_shares = zone_shares_after_defense(shares[shooter], defensive["rim_deterrence"][active_idx])
                # Late-game shot selection: trailing teams seek more threes,
                # leaders favor higher-percentage twos. Only affects the final
                # ~two minutes, not the player's entire game shot profile.
                late = step >= (possessions[live] - np.maximum(3, np.rint(possessions[live] * 2 / 48).astype(int)))
                margin = (stats[offense]['PTS'].sum(axis=1)[live] -
                          stats[defense]['PTS'].sum(axis=1)[live])
                trailing = late & (margin <= -3) & (margin >= -14)
                leading = late & (margin >= 3) & (margin <= 14)
                delta = np.where(trailing, .12, np.where(leading, -.045, 0.))
                old_three = adjusted_shares[:, 3:].sum(axis=1)
                new_three = np.clip(old_three + delta, .05, .90)
                adjusted_shares[:, :3] *= ((1. - new_three) / np.maximum(1. - old_three, 1e-8))[:, None]
                adjusted_shares[:, 3:] *= (new_three / np.maximum(old_three, 1e-8))[:, None]
                zone = (rng.random(len(live))[:, None] >
                        np.cumsum(adjusted_shares, axis=1)).sum(axis=1)
                zone = np.minimum(zone, 4)
                shot_prob = np.clip(make_probs[shooter, zone] +
                                    defensive['contest_delta'][active_idx], .10, .90)
                # A plausible on-court defender is sampled; it is NOT a real
                # player-tracking matchup or measured shot-contest assignment.
                if getattr(cfg, 'defender_profiles', None) is not None:
                    shot_prob = np.clip(shot_prob + lineup_contest_modifier(
                        defender_modifiers[defense], on_court[defense][live], zone, defensive_profiles[defense]), .10, .90)
                made = rng.random(len(live)) < shot_prob
                oa['FGA'][live] += 1
                _add(o['FGA'], live, shooter)
                from_three = zone >= 3
                _add(o['3PA'], live[from_three], shooter[from_three])
                mg, ms, mz = live[made], shooter[made], zone[made]
                if len(mg):
                    _add(o['PTS'], mg, ms, np.where(mz >= 3, 3, 2).astype(np.int16))
                    oa['FGM'][mg] += 1
                    _add(o['FGM'], mg, ms)
                    threes = mz >= 3
                    _add(o['3PM'], mg[threes], ms[threes])
                    # Assisted makes: scorer excluded from passing candidate pool.
                    # Catch-and-shoot threes are more often assisted than
                    # self-created twos. Still an aggregate proxy, not passes.
                    assisted = rng.random(len(mg)) < np.where(mz >= 3, .78, .53)
                    ag, ashooter = mg[assisted], ms[assisted]
                    if len(ag):
                        w = assist_w[offense][ag].copy() * on_court[offense][ag]
                        w[np.arange(len(ag)), ashooter] = 0
                        passer = _choose(rng, w)
                        _add(o['AST'], ag, passer)
                missed = ~made
                missg, misss = live[missed], shooter[missed]
                if not len(missg):
                    break
                miss_idx = np.searchsorted(active, missg)
                blocked = rng.random(len(missg)) < np.clip(
                    .085 + defensive["block_delta"][miss_idx], .03, .18)
                if blocked.any():
                    bg = missg[blocked]
                    blocker = _choose(rng, block_w[defense][bg] * on_court[defense][bg])
                    _add(d['BLK'], bg, blocker)
                orb = (rng.random(len(missg)) < np.clip(
                    .255 + defensive["oreb_delta"][miss_idx], .16, .35)) & (attempt == 0)
                og, dg = missg[orb], missg[~orb]
                if len(og):
                    rebounder = _choose(rng, reb_w[offense][og] * on_court[offense][og])
                    _add(o['REB'], og, rebounder)
                    _add(o['OREB'], og, rebounder)
                    oa['OREB'][og] += 1
                if len(dg):
                    rebounder = _choose(rng, reb_w[defense][dg] * on_court[defense][dg])
                    _add(d['REB'], dg, rebounder)
                    _add(d['DREB'], dg, rebounder)
                    da['DREB'][dg] += 1
                live = og
                if len(live):
                    shooter = _choose(rng, shooters[offense][live] * on_court[offense][live])
    if return_quarters:
        capture_quarter(all_games, 3)
        for t in (0, 1):
            for stat, values in quarter_cumulative[t]['players'].items():
                values[1:] -= values[:-1].copy()
                if stat == 'MIN':
                    values *= (48. / np.maximum(possessions, 1))[None, :, None]
            for stat, values in quarter_cumulative[t]['events'].items():
                values[1:] -= values[:-1].copy()
    for t in (0, 1):
        # Counts are measured from the *actual simulated units*, not copied from
        # the pregame target. Every regulation game sums to exactly 240 minutes.
        stats[t]['MIN'] = (on_court_counts[t].astype(np.float32) *
                           (48. / np.maximum(possessions, 1))[:, None])
        stats[t]['PF'] = personal_fouls[t].astype(np.int16)
        audits[t]['FOUL_OUTS'][:] = (personal_fouls[t] >= 6).sum(axis=1)
    return (stats, audits, quarter_cumulative) if return_quarters else (stats, audits)
