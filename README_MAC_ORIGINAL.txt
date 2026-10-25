## Coupled game-state upgrade (October 2026)

In **Detailed possessions** mode, foul calls are assigned to on-court defenders;
individual personal fouls and approximate quarterly team-bonus counts persist
through the game. Shooting fouls, non-shooting fouls, offensive fouls, 2/3-shot
free-throw trips, and bonus trips alter scoring and foul counts. Foul trouble
changes lineup selection; lineup selection changes opponent defensive event
rates and shooting probabilities; actual on-court possession counts generate
projected minutes. Fourth-quarter blowouts affect lineup selection as well.

The model is **approximate**, not optical tracking or official NBA play-by-play.
It does not reconstruct exact quarter clock, timeout, substitutions, matchup
assignments, or all edge-case NBA foul rules. Its defensive ratings are
conservative box-score-derived proxies unless measured defender profiles are
supplied. Fouls and substitution priors require historical walk-forward
backtesting before claiming predictive accuracy.

# COURTSIDE IQ — Possession Research Upgrade (October 2026)

**This build adds an optional event-level simulation engine and a richer projected box score.**

## New in this version

- **Possession-event mode:** The model iterates through approximate possessions for both teams, assigning ball-handler/shooter, turnover/steal, shooting foul/free throws, five-zone shot attempt, make/miss, block, assist, offensive rebound/one second-chance attempt, or defensive rebound. Every game uses the same pace environment for both teams.
- **Shot context:** Player shot-zone probabilities use uploaded or fetched real shot charts when available, empirical-Bayes smoothing, and optional imported opponent zone defense. Otherwise, league priors are clearly labeled. This is NOT actual defender tracking.
- **Pregame matchup calibration:** A separate 1,200-game pilot adjusts zone make probabilities (capped at 10 percentage points) toward pregame team scoring/defense baselines. This is a **heuristic**, not a trained xFG model. It is not fitted to game outcomes.
- **Player FT% estimates:** When NBA.com provides FTM and FTA, player free-throw make probabilities are smoothed toward a 77% prior. Missing FTM uses that prior.
- **Expanded box score:** FGA, FGM, 3PA, FTA, FTM, offensive rebounds and defensive rebounds are now shown directly below the game simulation when possession mode is selected, in addition to MIN, PTS, REB, AST, 3PM, STL, BLK, TOV and fantasy.
- **Possession event audit:** Team averages for possessions, field goals, free throws, offensive and defensive rebounds, and pilot calibration appear below the projected boxes.
- **Existing tools preserved:** Fantasy Lab, Box Score, Player Lab, Shot Lab, Prop Lab, Scenario Lab, live roster/injury polling while the app is open, optional historical ML training, market comparison, accuracy tracking and customizable themes.

### Use it

1. Unzip this download to a **new folder** (do not mix it with the old version). Double-click `START_COURTSIDE_IQ.command`.
2. Select an upcoming NBA game in the left sidebar. In **Shot-level intelligence**, enable **Possession-event simulator (experimental)**. Optionally use **Shot Lab** to fetch real player shot charts first.
3. In **Command Center**, run 150,000 simulations. Scroll below the win probabilities to see the projected fantasy leaderboard, **both full projected box scores**, and the event audit.
4. To compare the original statistical engine, turn the possession-event toggle **off** and rerun the same matchup and seed. The two engines have different assumptions; more detail does not prove more accuracy.

### What is still not implemented

- Exact five-man on-court lineup substitutions, defensive assignments, individual player movement, screens, pass trajectories, shot-clock-dependent decisions, intentional fouls, accurate personal foul accounting, or full overtime possession sequences.
- Official NBA injury report ingestion: the current automatic status feed is unofficial ESPN and can be late or incomplete. Always confirm late scratches.
- Automatically fetched opponent-zone defense data, defender-distance tracking, trained possession-transition probabilities, or a validated 2026–27 possession-level backtest.
- Continuous background updates after you close the app, guaranteed current sportsbook data without an API key, or any demonstrated Vegas-level accuracy.

The model intentionally uses simplified event probabilities and minute-weighted player selection; it **cannot** know everything happening in an actual NBA game. Possession-mode overtime is simplified, and the extra overtime points may not reconcile with overtime shot attempt counts. Some event priors and calibration settings are not statistically optimized. Use projections as research estimates, not certainty.

## Original documentation

# COURTSIDE IQ PRO — NBA / Fantasy Intelligence

A customizable **local Mac dashboard** that combines real NBA matchup data, injury-aware player box-score projections, 150,000 seeded simulations, fantasy rankings, historical machine-learning training, model accuracy evaluation, and optional sportsbook comparisons.

This repository contains source code, **not pretrained weights and not proprietary professional projections**. The model must be trained on real historical NBA data after installation. The application labels unverified feeds and offers offline synthetic demo mode.

## Mac setup — no coding required

1. Download `courtside_iq_pro_mac.zip` and unzip it in Downloads.
2. Open its **`courtside_iq_pro`** folder and double-click **`START_COURTSIDE_IQ.command`**.
3. If macOS blocks the file: right-click → **Open**. If necessary, open Terminal, type `bash ` (including a trailing space), drag the `.command` file into the Terminal window, and press Return.
4. First start installs required Python packages privately under `.venv`; **Python 3.10+** and internet access are needed. Install Python from <https://www.python.org/downloads/macos/> if missing.
5. Dashboard opens in Safari/your browser at **http://localhost:8501**. **Leave its Terminal process running** while using it.

Alternative Terminal commands (after unzipping):

```bash
cd ~/Downloads/courtside_iq_pro
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you already installed a previous COURTSIDE IQ version, use **this new unzipped folder** so the launcher installs the new ML dependencies.

## First-run workflow

1. In the sidebar, use **Data source: Live NBA data** for real, non-demo games. Select a scheduled upcoming matchup.
2. Select **ML Studio** to download and train on actual regular-season player logs (defaults: 2023–24 through 2025–26). NBA.com may refuse automated requests. Alternatively upload a real combined NBA `LeagueGameLog` CSV. This training can take time.
3. Inspect the holdout MAE table. Stat models are activated only if their held-out performance beats a prior-games rolling baseline by at least 1%; otherwise the rolling baseline is retained. No model is declared "accurate" by default.
4. In **Command Center**, simulate 150,000 game outcomes and browse **Fantasy Lab** and **Box Score**. The fantasy scoring preset is PTS + 2REB + 2AST + 3STL + 3BLK − TOV. Adjust weights any time.
5. Enable **Live Sync** in the sidebar to poll injuries, team rosters, playing-time baselines, game statuses, and automatically invalidate/rerun selected-game pregame forecasts. Optional full-slate monitoring and rankings are available.
6. Use **Accuracy Lab** to save real pregame predictions and check them against completed games. **Market Lab** can fetch live player-prop prices with your own *The Odds API* key (availability and plan permitting), compare simulation probabilities, and display raw/no-vig implied probabilities.
7. In **Appearance**, choose dark/light themes, accent colors, card shapes, typography size, chart type, density and dashboard content preferences.

## What's technically implemented

| Area | Implementation |
|---|---|
| Live games | ESPN's **unofficial** scoreboard feed; scheduled games and final results |
| Player and team baselines | NBA.com via community `nba_api`; current/previous season + last-ten blend |
| Rotation/injury assumptions | ESPN unofficial roster/injury reports; conservative feed state change detector, scenario status and minute caps |
| 150,000 Monte Carlo | Seeded game pace/score/rotations with shared team environment; 240 regulation player-minutes, player points sum to each simulated team score |
| Player boxes | Minutes, points, rebounds, assists, three-pointers, steals, blocks, turnovers, plus 10th/90th percentiles |
| Fantasy | Your exact custom scoring, mean/median/ceiling/floor, simulated threshold probabilities, boom/bust rates, full slate |
| Historical ML | Separate gradient-boosting models for MIN/PTS/REB/AST/STL/BLK/TOV/3PM; rolling pregame histories, rest/back-to-back, opponent lagged defense, game context |
| Validation | Most recent 20% of historical dates held out, no same-game feature leakage, rolling-average comparator; per-stat guardrails, fantasy-score MAE |
| Automatic retraining | Once-daily current-season game-log check **while dashboard is open**, retrain only if ≥50 additional player-game rows are found; otherwise preserve existing weights |
| Market research | Optional real sportsbook event/player-prop quotes from *The Odds API v4* using a user-supplied key; basic no-vig normalization where both sides are quoted |
| Persisted settings | Local `~/.courtsideiq/` JSON and trained models; no sportsbook authentication or logins |

## What it does **not** yet do (important)

- **Professional predictive performance is not demonstrated**. Professional modeling requires sustained walk-forward backtesting on actual forecasts, high-quality lineups/rotations, latency-tested injury data and reliable historical betting markets. Historic MAE alone does not prove probability calibration or gambling profitability.
- The 150,000 draws are **statistical game simulations**, not literal NBA possession-by-possession recreations. Individual scoring is allocated to coherent team totals; assist/rebound distributions use statistical approximations. No exact five-player lineup/defensive matchup on-off engine is present.
- **Neither NBA.com nor ESPN offers these interfaces as a fully guaranteed supported feed.** NBA.com can block data downloads or change its schema, and ESPN's unofficial injury list can lag the NBA's official report. "Not listed" does not mean healthy; roster deltas are not confirmed transactions.
- No automatic sportsbook login, wagering, buy/sell transaction, guaranteed official starting lineups, realtime possession-level in-game predictor, mobile push notification, or monitoring **while the Mac app is closed**.
- Automatic ML updates only run after an initial successful training. Daily checks while open may fail from API throttling; if they fail, old weights are retained. The training file is sourced from actual NBA game logs **only if fetch or CSV import succeeds**. **Synthetic demo is never used to train or to make "live" claims.**
- Odds comparisons are indicative: model "edge" and hypothetical expected profit are not validated; verify injury and line timestamps. Odds API plans may require payment for player props.

### Data references

- `nba_api` open-source library and `LeagueGameLog` docs: <https://github.com/swar/nba_api>
- ESPN unofficial basketball endpoints, subject to changes: <https://hoopr.sportsdataverse.org/articles/espn-endpoints.html>
- Official reports to cross-check status: <https://official.nba.com/>
- The Odds API player props reference: <https://the-odds-api.com/sports/nba-odds.html>

## Tests / exports

```bash
python3 -m pytest -q
```

Automated tests include score reconciliation, 240-minute regulation constraint, fantasy formula, model seed reproducibility, historical feature look-ahead isolation, chronological training, a short synthetic **test-fixture-only** training run, and optional odds calculation. The tests **do not verify network access, Mac Safari rendering, or predictive performance on actual 2026–27 games**.

User preferences and saved model files go in `~/.courtsideiq/`; these are private local files. Team/player CSV templates are provided for explicitly imported-data mode.

## Shot Intelligence Lab — research upgrade (October 2026)

The **Shot Lab** adds a separate, opt-in shot-zone scoring engine to the original 150,000-draw simulation. Open **Shot Lab** after selecting a real game; click **Fetch real shot locations** to request NBA.com `ShotChartDetail` for both teams. Data is cached locally by season. Alternatively import aggregated player-zone CSV and an opponent-zone defense CSV. Turn on **Simulate shots by court zone** in the sidebar, then rerun Command Center. Full projected box scores are now also displayed **directly under the simulation results**.

When shot mode is enabled, the engine samples per-player field-goal attempts in five zones, binomial shot makes from shrinkage-adjusted zone FG%, and approximate free throws. Simulated player scoring determines team scores (rather than the previous score-first allocation). Made threes are derived from actual simulated three-point makes. Injuries, minutes, usage and pace remain shared inputs. This is a *zone-based shot attempt model*, not a chronological possession-by-possession NBA replay. The other box-score stats remain statistical approximations. No exact defenders, switches, fouls, passes, shot-clock state, ten-player coordinates or rebound positions are modeled.

Shot-zone make probabilities are smoothed toward league-prior values using 35 pseudo-attempts per zone; zone shares use 35 pseudo-attempts. Imported opponent-zone allowed FG% is shrunk with 150 pseudo-attempts, and adjustments are capped at ±6 percentage points. These hyperparameters are **heuristics**, not trained or validated optimal weights. Free-throw shooting uses a 77% prior, not player-specific FT%, in this experimental mode. Without enough actual chart observations, player zones use priors; the app explicitly labels this. Do not interpret a model edge as proof of profitable bets.

### Shot model sources and future research priorities

1. NBA.com `ShotChartDetail` provides shot location, distance, zone, player, date and made/missed result: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/shotchartdetail.md
2. NBA.com `LeagueDashTeamShotLocations` can provide team and **opponent** FG% by zone (a promising future automatic defensive input): https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashteamshotlocations.md
3. NBA.com `PlayerDashPtShots` provides closest-defender, shot-clock, touch-time and dribble bins; these are **aggregate bins**, not exact defender locations per shot: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playerdashptshots.md
4. PBP Stats offers possessions, lineup-on-floor, shot-zone assists/rebounds, on/off, and playing-time distribution: https://api.pbpstats.com/docs and https://pbpstats.readthedocs.io/en/latest/
5. The official NBA injury report updates throughout the day; the current app's injury automation still relies on an unofficial ESPN feed: https://official.nba.com/nba-injury-report-2026-27-season/
6. Historical market odds can be purchased from providers such as The Odds API; plans and historical coverage vary: https://the-odds-api.com/historical-odds-data/

**Not yet implemented**: automated official injury-PDF parsing, lineup-aware possession transitions, defender-tracking xFG, automatic opponent-zone download, player-specific FT model, full shot-mode walk-forward validation, and trained possession-transition probabilities. These require reliable source access, calibration and historical testing. A sophisticated-looking model without validated forecasts is not necessarily more accurate.

### Shot-zone CSV format

Player shots (`TEAM_ABBREVIATION,PLAYER_NAME,ZONE,FGA,FGM`):

```csv
TEAM_ABBREVIATION,PLAYER_NAME,ZONE,FGA,FGM
BOS,Example Player,Rim,120,83
BOS,Example Player,Above-break 3,150,57
```

Opponent defense (`TEAM_ABBREVIATION,ZONE,FGA_ALLOWED,FGM_ALLOWED`):

```csv
TEAM_ABBREVIATION,ZONE,FGA_ALLOWED,FGM_ALLOWED
NYK,Rim,500,305
```

Zone names: `Rim`, `Paint`, `Midrange`, `Corner 3`, `Above-break 3`. CSV examples are **illustrative**, not actual NBA player data.

## Lineup, defender, and historical calibration extension

- **Possession mode now samples exactly five eligible players per side** for each
  rotation segment (default eight possessions), restricting shooter, passer,
  rebounder, blocker, and steal candidates to the sampled unit. This is an
  approximation: sampled minutes are *not* a reconstruction of actual game
  substitutions, nor guaranteed to exactly match each player's projected minute
  fraction across every simulated game. Team and player scoring still reconcile.
- **Observed lineup priors**: In Shot Lab, optionally import a CSV with columns
  `TEAM_ABBREVIATION,PLAYER_1,PLAYER_2,PLAYER_3,PLAYER_4,PLAYER_5,POSSESSIONS`.
  Only five-player units whose players have positive projected minutes are used.
  Missing units fall back to minute-weighted lineup sampling. Use actual lineup
  counts from a licensed/reliable source, not the demonstration template.
- **Defender-by-zone observations**: Optionally import
  `TEAM_ABBREVIATION,DEFENDER_NAME,ZONE,FGA_CONTESTED,FGM_ALLOWED`.
  Observed rates are shrunk toward league-zone priors with 200 pseudo-attempts,
  capped at +/-4 percentage points. A *plausible* on-court defender is selected
  for each shot; it is NOT a real defensive assignment or optical tracking.
- **Accuracy Lab / historical probability calibration**: Import genuine completed
  pregame forecasts with `GAME_DATE,HOME_WIN_PROB,HOME_WON` (optionally `GAME_ID`
  and `FORECAST_SAVED_AT`). Minimum 80 completed games on 15 dates; 80/20
  chronological holdout by date, at least 50 train and 15 holdout games.
  Isotonic calibration is **research-only** and is not applied automatically.
  Both Brier score and log loss must improve on unseen games before a candidate
  is flagged. A holdout improvement alone is not evidence of profitable bets.
- **No actual optical tracking or proprietary sportsbook feed is bundled.**
  The code cannot infer real player coordinates, defender distance, screens,
  matchups or defensive schemes from ordinary NBA box scores. Shot and defender
  inputs require data provenance checks, coverage checks and legal access.
- **No pre-trained win-probability calibrator is shipped**, because historical
  pregame forecasts and outcomes must be collected without future leakage.
  A single template row is illustrative and insufficient to train a calibrator.

The included `lineup_frequencies_template.csv`, `defender_zone_template.csv`,
and `calibration_template.csv` are SCHEMA EXAMPLES ONLY; do not interpret their
values as actual NBA observations. For reliable projections, import real data.

### Verification performed for this release

- 31 automated tests passed (including injury eligibility, five-player lineup
  sampling, observed-unit fallback, defender shrinkage, chronology and leakage
  checks, and simulation reconciliation).
- A local **150,000-draw possession-mode smoke test** completed on synthetic
  demonstration rosters in approximately 23 seconds in the test environment.
  This measures execution, not real-game accuracy. The Mac UI and live NBA.com
  feeds were not verified on a physical Mac.

## Automatic feed + accuracy upgrade (October 2026)

### New automated public-data inputs

- **NBA.com five-player lineup aggregates:** attempts `LeagueDashLineups` for the chosen season and resolves lineup member IDs using the actual player-stat table. When the endpoint reports possessions, those are used as weights; otherwise the observed lineup **minutes** are used as relative weights (not mislabeled as observed possessions). Cached under `~/.courtsideiq/advanced_feeds/`.
- **NBA.com defended-shot aggregates:** attempts `LeagueDashPtDefend` for shots **under six feet** and **three-pointers**. The under-six-foot bucket is a rim proxy; all defended three-pointers are used as a broad prior for both corner and above-break threes. These are **not** true zone-by-zone tracking data and do **not** identify which defender was guarding a particular shooter. Paint non-rim and midrange zones are not populated from these feeds.
- **Automatic refresh:** the open app attempts a daily refresh and retains the last successful cached data if an endpoint fails. The **Shot Lab** has an explicit **Fetch NBA.com lineup and defender aggregates** button and reports per-source errors. The sidebar allows disabling daily fetches. NBA.com may throttle or change its schemas.
- **User CSVs remain supported** for more precise licensed data, if you have it.

### New automatic forecast auditing

- When **Automatically save genuine pregame forecasts** is enabled, a real scheduled-game simulation is locally timestamped before tipoff. Saves are rate-limited to once per matchup every four hours; synthetic/demo and post-tipoff results are excluded.
- When **Automatically grade completed games** is enabled, the dashboard checks pending forecasts against final NBA game status and actual box scores at most hourly **while the app is open**. Finished-game evaluations are cached locally and not overwritten by later forecasts.
- **Accuracy Lab** now displays observed fantasy mean absolute error, team score errors, and an export of verified predictions for chronological win-probability calibration. A calibration model still requires sufficient actual observations and a genuine held-out test set.
- **No background daemon** is installed. Refresh and grading do not happen when the app is closed, the Mac is asleep, or the data feed is inaccessible. The new automated data is **not** proof of Vegas-level predictive accuracy.

Tests: `python -m pytest -q` (36 local offline tests at packaging time). Live NBA endpoints and Mac UI have **not** been end-to-end verified in this environment.

## 2026–27 generalization and defensive reliability upgrade

- **New season:** The game date determines the requested season. Current-season
  roster reconciliation determines team membership; prior-season player statistics
  are blended by matching `PLAYER_ID` only, with stronger shrinkage for small
  current-season samples (18-game prior). Team scoring, pace and available team
  defensive efficiency also use prior-season estimates early in the season.
  A player who changes teams is **not** put back on their old team merely because
  their historical statistics come from that team.
- **New lineups:** All possession simulations continue to sample any five
  eligible players according to current projected minutes. Observed five-player
  units are optional evidence, blended with broad sampling rather than forced.
  A concentrated one-lineup sample receives a steep discount; invalid units
  containing absent players receive zero probability. An unfamiliar combination
  can still play even when the app has extensive historical lineup observations.
- **Defense:** Player box-score steal, block and rebound proxies now shrink
  based on *season games played × minutes per game*, rather than treating a
  single high-minute game as a reliable defensive sample. Team defensive
  opponent-scoring baselines are also stabilized early in the season when the
  previous season's advanced statistics can be retrieved.
- **Integrity:** Model Integrity shows early-season games-played and the
  approximate share of current-season evidence. The lineup and defender feeds
  remain best-effort and their failures do not create fake tracking data.
- **Fix:** The defensive-event engine module is explicitly included in this
  distribution, preventing an import failure present in an earlier ZIP.

**Important limitations:** The model is not perfect or exhaustively calibrated.
New players without comparable historical data still rely on generic priors;
real defensive assignments, actual lineup substitutions, exact shot contests,
coaching decisions, and future trades are unknown. Simulated detail is not a
substitute for prospective, out-of-sample accuracy validation.

## Optional smart pregame inputs (non-destructive)

The **Pregame intelligence** sidebar expander introduces an opt-in layer in front
of the existing statistical and possession-event engines. The original engines,
seeded sampling, box-score exports and fantasy scoring are unchanged. By default
this feature is **OFF** to preserve established behavior.

When real previous-season NBA player statistics are available, the layer blends
per-minute rates using `current_GP / (current_GP + 16)` and minutes using
`current_GP / (current_GP + 10)`. This prevents a two-game shooting surge from
completely overriding a longer prior history. These constants are transparent
heuristics, **not historically optimized parameters**. Players without matched
prior-year stats keep their current inputs; prior-year-only roster fallbacks
remain explicitly labeled. New teammates and new lineups are still eligible.

An optional injury-aware opportunity adjustment increases active teammates'
usage modestly when a teammate is ruled Out or Doubtful; manual usage overrides
are never overwritten. This is an approximate opportunity proxy, not a claim
that the system knows exact touches or rotations. Questionable players remain
probabilistic in the underlying simulation.

After running a matchup, expand **Compare original simulator vs smart inputs**
and run a 3,000-draw comparison. It uses the **same underlying engine** on both
sides with identical seed and settings. The original 150,000-draw run is not
replaced by the comparison. These comparisons show *differences*, not accuracy:
validating which model is better requires saved, genuinely pregame forecasts
and chronological evaluation against actual games.

Live NBA data access may fail or be incomplete; the app does not silently
invent a previous-season dataset. The smart layer does not currently ingest
commercial tracking feeds, verified starting lineups, or sportsbook model weights.


## NBA Database (October 2026)
Open **NBA Database** to inspect the local SQLite archive at `~/.courtsideiq/nba_archive.sqlite3`.
The app starts a nonblocking collection process when live mode opens. It fetches all
scheduled games for each processed date, downloads full available summary/boxscore/
play-by-play data after games are final, retains compressed source JSON, computes
estimated possessions/pace and four factors, and incrementally backfills from
2023-10-01. Re-running a day does not duplicate records. The archive starts empty;
previous seasons populate gradually and depend on third-party source availability.

For independent collection when the dashboard is closed, double-click
`INSTALL_BACKGROUND_SYNC.command` **after opening the app once**. It installs a
user-level launchd job every six hours while the Mac is awake. It does not run
when the Mac is asleep; the next run catches up. The script can be uninstalled
with the printed `launchctl bootout` command.

The database does **not** claim individual defender assignments, complete
optical tracking, verified five-player stints, or exact possession boundaries.
Source-coverage flags distinguish missing information from true zero values.
NBA.com prior-season baselines remain primary; database minutes and pace are
conservatively blended only from games dated before the matchup.


## COURTSIDE VISION (experimental)
A separate local video tracking workspace has been added. See COURTSIDE_VISION_README.txt. It requires user-supplied, authorized footage and an optional YOLO/OpenCV install. It records pixel-space detections, visible tracks, and heuristic ball proximity, not full Second Spectrum tracking or reliable named NBA player metrics. Vision observations are NOT fed into game predictions until calibrated and validated.
