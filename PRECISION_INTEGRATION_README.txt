COURTSIDE IQ precision integration (2026-10-24)

Included in this build:
- Preserve genuine NBA last-5 and last-10 player minute rates from the live feed.
- Pass recent minute rates and current-season game counts into rotation prior calculations.
- Flag unconfirmed starters, thin early-season data, missing shot profiles and absent defender tracking.
- Add leakage-safe rolling pregame historical box-score accuracy audit: precision_backtest.py.
- Retain existing possession simulation, team defense, rebounding, fouls, usage, historical ML training, and schedule-aware training features.

IMPORTANT: This does NOT mean every precision feature is trained or validated.
Historical ML requires downloaded genuine game logs and training. Historical injury
reports and exact on-ball matchups are not available from box scores. The rolling
audit is a box-score baseline, NOT a full possession-engine backtest.

To run: cd "COURTSIDE IQ.app/Contents/Resources/source";
python precision_backtest.py your_historical_player_game_logs.csv
