COURTSIDE IQ — Precision Guardrails Update (2026-10-24)

Implemented:
- Optional observed recent-minute priors (MIN_L5, MIN_L10, GAMES_PLAYED), shrunk toward season average for small samples.
- Optional confirmed starters via Config.confirmed_starters = {'MIA': ['name', ... 5 total]}; does not invent confirmed lineups.
- Explicit input warnings for missing recent minutes, unconfirmed starters, oversized rotation minutes, and invalid core inputs.
- Unresolved >5 point possession scoring calibration gaps flagged in simulation result as precision_warnings.
- All prior defense and video features preserved.

Not yet implemented or validated:
- Automated reliable starting-lineup feed and injury verification.
- Historically trained and activated ML model for this user's dataset.
- Full possession model recalibration and backtest, player usage/rebounding/foul overhaul.
- Tracking-quality defender assignments and full-game video learning.

The extra recent-minute and starter inputs only apply when passed in by an upstream data provider. Missing columns are not fabricated. Run a fresh diagnostic and inspect precision_warnings.
