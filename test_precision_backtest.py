import pandas as pd
import pytest
from precision_backtest import rolling_audit


def fake_logs():
    rows = []
    for i in range(36):
        rows.append(dict(GAME_ID=f'{i:010d}', GAME_DATE=(pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                         PLAYER_ID='123', PLAYER_NAME='Test Player', TEAM_ABBREVIATION='MIA', MATCHUP='MIA vs. CHA',
                         MIN=20+i%5, PTS=10+i%3, REB=5, AST=4, STL=1, BLK=1, TOV=2, FG3M=1))
    return pd.DataFrame(rows)


def test_backtest_uses_only_prior_games():
    original = fake_logs()
    a = rolling_audit(original)
    altered = original.copy()
    altered.loc[altered.index[-1], 'PTS'] = 60
    b = rolling_audit(altered)
    assert a['metrics']['PTS']['mae'] != b['metrics']['PTS']['mae']
    assert a['holdout_start'] == b['holdout_start']
    assert a['metrics']['MIN']['n'] > 0


def test_requires_real_history():
    with pytest.raises(ValueError):
        rolling_audit(fake_logs().head(3))
