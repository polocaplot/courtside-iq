import numpy as np
import pandas as pd
from data_sources import reconcile_shooting_lines
from model import _minute_prior
from diagnostic_bundle import _effective_minutes


def test_reconcile_impossible_shooting_after_historical_blend():
    p = pd.DataFrame([dict(FGA=6.18, FGM=0, FG3A=4.33, FG3M=1.62,
                           FTA=1.35, FTM=1.08, FG_PCT=0)])
    out = reconcile_shooting_lines(p)
    assert out.FGM.iloc[0] >= out.FG3M.iloc[0]
    assert out.FGM.iloc[0] <= out.FGA.iloc[0]
    assert out.SHOOTING_INPUT_RECONCILED.iloc[0]
    assert abs(out.FG_PCT.iloc[0] - out.FGM.iloc[0] / out.FGA.iloc[0]) < 1e-10


def test_reconcile_preserves_valid_lines():
    p = pd.DataFrame([dict(FGA=10., FGM=5., FG3A=4., FG3M=2., FTA=3., FTM=2.)])
    out = reconcile_shooting_lines(p)
    assert not out.SHOOTING_INPUT_RECONCILED.iloc[0]
    assert out.FGM.iloc[0] == 5.


def test_minute_audit_matches_simulator_prior():
    row = dict(MIN=29.614, MIN_L5=34.5, MIN_L10=33., GAMES_PLAYED=2)
    assert np.isclose(_effective_minutes(row), round(_minute_prior(pd.DataFrame([row]))[0], 3))
