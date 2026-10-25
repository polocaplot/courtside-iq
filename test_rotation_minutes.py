import numpy as np
import pandas as pd
from model import Config, _minutes


def test_regular_rotation_does_not_dilute_starters_across_thirteen_players():
    # Atlanta-like 13 player pool: 357 projected baseline minutes cannot all
    # occur in a single regulation game. The highest-usage players retain
    # a realistic share, rather than each getting 240/357 of their baseline.
    baseline = [33.2, 32., 32., 29.4, 27.9, 25.9, 23.2, 21., 20., 18.4, 17.1, 15.9, 15.4]
    roster = pd.DataFrame({'PLAYER_NAME': [f'Player {i}' for i in range(13)], 'MIN': baseline})
    minutes, active = _minutes(np.random.default_rng(91), 3000, roster,
                               Config(minutes_sd=0), np.zeros(3000))
    assert np.allclose(minutes.sum(axis=1), 240., atol=.01)
    assert 30. < minutes[:, 0].mean() < 36.
    assert 8.8 < (minutes > 0).sum(axis=1).mean() < 10.2
    assert np.all(minutes >= 0)


def test_short_rotation_respects_injuries_and_caps():
    roster = pd.DataFrame({'PLAYER_NAME': [f'Player {i}' for i in range(13)],
                           'MIN': [36.,35.,34.,33.,30.,27.,25.,22.,18.,15.,12.,10.,8.]})
    cfg = Config(injury_overrides={'Player 1':'Out'}, minutes_overrides={'Player 0':28.})
    minutes, active = _minutes(np.random.default_rng(7), 250, roster, cfg, np.zeros(250))
    assert np.allclose(minutes.sum(axis=1), 240., atol=.01)
    assert np.all(minutes[:,1] == 0)
    assert np.all(minutes[:,0] <= 28.0001)


def test_starter_minutes_resist_oversubscribed_roster():
    """A 310-minute pool must not shrink a 35-minute starter to ~27."""
    baseline = [35., 34., 33., 31., 30., 27., 25., 23., 21., 19., 17., 15., 13.]
    roster = pd.DataFrame({'PLAYER_NAME': [f'P{i}' for i in range(len(baseline))],
                           'MIN': baseline})
    minutes, _ = _minutes(np.random.default_rng(14), 5000, roster,
                          Config(minutes_sd=0), np.zeros(5000))
    assert np.allclose(minutes.sum(axis=1), 240., atol=.001)
    assert minutes[:, 0].mean() > 32.
    assert minutes[:, 1].mean() > 31.
    assert minutes[:, 2].mean() > 30.
    assert np.all(minutes <= 48.0001)


def test_minute_allocation_respects_hard_caps_and_out_status():
    baseline = [38., 36., 34., 32., 30., 28., 25., 22., 18., 16., 12.]
    roster = pd.DataFrame({'PLAYER_NAME': [f'P{i}' for i in range(len(baseline))],
                           'MIN': baseline})
    cfg = Config(minutes_sd=1.5, minutes_overrides={'P0': 25., 'P1': 30.},
                 injury_overrides={'P2': 'Out'})
    minutes, _ = _minutes(np.random.default_rng(11), 1000, roster,
                          cfg, np.zeros(1000))
    assert np.allclose(minutes.sum(axis=1), 240., atol=.001)
    assert (minutes[:, 0] <= 25.0001).all()
    assert (minutes[:, 1] <= 30.0001).all()
    assert (minutes[:, 2] == 0).all()
