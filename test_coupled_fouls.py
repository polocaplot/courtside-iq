"""End-to-end invariants of the coupled minutes/defense/foul simulation."""
import numpy as np
import pandas as pd
from model import Config, run


def _inputs():
    import json
    from pathlib import Path
    path = Path(__file__).parent / 'fixtures' / 'hou_atl_pregame.json'
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return pd.DataFrame(raw['pregame_team_inputs']), pd.DataFrame(raw['pregame_player_inputs'])


def test_coupled_real_diagnostic_sample():
    inp = _inputs()
    if inp is None:
        return
    teams, players = inp
    cfg = Config(iterations=300, seed=43, possession_mode=True,
                 shot_mode=True, calibrate_possession=False)
    result = run(teams, players, 'ATL', 'HOU', cfg)
    for side in ('ATL', 'HOU'):
        arr = result['team_arrays'][side]
        audit = result['possession_audits'][side]
        assert np.all(np.isclose(arr['MIN'].sum(axis=1), 240., atol=.01) | np.isclose(arr['MIN'].sum(axis=1), 265., atol=.01))
        assert np.all(arr['PF'] >= 0)
        assert np.array_equal(arr['PF'].sum(axis=1), audit['PERSONAL_FOULS'])
        assert np.array_equal(arr['PTS'].sum(axis=1), result['home_score' if side == 'ATL' else 'away_score'])
        assert np.all(arr['3PM'] <= arr['3PA'])
        assert np.all(arr['FGM'] <= arr['FGA'])
        assert np.all(arr['FTM'] <= arr['FTA'])
        assert np.all(arr['PF'] <= 7)  # no prolonged foul-out participation
        assert audit['SHOOTING_FOULS'].sum() > 0
        assert audit['NONSHOOTING_FOULS'].sum() > 0
        assert audit['BONUS_TRIPS'].sum() > 0
