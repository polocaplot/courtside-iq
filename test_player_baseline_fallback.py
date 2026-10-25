import numpy as np
from shot_intelligence import player_zone_model

def test_missing_charts_preserve_player_three_point_tendencies():
    shooter = player_zone_model('Shooter', 'HOU', None, player_baseline={'FGA': 14, 'FG3A': 8, 'FG3M': 3.2, 'GP': 30})
    interior = player_zone_model('Interior', 'HOU', None, player_baseline={'FGA': 14, 'FG3A': .5, 'FG3M': .1, 'GP': 30})
    assert shooter['shares'][3:].sum() > .5
    assert interior['shares'][3:].sum() < .1
    assert shooter['source'].startswith('player box-score')
    assert not np.allclose(shooter['shares'], interior['shares'])
