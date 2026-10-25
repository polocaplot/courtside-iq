import numpy as np
from model import _integer_percentiles

def test_fast_percentiles_match_numpy():
    rng = np.random.default_rng(42)
    for n in (1, 2, 3, 11, 100, 1001):
        a = rng.poisson(8, size=n).astype(np.int16)
        assert np.allclose(_integer_percentiles(a), np.quantile(a, [.1, .9]))
