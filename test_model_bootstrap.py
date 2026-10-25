from pathlib import Path
import pandas as pd
import pytest
from model_bootstrap import bootstrap_model


def test_bootstrap_uses_real_logs_and_writes_manifest(tmp_path):
    called = {}
    def fetch(seasons, cache_dir):
        called['seasons'] = seasons
        called['cache'] = cache_dir
        return pd.DataFrame({'GAME_ID':['001'], 'PLAYER_ID':[42]})
    def train(logs, folder):
        assert len(logs) == 1
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'manifest.json').write_text('{}')
        return {'all_games': 1}
    result = bootstrap_model(tmp_path, ('2024-25',), fetch=fetch, train=train)
    assert result['all_games'] == 1
    assert called['seasons'] == ('2024-25',)
    assert called['cache'] == tmp_path / 'game_logs'


def test_bootstrap_never_trains_on_missing_logs(tmp_path):
    def fetch(*args):
        return pd.DataFrame()
    def train(*args, **kwargs):
        raise AssertionError('must not train')
    with pytest.raises(ValueError, match='No historical'):
        bootstrap_model(tmp_path, fetch=fetch, train=train)


def test_bootstrap_does_not_claim_success_without_manifest(tmp_path):
    with pytest.raises(RuntimeError, match='manifest'):
        bootstrap_model(tmp_path, fetch=lambda *args: pd.DataFrame({'x':[1]}),
                        train=lambda *args, **kwargs: {'all_games': 1})


def test_duplicate_seasons_rejected(tmp_path):
    with pytest.raises(ValueError, match='unique'):
        bootstrap_model(tmp_path, ('2025-26', '2025-26'))
