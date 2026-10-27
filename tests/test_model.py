import numpy as np
import pandas as pd
from mlb_xwrc.model import expected_history, fit_expected, choose_features, performance_gap
from mlb_xwrc.data import FEATURES


def test_feature_allowlist_and_gap_sign():
    frame = pd.DataFrame({x:[.1]*50 for x in FEATURES}).assign(wrc_plus=120, OPS=1.1)
    selected = choose_features(frame)
    assert 'wrc_plus' not in selected and 'OPS' not in selected
    assert performance_gap(np.array([120,80]), np.array([100,100])).tolist() == [20,-20]


def test_same_season_model_never_sees_holdout_labels(panel):
    before = expected_history(panel, 2016, 2025, 100)[0]
    changed = panel.copy()
    changed.loc[changed.season >= 2024,'wrc_plus'] += 1000
    after = expected_history(changed, 2016, 2025, 100)[0]
    assert np.allclose(before.loc[before.season==2024,'xwrc'], after.loc[after.season==2024,'xwrc'])
    assert (before.train_end < before.season).all()


def test_feature_selection_uses_training_data_only(panel):
    train = panel[panel.season < 2025].copy()
    train['exit_velocity'] = np.nan
    model, features = fit_expected(train, 100)
    assert 'exit_velocity' not in features
