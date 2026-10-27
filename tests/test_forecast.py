import numpy as np
import pandas as pd
from mlb_xwrc.forecast import consecutive_pairs, fit_forecast, backtest_forecasts, prediction_radius
from mlb_xwrc.model import expected_history


def test_pairs_require_adjacent_years_and_do_not_match_names():
    frame = pd.DataFrame({'player_id':[1,1,2], 'season':[2020,2022,2021],
                          'age':[25,27,25], 'pa':[500]*3, 'wrc_plus':[100]*3,
                          'xwrc':[100]*3, 'history_wrc':[100]*3})
    assert consecutive_pairs(frame, frame, 100).empty


def test_forecast_training_stops_before_target_year(panel):
    oof = expected_history(panel,2016,2025,100)[0]
    pairs = consecutive_pairs(oof.assign(history_wrc=oof.wrc_plus), panel, 100)
    first, curve = fit_forecast(pairs[pairs.target_season < 2025])
    assert first['train_target_end'] == 2024
    assert (curve.pair_count >= 0).all()
    folds, _ = backtest_forecasts(pairs, [2023,2024,2025])
    assert (folds.train_target_end < folds.target_season).all()


def test_interval_uses_only_earlier_errors():
    errors = pd.DataFrame({'target_season':[2023,2024,2025],
                           'actual':[100,100,9999], 'forecast':[90,80,0]})
    assert prediction_radius(errors[errors.target_season<2025], .9) <= 20

def test_history_never_uses_future_seasons():
    from mlb_xwrc.forecast import add_history
    current = pd.DataFrame({'player_id':[1],'season':[2024]})
    history = pd.DataFrame({'player_id':[1,1,1,1], 'season':[2022,2023,2024,2025],
                            'pa':[500]*4, 'wrc_plus':[80,100,120,10000]})
    got = add_history(current,history)
    expected = (80*.25+100*.5+120)/1.75
    assert np.isclose(got.iloc[0].history_wrc,expected)
