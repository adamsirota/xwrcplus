"""Fixed-parameter XGBoost and chronological same-season evaluation."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from .data import CORE, FEATURES


def performance_gap(actual, expected):
    return actual - expected


def choose_features(train):
    selected = [c for c in FEATURES if c in train and train[c].notna().mean() >= .7]
    if 'bb_rate' not in selected or 'k_rate' not in selected:
        raise ValueError('Need at least 70% coverage of BB and K rates in training data.')
    return selected


def fit_expected(train, min_pa):
    train = train[train.pa >= min_pa].copy()
    if len(train) < 50:
        raise ValueError('At least 50 eligible training player-seasons required.')
    features = choose_features(train)
    model = XGBRegressor(n_estimators=180, max_depth=2, learning_rate=.04,
                         min_child_weight=10, subsample=.9, colsample_bytree=1,
                         reg_lambda=10, objective='reg:squarederror',
                         tree_method='hist', n_jobs=2, random_state=42)
    model.fit(train[features], train.wrc_plus, sample_weight=train.pa / train.pa.mean())
    return model, features


def metrics(actual, predicted):
    return {'mae':float(mean_absolute_error(actual,predicted)),
            'rmse':float(np.sqrt(mean_squared_error(actual,predicted))),
            'r2':float(r2_score(actual,predicted)) if len(actual)>1 else None,
            'n':len(actual)}


def expected_history(panel, start, end, min_pa, exclude_2020=False):
    records, scores, importance = [], [], []
    eligible = panel[(panel.season >= start) & (panel.season <= end) & (panel.pa >= min_pa)].copy()
    if exclude_2020:
        eligible = eligible[eligible.season != 2020]
    for year in range(start+3, end+1):
        train, test = eligible[eligible.season < year], eligible[eligible.season==year].copy()
        if test.empty:
            continue
        model, features = fit_expected(train, min_pa)
        test['xwrc'] = model.predict(test[features])
        test['train_end'] = int(train.season.max())
        test['missing_feature_fraction'] = test[features].isna().mean(axis=1)
        test['features_used'] = ','.join(features)
        records.append(test)
        baseline = np.repeat(np.average(train.wrc_plus, weights=train.pa), len(test))
        for label, predictions in [('xgboost',test.xwrc), ('training_mean',baseline)]:
            scores.append({'season':year,'model':label, **metrics(test.wrc_plus,predictions)})
        if year == end:
            values = permutation_importance(model, test[features], test.wrc_plus,
                        scoring='neg_mean_absolute_error', n_repeats=12, random_state=42, n_jobs=1)
            importance = pd.DataFrame({'feature':features, 'mae_increase':values.importances_mean,
                         'repeat_std':values.importances_std, 'holdout_season':year})
    if not records:
        raise ValueError('No chronological holdout seasons available.')
    return pd.concat(records,ignore_index=True), pd.DataFrame(scores), importance


def score_application(panel, config):
    year = config['application_season']
    train = panel[(panel.season >= config['start_season']) & (panel.season < year)]
    current = panel[panel.season == year].copy()
    if current.empty:
        raise ValueError(f'No application-season data for {year}.')
    model, features = fit_expected(train, config['min_train_pa'])
    current['xwrc'] = model.predict(current[features])
    current['gap_actual_minus_expected'] = performance_gap(current.wrc_plus,current.xwrc)
    current['missing_feature_fraction'] = current[features].isna().mean(axis=1)
    current['eligible_ranking'] = (current.pa >= config['min_rank_pa']) & (current.missing_feature_fraction <= .4)
    current['train_end'] = year-1
    current['features_used'] = ','.join(features)
    return current, model, features
