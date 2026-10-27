"""Next-season calibration and reliability-shrunk conditional age effects."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .model import metrics

PREDICTORS = ['xwrc','wrc_plus','history_wrc']


def add_history(frame, history):
    frame = frame.copy()
    values = []
    for row in frame.itertuples():
        past = history[(history.player_id==row.player_id) &
                       (history.season <= row.season) & (history.season >= row.season-2)]
        weights = past.pa * .5 ** (row.season-past.season)
        values.append(float(np.average(past.wrc_plus, weights=weights)))
    frame['history_wrc'] = values
    return frame


def consecutive_pairs(current, panel, min_pa):
    future = panel[['player_id','season','age','pa','wrc_plus']].rename(columns={
        'season':'target_season','age':'target_age','pa':'target_pa','wrc_plus':'actual'})
    current = current.copy()
    current['target_season'] = current.season + 1
    pairs = current.merge(future, on=['player_id','target_season'], validate='one_to_one')
    pairs = pairs[(pairs.pa >= min_pa) & (pairs.target_pa >= min_pa)].copy()
    # Reported season ages can be fractional, so require broadly consistent +1 aging.
    pairs = pairs[(pairs.target_age-pairs.age).between(.5,1.5)]
    pairs['pair_weight'] = 2 / (1/pairs.pa + 1/pairs.target_pa)
    return pairs


def fit_forecast(pairs):
    if len(pairs) < 50:
        raise ValueError('Need at least 50 eligible consecutive-season pairs for forecast calibration.')
    reg = Ridge(alpha=100)
    weights = pairs.pair_weight / pairs.pair_weight.mean()
    reg.fit(pairs[PREDICTORS], pairs.actual, sample_weight=weights)
    residual = pairs.actual.to_numpy() - reg.predict(pairs[PREDICTORS])
    records = []
    for age in range(16,56):
        kernel = np.maximum(0, 1 - np.abs(pairs.age.to_numpy()-age)/3)
        local_weights = pairs.pair_weight.to_numpy()*kernel
        mass = local_weights.sum()
        # A zero-effect prior equal to 30 reliable 500-PA season pairs prevents tail spikes.
        adjustment = float(np.sum(local_weights*residual)/(mass+15000))
        records.append({'age':age,'age_adjustment':adjustment,
                        'pair_count':int((kernel>0).sum()), 'weighted_pa':float(mass),
                        'effective_pairs':float(mass**2/np.sum(local_weights**2)) if mass else 0})
    curve = pd.DataFrame(records)
    return {'regression':reg,'curve':curve,'train_target_end':int(pairs.target_season.max())}, curve


def predict_forecast(model, frame):
    result = frame.copy()
    result['forecast_without_age'] = model['regression'].predict(frame[PREDICTORS])
    curve = model['curve']
    result['age_adjustment'] = np.interp(frame.age, curve.age, curve.age_adjustment)
    result['age_effective_pairs'] = np.interp(frame.age, curve.age, curve.effective_pairs)
    result['forecast'] = result.forecast_without_age + result.age_adjustment
    result['train_target_end'] = model['train_target_end']
    result['sparse_age_support'] = result.age_effective_pairs < 30
    return result


def prediction_radius(errors, level=.9):
    if errors.empty:
        return float('nan')
    absolute = np.abs(errors.actual-errors.forecast)
    quantile = min(1, np.ceil((len(absolute)+1)*level)/len(absolute))
    return float(np.quantile(absolute,quantile,method='higher'))


def backtest_forecasts(pairs, years):
    rows, scores = [], []
    for year in years:
        train, test = pairs[pairs.target_season < year], pairs[pairs.target_season==year]
        if len(train)<50 or test.empty:
            continue
        model, _ = fit_forecast(train)
        predictions = predict_forecast(model,test)
        prior = pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
        radius = prediction_radius(prior) if len(prior) else np.nan
        predictions['interval_low'] = predictions.forecast-radius
        predictions['interval_high'] = predictions.forecast+radius
        rows.append(predictions)
        for name, column in [('with_age','forecast'), ('without_age','forecast_without_age'),
                             ('previous_season','wrc_plus'), ('recent_history','history_wrc')]:
            item = {'target_season':year, 'model':name, **metrics(predictions.actual,predictions[column])}
            if name=='with_age' and np.isfinite(radius):
                item['interval_coverage'] = float(((predictions.actual>=predictions.interval_low) &
                                                   (predictions.actual<=predictions.interval_high)).mean())
                item['interval_radius'] = radius
            scores.append(item)
    if not rows:
        raise ValueError('No forecast backtest folds with sufficient prior-year pairs.')
    return pd.concat(rows,ignore_index=True), pd.DataFrame(scores)
