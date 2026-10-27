"""Reproducible orchestration. All outputs derive from validated input tables."""
import hashlib
import importlib.metadata
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .data import FEATURES, load_panel, load_history, write_json
from .model import expected_history, score_application, metrics
from .forecast import add_history, consecutive_pairs, fit_forecast, predict_forecast, backtest_forecasts, prediction_radius
from .report import create_report


def analyze_historical(panel, history, config, min_pa=None, exclude_2020=False):
    threshold = min_pa or config['min_train_pa']
    oof, expected_metrics, importance = expected_history(panel, config['start_season'],
                         config['application_season']-1, threshold, exclude_2020)
    calibration_history = history[history.season != 2020] if exclude_2020 else history
    pairs = consecutive_pairs(add_history(oof,calibration_history), panel, threshold)
    pairs = pairs[pairs.target_season < config['application_season']]
    if exclude_2020:
        pairs = pairs[(pairs.season!=2020) & (pairs.target_season!=2020)]
    years = range(config['start_season']+6,config['application_season'])
    backtests, forecast_metrics = backtest_forecasts(pairs,years)
    return oof, pairs, backtests, expected_metrics, forecast_metrics, importance


def run_analysis(config, root):
    root = Path(root)
    panel, provenance = load_panel(config,root)
    required = set(range(config['start_season'],config['application_season']+1))
    if not required.issubset(set(panel.season)):
        raise ValueError('Input must contain every requested training and application season.')
    history = load_history(panel,root)
    print('Evaluating chronological expected-performance and forecast folds...',flush=True)
    oof, pairs, backtests, expected_metrics, forecast_metrics, importance = analyze_historical(panel,history,config)
    current, model, features = score_application(panel,config)
    forecast_model, curve = fit_forecast(pairs)
    forecast_inputs = current[current.pa >= config['min_train_pa']].copy()
    forecasts = predict_forecast(forecast_model,add_history(forecast_inputs,history))
    forecasts['target_season'] = config['application_season']+1
    forecasts['interval_low'] = forecasts.forecast-prediction_radius(backtests)
    forecasts['interval_high'] = forecasts.forecast+prediction_radius(backtests)
    forecasts['projected_change'] = forecasts.forecast-forecasts.wrc_plus
    error_cutoff = float(np.quantile(np.abs(oof.wrc_plus-oof.xwrc),.9))
    current['candidate_gap'] = current.eligible_ranking & (current.gap_actual_minus_expected.abs()>=config['gap_threshold'])
    current['gap_exceeds_historical_p90_error'] = current.eligible_ranking & (current.gap_actual_minus_expected.abs()>error_cutoff)
    # Missingness is reported for the complete collected population, not only qualifiers.
    coverage = []
    for year, group in panel.groupby('season'):
        row = {'season':int(year),'players':len(group),
               'training_eligible':int((group.pa>=config['min_train_pa']).sum()),
               'ranking_pa_eligible':int((group.pa>=config['min_rank_pa']).sum())}
        row['all_contact_features_missing'] = int(group[['exit_velocity','launch_angle','barrel_rate','hard_hit_rate']].isna().all(axis=1).sum())
        if 'mlbam_id' in group:
            row['unmapped_mlbam'] = int(group.mlbam_id.isna().sum())
        row.update({f'{col}_missing':float(group[col].isna().mean()) for col in FEATURES})
        coverage.append(row)
    coverage = pd.DataFrame(coverage)
    sensitivity = []
    if config['sensitivity']:
        for name, threshold, exclude in [('exclude_2020',config['min_train_pa'],True), ('min_200_pa',200,False)]:
            print(f'Sensitivity: {name}',flush=True)
            _, _, tested, expected, _, _ = analyze_historical(panel,history,config,threshold,exclude)
            sensitivity.append({'scenario':name,'task':'forecast',**metrics(tested.actual,tested.forecast)})
            matched = backtests.merge(tested[['player_id','target_season']],
                                      on=['player_id','target_season'],validate='one_to_one')
            sensitivity.append({'scenario':name+'_primary_same_rows','task':'forecast',
                                **metrics(matched.actual,matched.forecast)})
            held = expected[(expected.season==config['application_season']-1)&(expected.model=='xgboost')].iloc[0]
            sensitivity.append({'scenario':name,'task':'same_season',**held[['mae','rmse','r2','n']].to_dict()})
    sensitivity = pd.DataFrame(sensitivity)
    out = root/config['results_dir']
    out.mkdir(parents=True,exist_ok=True)
    artifacts = {'expected_backtest_predictions.csv':oof,'expected_metrics.csv':expected_metrics,
                 'forecast_backtest_predictions.csv':backtests,'forecast_metrics.csv':forecast_metrics,
                 'feature_importance.csv':importance,'aging_curve.csv':curve,'coverage.csv':coverage,
                 'sensitivity.csv':sensitivity,
                 f'expected_{config["application_season"]}.csv':current.sort_values('gap_actual_minus_expected'),
                 f'forecasts_{config["application_season"]+1}.csv':forecasts.sort_values('forecast',ascending=False)}
    for name, frame in artifacts.items():
        frame.to_csv(out/name,index=False)
    processed = root/'data/processed'
    processed.mkdir(parents=True,exist_ok=True)
    panel.to_csv(processed/'player_seasons.csv',index=False)
    inputs = sorted((root/'data/raw').glob('*.csv'))
    if config.get('input_csv'):
        inputs.append(root/config['input_csv'])
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,'python':sys.version,
                'features':features,'xgboost_parameters':model.get_params(),
                'forecast_training_target_end':forecast_model['train_target_end'],
                'expected_training_end':config['application_season']-1,
                'gap_historical_p90_error':error_cutoff,
                'forecast_coefficients':dict(zip(['xwrc','actual','recent_history'],forecast_model['regression'].coef_.tolist())),
                'forecast_intercept':float(forecast_model['regression'].intercept_),
                'forecast_interval_radius':prediction_radius(backtests),
                'provenance':provenance['source'],'contact_join':provenance['contact_join'],
                'input_sha256':{str(p.relative_to(root)) if p.is_relative_to(root) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                'versions':{pkg:importlib.metadata.version(pkg) for pkg in ['numpy','pandas','scikit-learn','xgboost','matplotlib','pybaseball']}}
    write_json(out/'run_manifest.json',manifest)
    model.save_model(out/'expected_model.json')
    write_json(out/'forecast_calibration.json', {k:manifest[k] for k in ['forecast_coefficients','forecast_intercept','forecast_interval_radius']})
    create_report(out,config,panel,history,current,forecasts,importance,curve,expected_metrics,
                  forecast_metrics,coverage,sensitivity,manifest)
    print(f'Finished: {out / "report.md"}',flush=True)
    return out
