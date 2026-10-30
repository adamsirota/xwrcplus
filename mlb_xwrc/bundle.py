"""Prepare only the read-only artifacts needed by the hosted dashboard."""
import json
import shutil
from pathlib import Path
import pandas as pd
from .data import load_history, write_json

BUNDLED_FILES = ['expected_metrics.csv','forecast_metrics.csv','feature_importance.csv',
                 'aging_curve.csv','coverage.csv','sensitivity.csv','report.md',
                 'feature_importance.png','aging_curve.png','actual_vs_expected.png',
                 'performance_gaps.png','career_forecasts.png']


def bundle_findings(root, config):
    root = Path(root)
    source = root/config['results_dir']
    year = int(config['application_season'])
    names = BUNDLED_FILES+[f'expected_{year}.csv',f'forecasts_{year+1}.csv']
    for name in names+['run_manifest.json']:
        if not (source/name).is_file():
            raise ValueError(f'Missing {name}; run the analysis before bundling findings.')
    manifest = json.loads((source/'run_manifest.json').read_text(encoding='utf-8-sig'))
    history = load_history(pd.read_csv(root/'data/processed/player_seasons.csv'),root)
    public_config = {key:manifest['config'][key] for key in
                     ['application_season','start_season','min_train_pa','min_rank_pa','gap_threshold']}
    public_config['data_label'] = manifest['config'].get('data_label',manifest.get('data_label','Saved MLB analysis'))
    public_manifest = {key:manifest[key] for key in
                       ['created_utc','provenance','contact_join','forecast_interval_radius']}
    public_manifest['config'] = public_config
    target = root/'findings'
    target.mkdir(exist_ok=True)
    for name in names:
        shutil.copy2(source/name,target/name)
    history[['player_id','season','pa','wrc_plus']].to_csv(target/'player_history.csv',index=False)
    write_json(target/'run_manifest.json',public_manifest)
    print(f'Prepared hosted findings: {target}',flush=True)
    return target
