import json
from mlb_xwrc.pipeline import run_analysis


def test_offline_pipeline_exports_labeled_results(tmp_path, panel):
    panel.to_csv(tmp_path/'input.csv',index=False)
    config = dict(start_season=2016, application_season=2026, history_start=1990,
                  min_train_pa=100, min_rank_pa=300, input_csv='input.csv',
                  data_label='SYNTHETIC TEST DATA - NO MLB FINDINGS',
                  results_dir='results-test', sensitivity=False, gap_threshold=10)
    run_analysis(config,tmp_path)
    out = tmp_path/'results-test'
    assert (out/'forecasts_2027.csv').exists()
    assert 'SYNTHETIC TEST DATA' in (out/'report.md').read_text(encoding='utf-8')
    assert len(list(out.glob('*.png'))) >= 5
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['forecast_training_target_end'] <= 2025
