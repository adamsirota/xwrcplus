import json
import pandas as pd
import pytest
from mlb_xwrc.web import create_app


@pytest.fixture
def client(tmp_path):
    out = tmp_path/'results'
    out.mkdir()
    config = {'application_season':2026,'start_season':2016,'min_train_pa':100,
              'min_rank_pa':300,'gap_threshold':10,'results_dir':'results'}
    (tmp_path/'config.json').write_text(json.dumps(config))
    manifest = {'config':config,'created_utc':'2026-10-27T19:17:43+00:00',
                'forecast_interval_radius':40,'data_label':'SYNTHETIC TEST DATA',
                'provenance':'Test fixture','contact_join':'Test IDs'}
    (out/'run_manifest.json').write_text(json.dumps(manifest))
    expected = pd.DataFrame([
        dict(player_id=1,name='Ana Peña',age=24,season=2026,pa=500,wrc_plus=80,xwrc=110,
             gap_actual_minus_expected=-30,missing_feature_fraction=0,eligible_ranking=True),
        dict(player_id=2,name='Ben Jones',age=32,season=2026,pa=450,wrc_plus=140,xwrc=110,
             gap_actual_minus_expected=30,missing_feature_fraction=0,eligible_ranking=True),
        dict(player_id=3,name='<script>alert(1)</script>',age=25,season=2026,pa=150,wrc_plus=100,xwrc=100,
             gap_actual_minus_expected=0,missing_feature_fraction=0,eligible_ranking=False)])
    expected.to_csv(out/'expected_2026.csv',index=False)
    expected.assign(forecast=[115,105,100],projected_change=[35,-35,0],
        age_adjustment=[3,-3,0],interval_low=[75,65,60],interval_high=[155,145,140],
        sparse_age_support=[False,False,True],age_effective_pairs=[100,200,5])\
        .to_csv(out/'forecasts_2027.csv',index=False)
    for task in ['expected','forecast']:
        year_column = 'season' if task=='expected' else 'target_season'
        labels = ['xgboost','training_mean'] if task=='expected' else ['with_age','without_age','previous_season','recent_history']
        pd.DataFrame([{year_column:2025,'model':name,'mae':18+i,'rmse':24,'r2':.3,'n':100}
                      for i,name in enumerate(labels)]).to_csv(out/f'{task}_metrics.csv',index=False)
    pd.DataFrame({'feature':['k_rate'],'mae_increase':[5],'repeat_std':[.2]}).to_csv(out/'feature_importance.csv',index=False)
    pd.DataFrame({'age':[24,32],'age_adjustment':[3,-3],'pair_count':[100,200],
                  'effective_pairs':[100,200]}).to_csv(out/'aging_curve.csv',index=False)
    pd.DataFrame({'season':[2026],'players':[3],'training_eligible':[3],
                  'ranking_pa_eligible':[2],'barrel_rate_missing':[0]}).to_csv(out/'coverage.csv',index=False)
    app = create_app(tmp_path/'config.json')
    app.config['TESTING'] = True
    return app.test_client()


def test_explorer_displays_saved_snapshot_and_eligible_players(client):
    page = client.get('/')
    assert page.status_code == 200
    assert b'Player explorer' in page.data
    assert b'Ana Pe' in page.data and b'Ben Jones' in page.data
    assert b'2027' in page.data and b'Oct 27, 2026' in page.data


def test_search_is_literal_and_accent_insensitive(client):
    page = client.get('/?q=pena')
    assert b'Ana Pe' in page.data and b'Ben Jones' not in page.data
    assert b'No players match' in client.get('/?q=.*').data


def test_candidate_views_keep_actual_minus_expected_sign(client):
    below = client.get('/?view=under')
    above = client.get('/?view=over')
    assert b'Ana Pe' in below.data and b'Ben Jones' not in below.data
    assert b'Ben Jones' in above.data and b'Ana Pe' not in above.data


def test_filtered_csv_matches_search(client):
    response = client.get('/export/players.csv?q=pena')
    assert response.status_code == 200
    assert 'attachment' in response.headers['Content-Disposition']
    assert b'Ana Pe' in response.data and b'Ben Jones' not in response.data


def test_player_detail_and_unknown_player(client):
    assert b'115.0' in client.get('/players/1').data
    assert b'90%' in client.get('/players/1').data
    assert client.get('/players/999').status_code == 404


def test_invalid_filters_and_download_boundaries(client):
    assert client.get('/?min_pa=oops').status_code == 400
    assert client.get('/?page=-1').status_code == 400
    assert client.get('/downloads/config.json').status_code == 404
    assert client.get('/charts/secret.png').status_code == 404
    assert client.get('/downloads/forecasts').status_code == 200


def test_templates_escape_player_names(client):
    page = client.get('/players/3')
    assert b'<script>alert(1)</script>' not in page.data
    assert b'&lt;script&gt;' in page.data


def test_validation_and_methodology_pages(client):
    assert client.get('/validation').status_code == 200
    assert b'previous' in client.get('/validation').data.lower()
    assert client.get('/methodology').status_code == 200


def test_missing_results_are_actionable(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'results_dir':'results','application_season':2026}))
    client = create_app(config).test_client()
    response = client.get('/')
    assert response.status_code == 503
    assert b'Generate the findings' in response.data

def test_pagination_does_not_truncate_filtered_export(client, tmp_path):
    out = tmp_path/'results'
    expected = pd.read_csv(out/'expected_2026.csv').iloc[[0]]
    forecast = pd.read_csv(out/'forecasts_2027.csv').iloc[[0]]
    expected = pd.concat([expected.assign(player_id=i,name=f'Player {i:02}') for i in range(1,46)],ignore_index=True)
    forecast = pd.concat([forecast.assign(player_id=i,name=f'Player {i:02}') for i in range(1,46)],ignore_index=True)
    expected.to_csv(out/'expected_2026.csv',index=False)
    forecast.to_csv(out/'forecasts_2027.csv',index=False)
    page = client.get('/?page=2').data.split(b'<tbody>')[1].split(b'</tbody>')[0]
    assert page.count(b'<tr>') == 20
    assert b'Player 21' in page and b'Player 01' not in page
    export = client.get('/export/players.csv?page=2').data.decode('utf-8-sig')
    assert len(export.splitlines()) == 46


def test_synthetic_dataset_label_is_preserved(client):
    assert b'SYNTHETIC TEST DATA' in client.get('/').data

def test_csv_export_neutralizes_formula_names(client, tmp_path):
    path = tmp_path/'results/expected_2026.csv'
    frame = pd.read_csv(path)
    frame.loc[frame.player_id==1,'name'] = '=HYPERLINK("https://example.invalid")'
    frame.to_csv(path,index=False)
    response = client.get('/export/players.csv')
    assert b"'=HYPERLINK" in response.data


def test_cli_serve_reuses_configuration_without_training(monkeypatch, tmp_path):
    import sys
    from mlb_xwrc import __main__
    config = tmp_path/'config.json'
    config.write_text('{}')
    called = []
    class Server:
        def run(self, **kwargs):
            called.append(kwargs)
    monkeypatch.setattr('mlb_xwrc.web.create_app',lambda path: Server())
    monkeypatch.setattr(sys,'argv',['mlb_xwrc','serve','--config',str(config),'--port','5055'])
    assert __main__.main() == 0
    assert called == [{'host':'127.0.0.1','port':5055,'debug':False,'use_reloader':False}]

def test_dashboard_uses_bundled_findings_on_a_fresh_checkout(client, tmp_path):
    (tmp_path/'results').rename(tmp_path/'findings')
    assert client.get('/').status_code == 200
    assert b'Ana Pe' in client.get('/').data

def test_bundled_snapshot_excludes_local_environment_metadata(client, tmp_path):
    from mlb_xwrc.bundle import BUNDLED_FILES, bundle_findings
    source = tmp_path/'results'
    for name in BUNDLED_FILES + ['expected_2026.csv','forecasts_2027.csv']:
        if not (source/name).exists():
            (source/name).write_text('fixture')
    processed = tmp_path/'data/processed'
    processed.mkdir(parents=True)
    pd.read_csv(source/'expected_2026.csv')[['player_id','season','pa','wrc_plus']].to_csv(processed/'player_seasons.csv',index=False)
    manifest = json.loads((source/'run_manifest.json').read_text())
    manifest['python'] = 'HIDDEN LOCAL ENVIRONMENT'
    manifest['config']['input_csv'] = 'HIDDEN LOCAL FILE'
    (source/'run_manifest.json').write_text(json.dumps(manifest))
    target = bundle_findings(tmp_path,json.loads((tmp_path/'config.json').read_text()))
    assert 'HIDDEN LOCAL' not in (target/'run_manifest.json').read_text()
    assert (target/'player_history.csv').exists()
    assert json.loads((target/'run_manifest.json').read_text())['config']['data_label'] == 'SYNTHETIC TEST DATA'
