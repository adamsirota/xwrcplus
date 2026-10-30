"""Read-only Flask views over completed analysis artifacts."""
import csv
import json
import math
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from flask import Flask, abort, g, make_response, render_template, request, send_file, url_for
from .data import load_history

VIEWS = {'all':'All hitters', 'under':'Below expected', 'over':'Above expected',
         'rebound':'Rebound signals', 'regression':'Regression signals'}
SORTS = {'forecast':('forecast',False), 'rebound':('projected_change',False),
         'regression':('projected_change',True), 'shortfall':('gap_actual_minus_expected',True),
         'excess':('gap_actual_minus_expected',False), 'name':('name',True)}
FIGURES = {'actual_vs_expected','feature_importance','performance_gaps','aging_curve','career_forecasts'}


def folded(value):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(value)) if not unicodedata.combining(c)).casefold()


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def scatter_chart(frame):
    points = frame.dropna(subset=['wrc_plus','xwrc'])
    low = math.floor(min(points.wrc_plus.min(),points.xwrc.min(),50)/25)*25 if len(points) else 50
    high = math.ceil(max(points.wrc_plus.max(),points.xwrc.max(),150)/25)*25 if len(points) else 150
    span = max(25,high-low)
    x = lambda value: 44+(value-low)/span*258
    y = lambda value: 260-(value-low)/span*218
    return {'points':[dict(x=x(r.wrc_plus),y=y(r.xwrc),name=r.name,
                          actual=r.wrc_plus,expected=r.xwrc,gap=r.gap_actual_minus_expected)
                      for r in points.itertuples()],
            'ticks':[dict(value=v,x=x(v),y=y(v)) for v in range(low,high+1,25)],
            'line':[x(low),y(low),x(high),y(high)]}


def career_chart(history, player, application):
    years = history.season.astype(int).tolist()
    earliest = min(years+[application])
    last = application+1
    values = history.wrc_plus.tolist()+[player['wrc_plus'],player['xwrc']]
    for col in ['forecast','interval_low','interval_high']:
        if pd.notna(player.get(col)):
            values.append(player[col])
    low, high = math.floor(min(values+[100])/25)*25-25, math.ceil(max(values+[100])/25)*25+25
    x = lambda value: 56+(value-earliest)/max(1,last-earliest)*656
    y = lambda value: 250-(value-low)/(high-low)*212
    step = max(1,math.ceil((last-earliest)/9))
    ticks = sorted(set(list(range(earliest,last+1,step))+[application,last]))
    return {'points':' '.join(f'{x(r.season):.2f},{y(r.wrc_plus):.2f}' for r in history.itertuples()),
            'observed':[dict(x=x(r.season),y=y(r.wrc_plus),season=int(r.season),value=r.wrc_plus) for r in history.itertuples()],
            'years':[dict(value=v,x=x(v)) for v in ticks],
            'ticks':[dict(value=v,y=y(v)) for v in range(low,high+1,25)],
            'baseline':y(100),'expected':dict(x=x(application),y=y(player['xwrc'])),
            'forecast':dict(x=x(last),y=y(player['forecast']),low=y(player['interval_low']),high=y(player['interval_high']))
                       if pd.notna(player.get('forecast')) else None}


def create_app(config_path='config.json'):
    app = Flask(__name__)
    config_path = Path(config_path).resolve()

    @lru_cache(maxsize=1)
    def snapshot():
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        root = config_path.parent
        out = (root/config['results_dir']).resolve()
        if not (out/'run_manifest.json').exists() and (root/'findings/run_manifest.json').exists():
            out = root/'findings'
        manifest = json.loads((out/'run_manifest.json').read_text(encoding='utf-8-sig'))
        year = int(config['application_season'])
        if manifest['config']['application_season'] != year:
            raise ValueError('The saved findings do not match the configured application season.')
        def read(name):
            return pd.read_csv(out/name)
        expected, forecasts = read(f'expected_{year}.csv'), read(f'forecasts_{year+1}.csv')
        columns = ['player_id','forecast','projected_change','age_adjustment','interval_low','interval_high',
                   'sparse_age_support','age_effective_pairs']
        players = expected.merge(forecasts[columns],on='player_id',how='left',validate='one_to_one')
        expected_metrics, forecast_metrics = read('expected_metrics.csv'), read('forecast_metrics.csv')
        coverage = read('coverage.csv')
        last_test = int(forecast_metrics.target_season.max())
        latest = forecast_metrics[forecast_metrics.target_season==last_test].set_index('model')
        expected_latest = expected_metrics[(expected_metrics.season==year-1)&(expected_metrics.model=='xgboost')].iloc[0]
        history_path = root/'data/processed/player_seasons.csv'
        if (out/'player_history.csv').exists():
            history = pd.read_csv(out/'player_history.csv')
        else:
            history = load_history(pd.read_csv(history_path),root) if history_path.exists() else pd.DataFrame(columns=['player_id','season','wrc_plus','pa'])
        return dict(root=root,out=out,config=config,manifest=manifest,year=year,players=players,
                    forecasts=forecasts,history=history,coverage=coverage,
                    expected_metrics=expected_metrics,forecast_metrics=forecast_metrics,
                    importance=read('feature_importance.csv'),curve=read('aging_curve.csv'),
                    sensitivity=read('sensitivity.csv') if (out/'sensitivity.csv').exists() and (out/'sensitivity.csv').stat().st_size>4 else pd.DataFrame(),
                    date=datetime.fromisoformat(manifest['created_utc']).astimezone(ZoneInfo('America/New_York')).strftime('%b %d, %Y'),
                    data_label=manifest['config'].get('data_label',manifest.get('data_label','Saved MLB analysis')),
                    total_seasons=int(coverage.players.sum()),eligible=int(players.eligible_ranking.sum()),
                    forecast_mae=float(latest.loc['with_age','mae']),expected_mae=float(expected_latest.mae),
                    previous_mae=float(latest.loc['previous_season','mae']),recent_mae=float(latest.loc['recent_history','mae']),
                    age_gain=float(latest.loc['without_age','mae']-latest.loc['with_age','mae']),test_year=last_test)

    @app.before_request
    def load_snapshot():
        if request.endpoint=='static':
            return None
        try:
            g.findings = snapshot()
        except (OSError,ValueError,KeyError,IndexError) as exc:
            app.logger.warning('Dashboard artifacts unavailable: %s',exc)
            return render_template('error.html',title='Generate the findings',
                message='The saved analysis is missing or incomplete. Run the analysis, then reload this page.',
                command=f'python -m mlb_xwrc run --config "{config_path.name}"'),503

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        return response

    @app.context_processor
    def common():
        return dict(s=getattr(g,'findings',None),nav=request.endpoint)

    @app.template_filter('num')
    def number(value, digits=1, signed=False):
        if value is None or pd.isna(value):
            return '—'
        return f'{value:+,.{digits}f}' if signed else f'{value:,.{digits}f}'

    def filtered_players():
        state = dict(q=request.args.get('q','').strip()[:100],view=request.args.get('view','all'),
                     sort=request.args.get('sort',''),min_pa=request.args.get('min_pa',g.findings['config'].get('min_rank_pa',300)))
        if state['view'] not in VIEWS:
            abort(400,description='Choose a valid candidate view.')
        try:
            state['min_pa']=int(state['min_pa'])
            page=int(request.args.get('page','1'))
        except (ValueError,TypeError):
            abort(400,description='Plate appearances and page must be whole numbers.')
        if not g.findings['config'].get('min_train_pa',100)<=state['min_pa']<=1000 or page<1:
            abort(400,description='Use a supported PA threshold and a positive page number.')
        default_sort={'all':'forecast','under':'shortfall','over':'excess','rebound':'rebound','regression':'regression'}
        state['sort']=state['sort'] or default_sort[state['view']]
        if state['sort'] not in SORTS:
            abort(400,description='Choose a valid sort order.')
        frame=g.findings['players']
        frame=frame[(frame.pa>=state['min_pa'])&(frame.missing_feature_fraction<=.4)&frame.forecast.notna()].copy()
        if state['q']:
            frame=frame[frame.name.map(folded).str.contains(folded(state['q']),regex=False)]
        cutoff=g.findings['config'].get('gap_threshold',10)
        if state['view']=='under':
            frame=frame[frame.gap_actual_minus_expected<=-cutoff]
        elif state['view']=='over':
            frame=frame[frame.gap_actual_minus_expected>=cutoff]
        elif state['view']=='rebound':
            frame=frame[frame.projected_change>=cutoff]
        elif state['view']=='regression':
            frame=frame[frame.projected_change<=-cutoff]
        column,ascending=SORTS[state['sort']]
        return frame.sort_values([column,'player_id'],ascending=[ascending,True]),state,page

    @app.get('/')
    def index():
        frame,state,page=filtered_players()
        page_count=max(1,math.ceil(len(frame)/20))
        page=min(page,page_count)
        links={key:url_for('index',**{**state,'view':key,'sort':''}) for key in VIEWS}
        return render_template('explorer.html',players=records(frame.iloc[(page-1)*20:page*20]),
            state=state,views=VIEWS,links=links,count=len(frame),page=page,pages=page_count,
            previous=url_for('index',**state,page=page-1),next=url_for('index',**state,page=page+1),
            export_url=url_for('export_players',**state),scatter=scatter_chart(frame),
            rebound=records(frame.nlargest(1,'projected_change')),regression=records(frame.nsmallest(1,'projected_change')))

    @app.get('/players/<int:player_id>')
    def player(player_id):
        match=g.findings['players'].query('player_id == @player_id')
        if match.empty:
            abort(404,description='This player is not in the saved analysis.')
        person=records(match)[0]
        history=g.findings['history']
        history=history[(history.player_id==player_id)&(history.season<=g.findings['year'])].sort_values('season')
        skills=[('Walk rate','bb_rate','rate'),('Strikeout rate','k_rate','rate'),('Barrel rate','barrel_rate','rate'),
                ('Hard-hit rate','hard_hit_rate','rate'),('Exit velocity','exit_velocity','mph'),('Launch angle','launch_angle','deg')]
        return render_template('player.html',player=person,career=records(history),
                               chart=career_chart(history,person,g.findings['year']),skills=skills)

    @app.get('/validation')
    def validation():
        frame=g.findings['forecast_metrics']
        folds=frame.pivot(index='target_season',columns='model',values='mae').reset_index()
        return render_template('validation.html',folds=records(folds),
            expected=records(g.findings['expected_metrics']),sensitivity=records(g.findings['sensitivity']))

    @app.get('/methodology')
    def methodology():
        return render_template('methodology.html',coverage=records(g.findings['coverage']))

    @app.get('/export/players.csv')
    def export_players():
        frame,_,_=filtered_players()
        # Prevent spreadsheet programs from treating untrusted names as formulas.
        frame=frame.copy()
        for col in frame.select_dtypes(include=['object','string']).columns:
            frame[col]=frame[col].map(lambda v:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v)
        response=make_response('\ufeff'+frame.to_csv(index=False,quoting=csv.QUOTE_MINIMAL))
        response.headers['Content-Type']='text/csv; charset=utf-8'
        response.headers['Content-Disposition']='attachment; filename=filtered_players.csv'
        return response

    @app.get('/downloads/<kind>')
    def download_result(kind):
        year=g.findings['year']
        names={'expected':f'expected_{year}.csv','forecasts':f'forecasts_{year+1}.csv',
               'metrics':'forecast_metrics.csv','report':'report.md'}
        if kind not in names:
            abort(404)
        path=g.findings['out']/names[kind]
        if not path.is_file():
            abort(404)
        return send_file(path,as_attachment=True,download_name=path.name)

    @app.get('/charts/<name>.png')
    def figure(name):
        if name not in FIGURES:
            abort(404)
        path=g.findings['out']/f'{name}.png'
        if not path.is_file():
            abort(404)
        return send_file(path,mimetype='image/png')

    @app.errorhandler(400)
    @app.errorhandler(404)
    def error_page(error):
        return render_template('error.html',title='Check your filters' if error.code==400 else 'Page not found',
                               message=error.description,command=None),error.code

    return app
