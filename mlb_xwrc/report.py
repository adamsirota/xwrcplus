"""Publication of transparent CSV summaries, figures, and a Markdown research report."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def table(frame):
    if frame.empty:
        return '(No eligible rows.)'
    cols = list(frame)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        values = [f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/') for v in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def create_report(out, config, panel, history, current, forecasts, importance, curve,
                  expected_metrics, forecast_metrics, coverage, sensitivity, manifest):
    out = Path(out)
    label = config['data_label']
    plt.rcParams.update({'figure.dpi':140, 'axes.spines.top':False, 'axes.spines.right':False,
                        'font.size':10, 'axes.titleweight':'bold'})
    def save(fig, name):
        fig.suptitle(label, fontsize=9, color='#555555')
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(out/name, bbox_inches='tight')
        plt.close(fig)

    imp = importance.sort_values('mae_increase')
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(imp.feature, imp.mae_increase, xerr=imp.repeat_std, color='#246875')
    ax.set(xlabel='Increase in held-out MAE after shuffling (wRC+ points)', title='Held-out feature importance')
    save(fig,'feature_importance.png')

    ranked = current[current.eligible_ranking].copy()
    fig, ax = plt.subplots(figsize=(7,6))
    color_limit = max(1, ranked.gap_actual_minus_expected.abs().max())
    dots = ax.scatter(ranked.wrc_plus, ranked.xwrc, c=ranked.gap_actual_minus_expected,
                      cmap='coolwarm',vmin=-color_limit,vmax=color_limit,s=25,alpha=.8)
    bounds = [min(ranked.wrc_plus.min(),ranked.xwrc.min())-5,
              max(ranked.wrc_plus.max(),ranked.xwrc.max())+5] if len(ranked) else [0,200]
    ax.plot(bounds,bounds,color='#888888',linestyle='--')
    for row in ranked.reindex(ranked.gap_actual_minus_expected.abs().sort_values(ascending=False).index).head(6).itertuples():
        ax.annotate(row.name,(row.wrc_plus,row.xwrc),fontsize=8,xytext=(3,3),textcoords='offset points')
    ax.set(xlabel='Actual wRC+',ylabel='Model expected wRC+',title=f'{config["application_season"]}: actual and expected')
    fig.colorbar(dots,ax=ax,label='Actual minus expected')
    save(fig,'actual_vs_expected.png')

    gaps = pd.concat([ranked.nsmallest(10,'gap_actual_minus_expected'),
                      ranked.nlargest(10,'gap_actual_minus_expected')]).drop_duplicates('player_id')
    gaps = gaps.sort_values('gap_actual_minus_expected')
    fig, ax = plt.subplots(figsize=(9,7))
    ax.barh(gaps.name, gaps.gap_actual_minus_expected,
            color=np.where(gaps.gap_actual_minus_expected>0,'#b75d49','#246875'))
    ax.axvline(0,color='#888888',linewidth=1)
    ax.set(xlabel='Actual minus expected wRC+', title='Largest model discrepancies (eligible hitters)')
    save(fig,'performance_gaps.png')

    fig, (ax, support) = plt.subplots(2,1,figsize=(8,6),sharex=True,height_ratios=[2,1])
    supported = curve.effective_pairs >= 30
    ax.plot(curve.age,curve.age_adjustment,color='#999999',linestyle='--',label='Sparse / shrunk to zero')
    ax.plot(curve.age,np.where(supported,curve.age_adjustment,np.nan),color='#246875',label='At least 30 effective pairs')
    ax.legend(fontsize=8)
    ax.axhline(0,color='#888888',linestyle='--')
    ax.set(ylabel='wRC+ point adjustment',title='Conditional aging adjustment after performance calibration')
    support.bar(curve.age,curve.pair_count,color='#9bbfc2')
    support.set(xlabel='Age in input season',ylabel='Nearby pairs',xlim=(18,45))
    save(fig,'aging_curve.png')

    eligible = forecasts[forecasts.eligible_ranking].copy()
    selected = pd.concat([eligible.nlargest(3,'projected_change'),eligible.nsmallest(3,'projected_change')]).drop_duplicates('player_id')
    fig, axes = plt.subplots(3,2,figsize=(12,10),squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, row in zip(axes.flat,selected.itertuples()):
        ax.set_visible(True)
        career = history[(history.player_id==row.player_id) & (history.season<=config['application_season'])].sort_values('season')
        ax.plot(career.season,career.wrc_plus,'o-',markersize=3,color='#555555',label='Observed season wRC+')
        ax.scatter([config['application_season']],[row.xwrc],marker='D',color='#246875',label='Expected')
        yerr = [[row.forecast-row.interval_low],[row.interval_high-row.forecast]]
        ax.errorbar([config['application_season']+1],[row.forecast],yerr=yerr,
                    fmt='*',markersize=10,color='#b75d49',capsize=3,label='Forecast / 90% empirical range')
        ax.axhline(100,color='#bbbbbb',linestyle=':',linewidth=1)
        ax.set(title=row.name,ylabel='wRC+',xlabel='Season')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    visible = [a for a in axes.flat if a.get_visible()]
    if visible:
        visible[0].legend(fontsize=7)
    save(fig,'career_forecasts.png')

    last_expected = expected_metrics[expected_metrics.season==config['application_season']-1]
    headline = eligible[['name','age','pa','wrc_plus','xwrc','forecast','age_adjustment','projected_change','sparse_age_support']]
    gaps_table = ranked[['name','pa','wrc_plus','xwrc','gap_actual_minus_expected']]
    latest_year = forecast_metrics.target_season.max()
    latest = forecast_metrics[forecast_metrics.target_season==latest_year].set_index('model')
    age_gain = latest.loc['without_age','mae']-latest.loc['with_age','mae']
    content = f'''# MLB expected hitting and next-season projections

**Dataset: {label}**

Generated at {manifest['created_utc']}. This project-specific xwRC+ is an XGBoost estimate, not an official MLB or FanGraphs statistic. Same-season estimates and next-season forecasts answer different questions.

## Evaluation

Final historical same-season holdout:

{table(last_expected)}

Next-season chronological backtests (only earlier target years train each fold):

{table(forecast_metrics)}

The age-adjusted model should be judged against both the unadjusted calibration and simple history baselines. A worse score is evidence against the adjustment; no improvement is assumed. R-squared can be negative. Metrics describe returning hitters meeting the PA threshold, not everyone on an MLB roster.

In the final {latest_year} forecast holdout, age-adjusted MAE was {latest.loc['with_age','mae']:.2f} points, versus {latest.loc['previous_season','mae']:.2f} for previous-season wRC+ and {latest.loc['recent_history','mae']:.2f} for recent history. The age adjustment reduced MAE by {age_gain:.2f} points relative to calibration without age; a negative reduction means it worsened predictions. Check all folds rather than attributing the full forecasting improvement to aging.

## Candidate signals

Candidates below have at least {config['min_rank_pa']} PA and at most 40% missing selected features. The gap is actual minus expected: positive means above-model performance and negative below-model performance. The {config['gap_threshold']}-point candidate cutoff is a configurable screening rule, not statistical significance. See CSV flags for discrepancies exceeding the historical 90th percentile absolute model error.

Largest below-model 2026 results:

{table(gaps_table.nsmallest(10,'gap_actual_minus_expected'))}

Largest above-model 2026 results:

{table(gaps_table.nlargest(10,'gap_actual_minus_expected'))}

Largest projected improvements:

{table(headline.nlargest(10,'projected_change'))}

Largest projected declines:

{table(headline.nsmallest(10,'projected_change'))}

## Figures

![Feature importance](feature_importance.png)

![Actual versus expected](actual_vs_expected.png)

![Performance gaps](performance_gaps.png)

![Aging adjustment](aging_curve.png)

![Career trajectories](career_forecasts.png)

## Methods and limitations

- Training window: {config['start_season']}–{config['application_season']-1}. Chronological expected-performance predictions start after three training seasons. XGBoost parameters are fixed in advance; no holdout tuning is performed. Native missing-value handling avoids global imputation.
- Features are selected from a fixed skills allowlist with at least 70% coverage in each training fold. K and BB rates are mandatory. Target derivatives such as wOBA, OPS, and wRC are excluded. Permutation importance measures association; correlated features can share importance, and repeat standard deviations are not confidence intervals.
- Forecast calibration is a PA-weighted ridge regression using chronological xwRC+, current actual wRC+, and a three-year PA/recency-weighted history. Historical calibration inputs use out-of-time xwRC+, never fitted values. Final calibration only uses target seasons through {manifest['forecast_training_target_end']}; 2026 actual outcomes are inputs for 2027, not calibration labels.
- Aging is the smoothed residual next-season change after that calibration, estimated from consecutive seasons with a triangular age kernel (distance <3 years). Harmonic-mean PA weights reduce unreliable pairs; a zero-effect prior of 15,000 PA shrinks sparse ages. This is a conditional observational adjustment, not a causal biological aging curve. It does not add a second age term to an age-aware model.
- Prediction ranges use the finite-sample 90th percentile of absolute earlier out-of-time forecast errors. Coverage is checked only on subsequent years. They are empirical ranges conditional on returning and meeting the PA threshold, with no formal coverage guarantee under distribution shift. They do not predict injuries, playing time, or retirement.
- Discrepancies can reflect omitted skills, park effects, measurement error, and model bias as well as randomness. Large discrepancies do not establish that a hitter was lucky or will regress. Forecast uncertainty is much larger than many player-to-player differences.
- Career charts show observed individual seasons from the downloaded history; historical coverage begins at {int(history.season.min())}. No official full-career aggregate wRC+ is reconstructed by averaging yearly indexes.
- Selection of returning players biases age estimates. League changes, tracking changes, and the shortened 2020 season can shift relationships. See sensitivity results below.

## Data audit

{table(coverage)}

Source: {manifest['provenance']}. Contact join: {manifest['contact_join']}. Unmatched identifiers remain missing; raw caches and per-download metadata are preserved. The manifest records input hashes and dependency versions. History is not silently treated as a complete career.

## Sensitivity analysis

{table(sensitivity) if len(sensitivity) else 'Not requested for this run.'}

Sensitivity runs refit the model and calibration with the indicated exclusions/PA thresholds. Compare errors on their reported populations; changing PA thresholds changes the population as well as reliability.

## Sources

- [pybaseball batting data](https://github.com/jldbc/pybaseball/blob/master/docs/batting_stats.md)
- [pybaseball Statcast leaderboards](https://github.com/jldbc/pybaseball/blob/master/docs/statcast_batter.md)
- [FanGraphs wRC+ definition](https://library.fangraphs.com/offense/wrc/)
- [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
'''
    (out/'report.md').write_text(content,encoding='utf-8')
