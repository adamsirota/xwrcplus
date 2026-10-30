# MLB xwRC+ and 2027 hitting projections

A reproducible Python project for estimating 2026 expected hitting and evaluating 2027 breakout/regression signals. The expected-performance model is XGBoost; a separate calibrated forecasting step estimates next-season performance and a reliability-shrunk aging adjustment.

**xwRC+ here is a project-specific model estimate, not an official Statcast statistic.** A discrepancy is a research signal, not proof of luck or future improvement.

## Run the dashboard

The checked-in `findings/` snapshot lets a fresh clone display the completed analysis immediately. No live data download, API key, or model training is needed to view it.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:5000**. Use `python -m mlb_xwrc serve --port 5001` to choose another port. On macOS/Linux, create the environment with `python3 -m venv .venv` and use `.venv/bin/python`.

The explorer provides player search, candidate filters, sorting, PA thresholds, CSV exports, and individual career/forecast views. Model validation and research notes explain the backtests and limitations. All UI assets are local.

## Host from this GitHub repository

GitHub Pages serves static sites and cannot execute Flask. To run this app publicly, connect the repository to a Python web host. A minimal `render.yaml` is included for Render:

1. Create a **Web Service** from `adamsirota/xwrcplus`, or use Render's Blueprint flow to load `render.yaml`.
2. Build: `pip install -r requirements-web.txt`
3. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 60`

`.python-version` selects the latest Python 3.12 patch. The health check uses `/` and fails clearly if the findings are incomplete. Gunicorn is installed on Linux; Windows development uses Flask's local server. Hosting is configured but no external service is automatically created or deployed by this repository.

The tracked snapshot contains only the tables, charts, public provenance, and compact player history needed to display/download the findings. Raw source caches, virtual environments, trained model files, working notes, and local environment metadata are excluded from Git.

To refresh the published findings after a full analysis run:

```powershell
.\.venv\Scripts\python.exe -m mlb_xwrc bundle --config config.json
```

Review and commit the changed `findings/` files, then redeploy. The dashboard prefers locally regenerated `results/` when available; otherwise it uses `findings/`. Restart the server after updating a snapshot.

## Reproduce the full analysis (Windows PowerShell)

Use Python 3.10 or newer. From this folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m mlb_xwrc download --config config.json
.\.venv\Scripts\python.exe -m mlb_xwrc run --config config.json
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` for the remaining commands. No API key is required. A download can take several minutes. It resumes from saved CSVs; move a particular cache file aside to refresh it. Cached seasons are snapshots rather than automatically refreshed live data.

`download` exits nonzero if any source remains unavailable, and saves error details. Read them before interpreting results. `run` refuses missing training/application batting seasons. An unavailable historical-career table allows observed-window charts with explicit coverage labels. Missing Statcast data permits an explicitly reported discipline-only model if K and BB coverage passes validation.

## Source compatibility

pybaseball 2.2.7's `batting_stats` uses a legacy FanGraphs endpoint that returned HTTP 403 during development. The downloader first tries pybaseball, then reads the current public FanGraphs leaderboard API. It preserves paginated API responses and records that fallback in each table's metadata. Statcast contact tables and the Chadwick ID register use pybaseball. The current API is an upstream dependency and could change.

No login challenge, authentication, or paid export restriction is bypassed. If both provider routes fail, supply permitted CSV exports as described below.

## What runs

- Collect 2016–2025 historical seasons and 2026 application data. Download earlier batting seasons for career trajectories (1990–2015 by default).
- Join FanGraphs and MLBAM IDs using the Chadwick register; never match by player name.
- Train XGBoost on underlying plate discipline and contact-quality features. Use fixed parameters and expanding chronological holdouts; 2025 is the final historical same-season holdout. BB/K rates are mandatory; optional features need 70% training coverage. Features are chosen independently within each fold.
- Produce 2026 xwRC+ and `actual - expected` gaps. Exclude same-season wOBA, OPS, and target derivatives from this model.
- Train next-season calibration on historical out-of-time xwRC+, actual wRC+, and a three-year history weighted by PA and `0.5 ** years_ago`. Consecutive-season pairs need at least 100 PA in both seasons by default.
- Estimate age effects from the calibration's residuals using a triangular age kernel (within 3 years) and 15,000 PA of shrinkage toward zero. These are conditional age adjustments, not an unadjusted biological aging curve.
- Backtest target seasons 2022–2025 against previous-season and recent-history baselines and against calibration without age. Predict 2027 using 2026 inputs. Final calibration does not use 2026 as a target label.
- Estimate 90% empirical prediction ranges from earlier out-of-time absolute errors. Check coverage on subsequent backtest years; do not interpret these as guaranteed coverage or confidence intervals for true talent.
- Refit sensitivity runs excluding 2020 and raising the training/pair PA threshold to 200.

## Outputs

`results/report.md` includes findings, metrics, charts, and limitations. Other outputs:

| File | Contents |
| --- | --- |
| `expected_2026.csv` | Actual wRC+, xwRC+, gap and eligibility flags |
| `forecasts_2027.csv` | Prediction, age effect, empirical interval and support flags |
| `expected_metrics.csv` | Same-season chronological model/baseline errors |
| `forecast_metrics.csv` | Next-season chronological errors and available interval coverage |
| `*_backtest_predictions.csv` | Individual held-out predictions for auditing |
| `feature_importance.csv` | Held-out permutation MAE increases and repeat variation |
| `aging_curve.csv` | Conditional age effect and local sample support |
| `coverage.csv` | Per-season population and feature missingness |
| `sensitivity.csv` | Refit results excluding 2020 or raising PA thresholds |
| `run_manifest.json` | Configuration, input hashes, versions, selected features and coefficients |
| `expected_model.json` | Saved XGBoost estimator |
| `forecast_calibration.json` | Forecast coefficients, intercept and interval width; use with aging_curve.csv |

Five PNGs show feature importance, actual/expected hitting, discrepancies, conditional aging, and selected observed career trajectories with projections. Tables contain all collected players; use eligibility/support flags when interpreting them. Default displayed rankings require 300 PA and at most 40% missing selected features. The 10-point gap screen is a heuristic; a separate flag compares gaps with the historical 90th percentile absolute model error.

## Configuration

All paths in `config.json` are resolved relative to that config file. Adjust PA thresholds, the displayed gap threshold, and output directory there. The requested date window defaults to 2016–2025 training, 2026 application, and 2027 projection. Keep at least eight historical seasons for the forecast backtests. Fixed XGBoost settings are documented in `mlb_xwrc/model.py`; do not tune them on 2025 if presenting 2025 as an untouched test.

## Import local CSVs

Option 1: Place FanGraphs season exports in `data/raw/batting_2016.csv` through `batting_2026.csv` (one source aggregate row per player-season). Preserve IDfg, Season, Name, Age, PA, wRC+, BB%, and K%; optional O-Swing%, Contact%, and SwStr% improve the available feature set. An `IDfg` column must contain stable numeric FanGraphs IDs. This path joins to cached Savant contact data and the player register.

Option 2: Set `input_csv` in config.json to one combined file and set `data_label` to describe its provenance. Required canonical columns:

```text
player_id,season,name,age,pa,wrc_plus,bb_rate,k_rate
```

Optional canonical features: `o_swing_rate`, `contact_rate`, `swstr_rate`, `exit_velocity`, `launch_angle`, `barrel_rate`, `hard_hit_rate`. FanGraphs column names are also accepted. Rates must be fractions (0.10), or strings such as `10%`; numeric 10 is rejected for a probability. Exit velocity is mph and launch angle is degrees. A combined canonical CSV supplies its own contact features; it does not automatically join raw Savant caches. Savant leaderboard percentages are converted from percentage points only in the separate identifier-join path.

Keep raw originals and document export settings and dates. Duplicate traded-player stints require a single source `2 Tms`, `TOT`, or `---` aggregate row; the program will not average wRC+ across teams. All requested training and application seasons must be present.

## Validation and interpretation

Tests cover ID joins, traded-player totals, rate units, feature allowlisting, chronological label isolation, consecutive-season pairs, interval cutoffs, API normalization, and a complete offline synthetic pipeline. Synthetic player names and charts are explicitly marked and never mixed with real output.

Errors are conditional on returning hitters meeting the minimum PA. Injuries, retirement, roster selection, and survivorship bias are outside this model. A forecast below a player's 2026 result may reflect statistical regression as well as aging. Small differences between players are usually not meaningful. Correlated-feature importance is not causal. Career charts plot season indexes; averaging them is not an exact official career wRC+ calculation. The final expected estimator has more historical training data than early calibration inputs, which can introduce distribution shift.

`requirements.txt` contains the full research/test dependencies. `requirements-web.txt` contains only the hosted viewer dependencies.

## Flask findings dashboard

The read-only dashboard displays the completed results without retraining or downloading data. It includes searchable/sortable player rankings, below/above-expected and rebound/regression views, PA filters, paginated tables, filtered CSV exports, individual player career charts and empirical forecast ranges, model validation, and research notes.

After installing the updated requirements, launch it from this folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m mlb_xwrc serve --config config.json
```

Open **http://127.0.0.1:5000**. To choose another port, add `--port 5001`. An equivalent Flask entry point is `python -m flask --app app run` (or `python app.py`). The app binds to loopback and disables debug mode. Press Ctrl+C in the serving terminal to stop it.

The snapshot date and original dataset label are shown in the UI. Candidate tabs apply the configured gap threshold (10 points by default). Search is literal and accent-insensitive. The filtered export includes all matching rows, not only the current page. Lowering PA below the default ranking threshold displays a small-sample notice. Rows with more than 40% missing selected skills data stay excluded from the explorer. A missing/incomplete result set shows an actionable 503 page; unknown players and invalid filters have clear error pages.

The app reads a snapshot once per process. After regenerating the analysis, restart the dashboard to see the new results. Templates, CSS, icons, and SVG charts are served locally; no external JavaScript/font/CDN dependency is needed. Static validation figures reuse the generated PNGs. For public hosting, use the production Gunicorn entry point described above.

Frontend tests use Flask's test client: `python -m pytest tests/test_web.py -q`. The existing modeling tests are unchanged.
