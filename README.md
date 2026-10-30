# xwRC+ Lab

[Explore the live dashboard](https://xwrcplus.onrender.com)

MLB hitting analysis that looks beyond observed results to identify potential breakout and regression candidates.

## What's inside

- **2026 actual vs. expected wRC+** to highlight under- and over-performers.
- **2027 hitting projections** with aging adjustments and forecast ranges.
- **Interactive player rankings and career charts**, plus model validation and CSV exports.

## How it works

Uses 2016–2025 MLB data from pybaseball, FanGraphs, and Baseball Savant to train an XGBoost expected-performance model. A separate forecasting step combines expected performance, recent results, and an aging adjustment to project 2027 hitting.

Built with Python and Flask. Projections are estimates, and historical backtests show their uncertainty.
