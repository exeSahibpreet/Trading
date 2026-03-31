# Algo Trading App

## Overview

This project is a Flask-based stock and index backtesting application with:

- single-strategy backtesting
- strategy ranking
- an adaptive multi-strategy engine
- a walk-forward optimization module
- a browser UI for running the workflows

The app works with OHLCV data fetched through `yfinance`, generates strategy signals, routes capital based on an adaptive regime engine, and then evaluates the result with custom metrics.

## High-Level Flow

1. The browser loads the Flask app from `app.py`.
2. The user chooses a symbol, mode, and workflow from the UI.
3. `app.py` fetches or reuses cached market data through `utils.fetch_data`.
4. Depending on the route:
   - single strategy: one strategy signal series is generated and passed to `Backtester`
   - ranking: all active strategies are run one by one and scored
   - adaptive engine: `RegimeDetector` builds a regime frame, `PortfolioManager` turns it into allocations, and `MultiStrategyBacktester` simulates the portfolio
5. Metrics are computed with `utils.compute_metrics`.
6. The UI renders equity curves, rankings, and adaptive regime output.

## Active File Map

- `app.py`: Flask routes and orchestration
- `utils.py`: data fetch, ATR calculation, train/test split, metrics, ranking
- `strategies/__init__.py`: active strategy registry and labels
- `strategies/trend_following.py`: moving-average trend strategy
- `strategies/mean_reversion.py`: RSI plus Bollinger mean reversion
- `strategies/opening_range_breakout.py`: ORB-style breakout proxy
- `strategies/index_rebalancing.py`: index rebalancing proxy
- `strategies/vwap_strategy.py`: VWAP-style momentum/reversion proxy
- `adaptive_strategy.py`: adaptive regime engine, z-score logic, execution guard
- `regime_detector.py`: compatibility wrapper around the adaptive engine
- `portfolio_manager.py`: converts the regime frame into capital allocations
- `backtester.py`: single-strategy and portfolio backtesting engines
- `walk_forward.py`: rolling or anchored walk-forward optimization engine
- `templates/index.html`: main page template
- `static/js/main.js`: frontend logic
- `static/css/style.css`: styling

Note: the `strategies/` folder still contains some legacy files from older versions of the project, but the live application uses only the strategies registered in `strategies/__init__.py`.

## `app.py`

### Purpose

`app.py` is the top-level controller of the application. It defines the Flask app, caches downloaded datasets, and exposes the HTTP API used by the frontend.

### Global Variables and Dictionaries

- `app`: Flask application instance
- `cached_data`: in-memory dictionary keyed by ticker symbol to avoid repeated downloads
- `universe`: list of symbols shown in the UI
- `STRATEGY_LIST`: list of active strategy keys pulled from the strategy registry

### Functions

- `get_cached_data(ticker)`
  - downloads market data once and stores it in `cached_data`
  - tags the DataFrame with `instrument_type` metadata so the adaptive engine knows whether the asset is a stock or an index

- `get_benchmark_for_ticker(ticker)`
  - chooses a benchmark ticker for adaptive relative-strength logic
  - stocks use `^NSEI`
  - indices use the alternate index when possible

- `index()`
  - renders the main HTML page

- `run_single_strategy(strategy_name, df_train, df_test, mode)`
  - creates a signal series for a selected strategy
  - runs the single-strategy backtester on both train and test splits
  - returns metrics and equity curves

- `run_backtest()`
  - API route for single-strategy mode

- `rank_strategies_api()`
  - API route for ranking all active strategies

- `run_portfolio_api()`
  - API route for the adaptive multi-strategy engine
  - builds benchmark-aware regime frames
  - converts them into allocations
  - runs the multi-strategy backtester
  - returns combined metrics, per-strategy equity curves, and regime labels

## `utils.py`

### Purpose

This module contains shared utilities for downloading data and computing backtest metrics.

### Functions

- `fetch_data(ticker, start, end)`
  - downloads data with `yfinance`
  - normalizes columns
  - computes ATR

- `calculate_atr(df, window=14)`
  - computes Average True Range from OHLC data

- `train_test_split(df, train_ratio=0.7)`
  - splits a DataFrame into train and test partitions

- `calculate_drawdown_series(equity_curve)`
  - computes the rolling drawdown path from an equity curve

- `calculate_recovery_metrics(equity_curve, max_dd, total_profit)`
  - computes recovery factor and longest underwater period

- `compute_metrics(equity_curve, trades)`
  - computes final capital, total profit, annual return, win rate, drawdown, Sharpe, Sortino, Calmar, recovery factor, and drawdown series

- `rank_strategies(results_list)`
  - scores strategy results using Calmar, Sharpe, WFE, drawdown, recovery factor, win rate, and trade count
  - also flags weak systems such as low sample size or overfit behavior

## `strategies/__init__.py`

### Purpose

This file defines the active strategy registry used by the whole application.

### Dictionaries

- `STRATEGY_REGISTRY`
  - maps strategy keys to signal functions

- `STRATEGY_LABELS`
  - maps strategy keys to user-facing names

### Function

- `get_strategy_signal(strategy_name, df, context=None)`
  - resolves the strategy from `STRATEGY_REGISTRY`
  - runs the strategy function

## Active Strategy Files

### `strategies/trend_following.py`

- `strategy_trend_following(df, context=None)`
  - calculates 50 and 200 period moving averages
  - goes long when the short average is above the long average and rising
  - goes short when the short average is below the long average and falling

### `strategies/mean_reversion.py`

- `_rsi(close, window=14)`
  - helper to compute RSI

- `strategy_mean_reversion(df, context=None)`
  - builds Bollinger Bands and RSI
  - enters long in oversold conditions and short in overbought conditions
  - exits back toward the mean

### `strategies/opening_range_breakout.py`

- `strategy_opening_range_breakout(df, context=None)`
  - uses a daily proxy for opening range breakout
  - compares price against an intraday-style opening range estimate, rolling highs and lows, and volume confirmation

### `strategies/index_rebalancing.py`

- `strategy_index_rebalancing(df, context=None)`
  - uses volume and price jump logic near month-end as a proxy for index inclusion or deletion flows
  - holds positions for a limited number of bars

### `strategies/vwap_strategy.py`

- `strategy_vwap(df, context=None)`
  - calculates a rolling VWAP proxy
  - trades VWAP crosses with volume confirmation
  - exits when price reverts back toward VWAP

## `adaptive_strategy.py`

### Purpose

This is the main adaptive brain of the application. It builds a benchmark-aware regime frame and a faster execution gate.

### Dataclass

- `AdaptiveConfig`
  - stores all adaptive-engine parameters in one place
  - examples include z-score lookback, efficiency-ratio windows, walk-forward scoring cadence, volatility sizing bounds, slippage buffer scaling, and cooldown settings

### Utility Functions

- `sigmoid(x)`
  - smooth nonlinear transform used for weight scaling

- `_numpy_mean(arr)`
  - rolling helper using NumPy mean

- `_numpy_std(arr)`
  - rolling helper using NumPy standard deviation

- `rolling_zscore(series, window)`
  - computes a rolling z-score with `numpy.mean` and `numpy.std`

- `calculate_hurst_exponent(series)`
  - estimates the Hurst exponent from rolling log-log variance behavior

- `rolling_hurst(series, window)`
  - applies Hurst estimation over a rolling window

### `VectorizedRegimeEngine`

This class creates the slow-changing regime frame.

#### Important Class Data

- `STRATEGY_KEYS`
  - list of strategy names used throughout the adaptive engine

#### Main Methods

- `__init__(df, benchmark_df=None, config=None)`
  - stores market data, benchmark data, and adaptive settings

- `_calculate_atr()`
  - ATR helper if ATR is not already present

- `_calculate_efficiency_ratio(close)`
  - computes Kaufman-style Efficiency Ratio used to blend fast and slow indicator windows

- `_calculate_rsi(close, window=14)`
  - RSI helper

- `_adaptive_blend(fast_series, slow_series, efficiency_ratio)`
  - shortens or lengthens indicator behavior depending on trend efficiency

- `_rolling_percentile_flags(z_series)`
  - computes rolling percentile thresholds for a z-score series

- `_forward_freeze_mask(trigger_series, freeze_bars)`
  - turns one gap event into a forward freeze period

- `_prepare_features()`
  - the core feature-engineering function
  - creates ATR, returns, adaptive means, adaptive standard deviation, adaptive RSI, Hurst, VWAP distance, gap statistics, spread proxies, relative strength, volatility clusters, and rolling z-scores

- `_apply_regime_cooldown(candidate_strategy, hard_stop)`
  - forces a minimum hold time for the currently active regime

- `_compute_strategy_score_matrix(df)`
  - calculates rolling Information Ratio values for all 5 strategies
  - refreshes the score matrix every `score_rebalance_bars`

- `build_regime_frame()`
  - final output builder
  - assigns candidate and active strategies
  - computes confidence, position scale, stop-loss multiplier, slippage entry buffer, liquidity flags, gap freeze flags, and per-strategy weights

### `ExecutionGuard`

This class represents the faster execution layer.

- `__init__(regime_frame)`
  - accepts the regime frame

- `build_execution_frame()`
  - adds `can_trade` and `execution_state`
  - meant to run more frequently than the regime engine in a live system

## `regime_detector.py`

### Purpose

This file exists mainly to keep the rest of the app simple.

### Class

- `RegimeDetector`
  - compatibility wrapper
  - constructs a `VectorizedRegimeEngine`
  - then applies `ExecutionGuard`

### Methods

- `__init__(df, benchmark_df=None)`
- `detect_regimes()`

## `portfolio_manager.py`

### Purpose

This module converts the regime frame into actual portfolio allocations.

### Class

- `PortfolioManager`

### Class Data

- `STRATEGY_KEYS`
  - active strategy list

### Method

- `get_allocation(date)`
  - reads the row for a given timestamp from the regime frame
  - blocks trading if `can_trade` is false
  - reads the per-strategy weights
  - scales the final allocation by `position_scale`

## `backtester.py`

### `Backtester`

Single-strategy simulator used in standalone strategy mode.

#### Methods

- `__init__(...)`
  - stores capital, risk, brokerage, slippage, and leverage settings

- `run(df, signals)`
  - simulates one strategy over time
  - opens and closes one position at a time
  - returns an equity curve and trade list

### `MultiStrategyBacktester`

Portfolio simulator used by the adaptive engine.

#### Methods

- `__init__(mode="equity", initial_capital=100000)`
  - sets portfolio risk parameters

- `run_portfolio(df, strategies_signals, portfolio_allocations, regime_data=None)`
  - evaluates all strategy signals together
  - manages open trades, exits, stop losses, dynamic stop multipliers, and entry buffers
  - returns combined equity, per-strategy equity curves, and trade dictionaries

## `walk_forward.py`

### Purpose

This module adds Rolling Walk-Forward Optimization.

### Dataclass

- `WalkForwardConfig`
  - stores WFO settings such as train size, test size, purge window, and anchored versus rolling mode

### Functions

- `calculate_profit_factor(trades)`
  - computes gross profit divided by gross loss

- `calculate_sortino_from_equity(equity_curve)`
  - computes Sortino Ratio directly from an equity curve

- `build_fold_schedule(index, train_size, test_size, purge_size=0, anchored=False)`
  - creates fold definitions for rolling or anchored walk-forward runs

- `_generate_parameter_grid(param_space)`
  - expands parameter combinations for optimization

- `_build_signals(df)`
  - creates all registered strategy signals for a dataset

- `_run_adaptive_backtest(df, benchmark_df, adaptive_params, mode="equity", initial_capital=100000)`
  - helper that runs a full adaptive backtest with one parameter set

- `_evaluate_param_candidate(args)`
  - multiprocessing-safe wrapper for one candidate configuration

- `optimize_fold_parameters(...)`
  - searches a parameter grid for the best in-sample Sortino Ratio

- `run_walk_forward(...)`
  - main WFO entry point
  - performs fold creation, in-sample optimization, out-of-sample testing, WFE calculation, and overfit flagging
  - returns a summary table and detailed fold outputs

## Frontend Flow

### `templates/index.html`

Defines the dashboard structure:

- sidebar controls
- tab navigation
- metric cards
- ranking table
- adaptive engine charts

### `static/js/main.js`

Controls frontend behavior:

- tab switching
- form submission
- API calls to Flask endpoints
- rendering Plotly charts
- populating result tables

### `static/css/style.css`

Defines the application styling and responsive layout.

## Typical Adaptive Run

1. User selects a stock or index in the browser.
2. `run_portfolio_api()` fetches the asset and benchmark.
3. `RegimeDetector.detect_regimes()` creates the adaptive regime frame.
4. `PortfolioManager.get_allocation()` converts that frame into position weights.
5. Strategy signals are generated for all active strategies.
6. `MultiStrategyBacktester.run_portfolio()` simulates entries, exits, stop losses, and equity.
7. `compute_metrics()` summarizes the outcome.
8. The browser renders the charts and ranking tables.

## Typical Walk-Forward Run

1. `run_walk_forward()` splits the full dataset into train and test folds.
2. Each fold searches a parameter grid on the training segment.
3. The best parameters are selected by Sortino Ratio.
4. Those parameters are applied to the test segment.
5. Fold metrics are recorded and combined into a Walk-Forward Efficiency score.

## Notes

- The adaptive logic is designed to be benchmark-aware for stocks and benchmark-light for indices.
- The project currently uses daily OHLCV data, so some intraday concepts such as VWAP and ORB are implemented as daily proxies.
- Some older files remain in the repository but are not part of the current live flow.
