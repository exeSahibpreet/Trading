from dataclasses import asdict, dataclass
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from adaptive_strategy import AdaptiveConfig, ExecutionGuard, VectorizedRegimeEngine
from backtester import MultiStrategyBacktester
from portfolio_manager import PortfolioManager
from strategies import STRATEGY_REGISTRY
from utils import compute_metrics


@dataclass
class WalkForwardConfig:
    train_size: int
    test_size: int
    purge_size: int = 0
    anchored: bool = False
    mode: str = "equity"
    initial_capital: float = 100000
    use_multiprocessing: bool = True
    max_workers: int | None = None


def calculate_profit_factor(trades):
    wins = [trade.get("pnl", 0.0) for trade in trades if trade.get("pnl", 0.0) > 0]
    losses = [abs(trade.get("pnl", 0.0)) for trade in trades if trade.get("pnl", 0.0) < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_sortino_from_equity(equity_curve):
    if len(equity_curve) <= 1:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    downside = returns[returns < 0]
    downside_std = downside.std()
    if pd.isna(downside_std) or downside_std == 0:
        return 0.0
    return float(np.sqrt(252) * (returns.mean() / downside_std))


def build_fold_schedule(index, train_size, test_size, purge_size=0, anchored=False):
    folds = []
    total = len(index)
    train_start = 0
    train_end = train_size

    while train_end + purge_size + test_size <= total:
        test_start = train_end + purge_size
        test_end = test_start + test_size

        folds.append(
            {
                "train_start": 0 if anchored else train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        if anchored:
            train_end += test_size
        else:
            train_start += test_size
            train_end = train_start + train_size

    return folds


def _generate_parameter_grid(param_space):
    keys = list(param_space.keys())
    values = [param_space[key] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _build_signals(df):
    return {key: STRATEGY_REGISTRY[key](df).reindex(df.index).fillna(0) for key in STRATEGY_REGISTRY}


def _run_adaptive_backtest(df, benchmark_df, adaptive_params, mode="equity", initial_capital=100000):
    config = AdaptiveConfig(**adaptive_params)
    regime_frame = ExecutionGuard(VectorizedRegimeEngine(df, benchmark_df=benchmark_df, config=config).build_regime_frame()).build_execution_frame()
    portfolio_manager = PortfolioManager(regime_frame)
    allocations = pd.DataFrame([portfolio_manager.get_allocation(ts) for ts in df.index], index=df.index)
    signals = _build_signals(df)
    backtester = MultiStrategyBacktester(mode=mode, initial_capital=initial_capital)
    combined_equity, _, trades = backtester.run_portfolio(df, signals, allocations, regime_data=regime_frame)

    flat_trades = []
    for trade_list in trades.values():
        flat_trades.extend(trade_list)

    metrics = compute_metrics(combined_equity, flat_trades)
    metrics["sortino_objective"] = calculate_sortino_from_equity(combined_equity)
    metrics["profit_factor"] = calculate_profit_factor(flat_trades)

    dominance = (
        regime_frame["active_strategy"]
        .value_counts()
        .idxmax()
        if "active_strategy" in regime_frame.columns and not regime_frame["active_strategy"].dropna().empty
        else "none"
    )

    return {
        "equity": combined_equity,
        "trades": flat_trades,
        "metrics": metrics,
        "regime_frame": regime_frame,
        "strategy_dominance": dominance,
    }


def _evaluate_param_candidate(args):
    train_df, benchmark_train, adaptive_params, mode, initial_capital = args
    result = _run_adaptive_backtest(
        train_df,
        benchmark_train,
        adaptive_params=adaptive_params,
        mode=mode,
        initial_capital=initial_capital,
    )
    return {
        "params": adaptive_params,
        "objective": result["metrics"]["sortino_objective"],
        "metrics": result["metrics"],
        "dominance": result["strategy_dominance"],
    }


def optimize_fold_parameters(
    train_df,
    benchmark_train,
    param_space,
    mode="equity",
    initial_capital=100000,
    use_multiprocessing=True,
    max_workers=None,
):
    candidates = list(_generate_parameter_grid(param_space))
    tasks = [
        (train_df, benchmark_train, candidate, mode, initial_capital)
        for candidate in candidates
    ]

    if use_multiprocessing and len(tasks) > 1:
        workers = max_workers or max(1, min(cpu_count() - 1, len(tasks)))
        with Pool(processes=workers) as pool:
            results = pool.map(_evaluate_param_candidate, tasks)
    else:
        results = [_evaluate_param_candidate(task) for task in tasks]

    best = max(results, key=lambda item: item["objective"])
    return best, results


def run_walk_forward(
    data,
    train_size,
    test_size,
    benchmark_data=None,
    purge_size=0,
    anchored=False,
    mode="equity",
    initial_capital=100000,
    param_space=None,
    use_multiprocessing=True,
    max_workers=None,
):
    if param_space is None:
        param_space = {
            "z_trigger_level": [1.0, 1.25, 1.5],
            "score_boost_multiplier": [0.25, 0.5, 0.75],
            "slippage_atr_multiplier": [0.05, 0.1, 0.15],
            "kelly_fraction": [0.1, 0.25],
            "regime_cooldown_bars": [3, 5],
        }

    folds = build_fold_schedule(
        data.index,
        train_size=train_size,
        test_size=test_size,
        purge_size=purge_size,
        anchored=anchored,
    )

    summaries = []
    detailed_results = []

    for fold_number, fold in enumerate(folds, start=1):
        train_df = data.iloc[fold["train_start"] : fold["train_end"]].copy()
        test_df = data.iloc[fold["test_start"] : fold["test_end"]].copy()

        benchmark_train = None
        benchmark_test = None
        if benchmark_data is not None and len(benchmark_data) > 0:
            benchmark_train = benchmark_data.reindex(train_df.index).ffill()
            benchmark_test = benchmark_data.reindex(test_df.index).ffill()

        best_params, search_results = optimize_fold_parameters(
            train_df=train_df,
            benchmark_train=benchmark_train,
            param_space=param_space,
            mode=mode,
            initial_capital=initial_capital,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
        )

        train_result = _run_adaptive_backtest(
            train_df,
            benchmark_train,
            adaptive_params=best_params["params"],
            mode=mode,
            initial_capital=initial_capital,
        )
        test_result = _run_adaptive_backtest(
            test_df,
            benchmark_test,
            adaptive_params=best_params["params"],
            mode=mode,
            initial_capital=initial_capital,
        )

        train_sortino = train_result["metrics"]["sortino_objective"]
        test_sortino = test_result["metrics"]["sortino_objective"]
        wfe = max(0.0, test_sortino / train_sortino) if train_sortino > 0 else 0.0
        overfit_flag = wfe < 0.5

        fold_summary = {
            "fold": fold_number,
            "train_start": train_df.index[0] if len(train_df) > 0 else None,
            "train_end": train_df.index[-1] if len(train_df) > 0 else None,
            "test_start": test_df.index[0] if len(test_df) > 0 else None,
            "test_end": test_df.index[-1] if len(test_df) > 0 else None,
            "train_label": (
                f"{train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}"
                if len(train_df) > 0 else ""
            ),
            "test_label": (
                f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}"
                if len(test_df) > 0 else ""
            ),
            "train_sortino": train_sortino,
            "test_sortino": test_sortino,
            "test_profit": test_result["metrics"]["total_profit"],
            "test_final_capital": test_result["metrics"]["final_capital"],
            "test_max_drawdown": test_result["metrics"]["max_drawdown"],
            "test_profit_factor": test_result["metrics"]["profit_factor"],
            "strategy_dominance": test_result["strategy_dominance"],
            "wfe": wfe,
            "overfitted": overfit_flag,
            "best_params": best_params["params"],
        }
        summaries.append(fold_summary)
        detailed_results.append(
            {
                "fold": fold_number,
                "fold_definition": fold,
                "best_params": best_params,
                "search_results": search_results,
                "train_result": train_result,
                "test_result": test_result,
            }
        )

    summary_df = pd.DataFrame(summaries)
    final_wfe = float(summary_df["wfe"].mean()) if not summary_df.empty else 0.0

    return {
        "summary": summary_df,
        "folds": detailed_results,
        "final_wfe": final_wfe,
        "overfit_flag": final_wfe < 0.5,
    }
