import logging
import traceback
from datetime import date

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from backtester import Backtester, MultiStrategyBacktester
from portfolio_manager import PortfolioManager
from regime_detector import RegimeDetector
from strategies import STRATEGY_LABELS, STRATEGY_REGISTRY, get_strategy_signal
from utils import compute_metrics, fetch_data, rank_strategies, train_test_split
from walk_forward import run_walk_forward

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

cached_data = {}

universe = [
    "^NSEI",
    "^BSESN",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "TCS.NS",
    "LT.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ASIANPAINT.NS",
]

STRATEGY_LIST = list(STRATEGY_REGISTRY.keys())


def log_update(message):
    app.logger.info(message)
    print(message, flush=True)


def log_exception(context, exc):
    message = f"[ERROR] {context}: {exc}"
    app.logger.exception(message)
    print(message, flush=True)
    print(traceback.format_exc(), flush=True)


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def get_cached_data(ticker):
    if ticker not in cached_data:
        log_update(f"[DATA] Fetching fresh data for {ticker}")
        cached_data[ticker] = fetch_data(ticker, start="2014-01-01", end=date.today().isoformat())
        if cached_data[ticker].empty:
            raise ValueError(
                f"Could not load market data for {ticker}. "
                "Please check internet access in Colab and try again."
            )
        cached_data[ticker].attrs["instrument_type"] = "index" if str(ticker).startswith("^") else "stock"
    else:
        log_update(f"[DATA] Using cached data for {ticker}")
    return cached_data[ticker]


def get_benchmark_for_ticker(ticker):
    if str(ticker).startswith("^"):
        if ticker == "^NSEI":
            return "^BSESN"
        return "^NSEI"
    return "^NSEI"


@app.route("/")
def index():
    return render_template("index.html", universe=universe, strategy_labels=STRATEGY_LABELS)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})


def run_single_strategy(strategy_name, df_train, df_test, mode):
    sig_train = get_strategy_signal(strategy_name, df_train)
    sig_test = get_strategy_signal(strategy_name, df_test)

    backtester = Backtester(mode=mode)
    eq_train, trades_train = backtester.run(df_train, sig_train)
    eq_test, trades_test = backtester.run(df_test, sig_test)

    return {
        "train": {
            "metrics": compute_metrics(eq_train, trades_train),
            "dates": eq_train.index.strftime("%Y-%m-%d").tolist() if len(eq_train) > 0 else [],
            "equity": eq_train.tolist() if len(eq_train) > 0 else [],
        },
        "test": {
            "metrics": compute_metrics(eq_test, trades_test),
            "dates": eq_test.index.strftime("%Y-%m-%d").tolist() if len(eq_test) > 0 else [],
            "equity": eq_test.tolist() if len(eq_test) > 0 else [],
        },
    }


@app.route("/api/run_backtest", methods=["POST"])
def run_backtest():
    req = request.json or {}
    strategy_name = req.get("strategy", "trend_following")
    ticker = req.get("index", "^NSEI")
    mode = req.get("mode", "equity")

    try:
        log_update(f"[API] /api/run_backtest strategy={strategy_name} ticker={ticker} mode={mode}")
        df = get_cached_data(ticker)
        df_train, df_test = train_test_split(df)
        res = run_single_strategy(strategy_name, df_train, df_test, mode)
        res["strategy_label"] = STRATEGY_LABELS.get(strategy_name, strategy_name)
        log_update(f"[API] /api/run_backtest completed for {strategy_name} on {ticker}")
        return jsonify(res)
    except Exception as exc:
        log_exception("/api/run_backtest failed", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/rank_strategies", methods=["POST"])
def rank_strategies_api():
    req = request.json or {}
    ticker = req.get("index", "^NSEI")
    mode = req.get("mode", "equity")

    try:
        log_update(f"[API] /api/rank_strategies ticker={ticker} mode={mode}")
        df = get_cached_data(ticker)
        df_train, df_test = train_test_split(df)

        results = []
        for strategy_name in STRATEGY_LIST:
            res = run_single_strategy(strategy_name, df_train, df_test, mode)
            results.append(
                {
                    "strategy_name": STRATEGY_LABELS[strategy_name],
                    "strategy_key": strategy_name,
                    "train_metrics": res["train"]["metrics"],
                    "test_metrics": res["test"]["metrics"],
                    "equity": res["test"]["equity"],
                    "drawdown": res["test"]["metrics"]["drawdown_series"],
                    "dates": res["test"]["dates"],
                }
            )

        ranked = rank_strategies(results)
        log_update(f"[API] /api/rank_strategies completed for {ticker} with {len(ranked)} strategies")
        return jsonify({"ranked": ranked})
    except Exception as exc:
        log_exception("/api/rank_strategies failed", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/run_portfolio", methods=["POST"])
def run_portfolio_api():
    req = request.json or {}
    ticker = req.get("index", "^NSEI")
    mode = req.get("mode", "equity")
    toggles = req.get("toggles", {})

    try:
        enabled_count = sum(1 for strategy_key in STRATEGY_LIST if toggles.get(strategy_key, True) is not False)
        log_update(f"[API] /api/run_portfolio ticker={ticker} mode={mode} enabled_strategies={enabled_count}")
        df = get_cached_data(ticker)
        df_train, df_test = train_test_split(df)

        benchmark_ticker = get_benchmark_for_ticker(ticker)
        benchmark_df = get_cached_data(benchmark_ticker)
        benchmark_train, benchmark_test = train_test_split(benchmark_df)

        train_regimes_df = RegimeDetector(df_train, benchmark_df=benchmark_train).detect_regimes()
        test_regimes_df = RegimeDetector(df_test, benchmark_df=benchmark_test).detect_regimes()

        pm_train = PortfolioManager(train_regimes_df)
        pm_test = PortfolioManager(test_regimes_df)
        allocs_train = pd.DataFrame([pm_train.get_allocation(d) for d in df_train.index], index=df_train.index)
        allocs_test = pd.DataFrame([pm_test.get_allocation(d) for d in df_test.index], index=df_test.index)

        for strategy_key in STRATEGY_LIST:
            if toggles.get(strategy_key, True) is False:
                allocs_train[strategy_key] = 0.0
                allocs_test[strategy_key] = 0.0

        sigs_train = {strategy_key: get_strategy_signal(strategy_key, df_train) for strategy_key in STRATEGY_LIST}
        sigs_test = {strategy_key: get_strategy_signal(strategy_key, df_test) for strategy_key in STRATEGY_LIST}

        backtester = MultiStrategyBacktester(mode=mode)
        combined_train, strat_train, trades_train = backtester.run_portfolio(df_train, sigs_train, allocs_train, regime_data=train_regimes_df)
        combined_test, strat_test, trades_test = backtester.run_portfolio(df_test, sigs_test, allocs_test, regime_data=test_regimes_df)

        def aggregate_trades(trades_dict):
            out = []
            for trade_list in trades_dict.values():
                out.extend(trade_list)
            return out

        regime_pnl = {}
        combined_returns = combined_test.pct_change().fillna(0)
        regimes_aligned = test_regimes_df["regime"].reindex(combined_test.index).fillna("BALANCED")
        recommended_aligned = test_regimes_df["active_strategy"].reindex(combined_test.index).fillna("mean_reversion")

        for regime_name in regimes_aligned.unique():
            mask = regimes_aligned == regime_name
            pnl_frac = (1 + combined_returns[mask]).prod() - 1 if mask.any() else 0
            regime_pnl[regime_name] = np.round(pnl_frac * 100, 2)

        result = {
            "train": {
                "combined_metrics": compute_metrics(pd.Series(combined_train), aggregate_trades(trades_train)),
            },
            "test": {
                "combined_metrics": compute_metrics(pd.Series(combined_test), aggregate_trades(trades_test)),
                "combined_equity": combined_test.tolist() if len(combined_test) > 0 else [],
                "dates": combined_test.index.strftime("%Y-%m-%d").tolist() if len(combined_test) > 0 else [],
                "regimes": regimes_aligned.tolist() if len(combined_test) > 0 else [],
                "recommended_strategies": recommended_aligned.tolist() if len(combined_test) > 0 else [],
                "regime_pnl": regime_pnl,
                "strat_equities": {key: series.tolist() for key, series in strat_test.items()},
                "strategy_labels": STRATEGY_LABELS,
                "hurst": test_regimes_df["hurst"].reindex(combined_test.index).fillna(0).tolist() if len(combined_test) > 0 else [],
            },
        }
        log_update(f"[API] /api/run_portfolio completed for {ticker}")
        return jsonify(result)
    except Exception as exc:
        log_exception("/api/run_portfolio failed", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/run_walk_forward", methods=["POST"])
def run_walk_forward_api():
    req = request.json or {}
    ticker = req.get("index", "^NSEI")
    mode = req.get("mode", "equity")
    anchored = bool(req.get("anchored", False))
    train_size = int(req.get("train_size", 504))
    test_size = int(req.get("test_size", 252))
    purge_size = int(req.get("purge_size", 5))

    try:
        log_update(
            f"[API] /api/run_walk_forward ticker={ticker} mode={mode} anchored={anchored} "
            f"train_size={train_size} test_size={test_size} purge_size={purge_size}"
        )
        df = get_cached_data(ticker)
        benchmark_ticker = get_benchmark_for_ticker(ticker)
        benchmark_df = get_cached_data(benchmark_ticker)

        result = run_walk_forward(
            data=df,
            benchmark_data=benchmark_df,
            train_size=train_size,
            test_size=test_size,
            purge_size=purge_size,
            anchored=anchored,
            mode=mode,
            use_multiprocessing=False,
        )

        summary_df = result["summary"].copy()
        if not summary_df.empty:
            for col in ["train_start", "train_end", "test_start", "test_end"]:
                if col in summary_df.columns:
                    summary_df[col] = summary_df[col].astype(str)

        response_payload = to_json_safe(
            {
                "summary": summary_df.to_dict(orient="records"),
                "final_wfe": result["final_wfe"],
                "overfit_flag": result["overfit_flag"],
            }
        )
        response = jsonify(response_payload)
        log_update(f"[API] /api/run_walk_forward completed for {ticker} with {len(summary_df)} folds")
        return response
    except Exception as exc:
        log_exception("/api/run_walk_forward failed", exc)
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    if request.path.startswith("/api/"):
        log_exception(f"Unhandled exception on {request.path}", exc)
        return jsonify({"error": str(exc)}), 500
    raise exc


if __name__ == "__main__":
    log_update("[APP] Starting Flask app on 0.0.0.0:5000")
    app.run(debug=False, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)
