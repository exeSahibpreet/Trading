from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies import STRATEGY_REGISTRY


@dataclass
class AdaptiveConfig:
    base_lookback: int = 100
    fast_window_floor: int = 20
    slow_window_ceiling: int = 160
    efficiency_window: int = 20
    atr_window: int = 14
    rs_window: int = 20
    score_rebalance_bars: int = 50
    score_lookback_bars: int = 50
    regime_cooldown_bars: int = 5
    gap_freeze_bars: int = 6
    z_trigger_level: float = 1.5
    gap_z_trigger_level: float = 2.0
    gap_threshold_scale: float = 1.75
    spread_z_trigger_level: float = 1.5
    volatility_target: float = 1.0
    kelly_fraction: float = 0.25
    minimum_position_scale: float = 0.1
    maximum_position_scale: float = 1.0
    slippage_atr_multiplier: float = 0.1
    stop_base_multiplier: float = 1.0
    stop_trend_multiplier: float = 1.8
    stop_mean_reversion_multiplier: float = 0.8
    stop_breakout_multiplier: float = 1.2
    stop_flow_multiplier: float = 1.0
    score_boost_multiplier: float = 0.5
    percentile_lower: float = 0.2
    percentile_upper: float = 0.8
    trend_bias_scale: float = 1.0
    mean_bias_scale: float = 1.0
    breakout_bias_scale: float = 1.0
    flow_bias_scale: float = 1.0
    event_bias_scale: float = 1.0
    commission_pct: float = 0.0003
    slippage_pct: float = 0.0010


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _numpy_mean(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr))


def _numpy_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.std(arr))


def rolling_zscore(series, window):
    rolling_mean = series.rolling(window=window, min_periods=max(10, window // 5)).apply(_numpy_mean, raw=True)
    rolling_std = series.rolling(window=window, min_periods=max(10, window // 5)).apply(_numpy_std, raw=True)
    return ((series - rolling_mean) / rolling_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def calculate_hurst_exponent(series):
    """
    The Hurst exponent is estimated by fitting a straight line in log-log space:
    log(std(P[t + lag] - P[t])) = H * log(lag) + c

    H > 0.5 implies persistence, H < 0.5 implies mean reversion.
    """
    clean = pd.Series(series).dropna()
    if len(clean) < 50:
        return np.nan

    lags = np.arange(2, 20)
    tau = []
    valid_lags = []

    for lag in lags:
        diff = clean.diff(lag).dropna()
        if diff.empty:
            continue
        std = _numpy_std(diff.to_numpy())
        if std > 0:
            tau.append(std)
            valid_lags.append(lag)

    if len(valid_lags) < 5:
        return np.nan

    slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    return float(slope)


def rolling_hurst(series, window):
    return series.rolling(window=window, min_periods=max(50, window // 2)).apply(calculate_hurst_exponent, raw=False)


class VectorizedRegimeEngine:
    STRATEGY_KEYS = [
        "trend_following",
        "mean_reversion",
        "opening_range_breakout",
        "index_rebalancing",
        "vwap",
    ]

    def __init__(self, df, benchmark_df=None, config=None):
        self.df = df.copy()
        self.benchmark_df = benchmark_df.copy() if benchmark_df is not None and len(benchmark_df) > 0 else None
        self.config = config or AdaptiveConfig()
        self.is_index = bool(str(self.df.attrs.get("instrument_type", "")).lower() == "index")

    def _calculate_atr(self):
        high_low = self.df["High"] - self.df["Low"]
        high_close = (self.df["High"] - self.df["Close"].shift(1)).abs()
        low_close = (self.df["Low"] - self.df["Close"].shift(1)).abs()
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=self.config.atr_window).mean()

    def _calculate_efficiency_ratio(self, close):
        direction = close.diff(self.config.efficiency_window).abs()
        volatility = close.diff().abs().rolling(window=self.config.efficiency_window).sum()
        return (direction / volatility.replace(0, np.nan)).clip(lower=0, upper=1).fillna(0)

    def _calculate_rsi(self, close, window=14):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _adaptive_blend(self, fast_series, slow_series, efficiency_ratio):
        return (fast_series * efficiency_ratio) + (slow_series * (1.0 - efficiency_ratio))

    def _rolling_percentile_flags(self, z_series):
        upper = z_series.rolling(window=self.config.base_lookback, min_periods=max(20, self.config.base_lookback // 5)).quantile(self.config.percentile_upper)
        lower = z_series.rolling(window=self.config.base_lookback, min_periods=max(20, self.config.base_lookback // 5)).quantile(self.config.percentile_lower)
        return lower, upper

    def _forward_freeze_mask(self, trigger_series, freeze_bars):
        trigger = trigger_series.fillna(False).astype(int).to_numpy()
        expanded = np.convolve(trigger, np.ones(freeze_bars, dtype=int), mode="full")[: len(trigger)]
        return pd.Series(expanded > 0, index=trigger_series.index)

    def _compute_strategy_information_ratio(self, strategy_signals, asset_returns):
        strategy_returns = strategy_signals.shift(1).fillna(0).mul(asset_returns, axis=0)
        rolling_mean = strategy_returns.rolling(window=self.config.score_lookback_bars, min_periods=max(10, self.config.score_lookback_bars // 5)).mean()
        rolling_std = strategy_returns.rolling(window=self.config.score_lookback_bars, min_periods=max(10, self.config.score_lookback_bars // 5)).std()
        return (rolling_mean / rolling_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)

    def _compute_strategy_score_matrix(self, df):
        asset_returns = df["Close"].pct_change().fillna(0)
        score_df = pd.DataFrame(index=df.index)

        for key in self.STRATEGY_KEYS:
            signals = STRATEGY_REGISTRY[key](self.df)
            score_df[key] = self._compute_strategy_information_ratio(signals, asset_returns)

        rebalance_mask = np.arange(len(score_df)) % self.config.score_rebalance_bars == 0
        rebalance_dates = score_df.index[rebalance_mask]
        score_rebalanced = score_df.loc[rebalance_dates].reindex(score_df.index).ffill().fillna(0)
        return score_rebalanced

    def _prepare_features(self):
        df = self.df.copy()
        cfg = self.config

        df["ATR"] = df["ATR"] if "ATR" in df.columns else self._calculate_atr()
        df["ATR_PCT"] = (df["ATR"] / df["Close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        df["RETURNS"] = df["Close"].pct_change().fillna(0)
        df["ER"] = self._calculate_efficiency_ratio(df["Close"])
        df["ADAPTIVE_WINDOW"] = (cfg.slow_window_ceiling - (cfg.slow_window_ceiling - cfg.fast_window_floor) * df["ER"]).round()

        df["VOL_MA_FAST"] = df["Volume"].rolling(window=cfg.fast_window_floor).mean()
        df["VOL_MA_SLOW"] = df["Volume"].rolling(window=cfg.slow_window_ceiling).mean()
        df["VOL_MA"] = self._adaptive_blend(df["VOL_MA_FAST"], df["VOL_MA_SLOW"], df["ER"])

        df["SMA_20_FAST"] = df["Close"].rolling(window=cfg.fast_window_floor).mean()
        df["SMA_20_SLOW"] = df["Close"].rolling(window=cfg.slow_window_ceiling).mean()
        df["ADAPTIVE_MEAN"] = self._adaptive_blend(df["SMA_20_FAST"], df["SMA_20_SLOW"], df["ER"])
        df["ADAPTIVE_STD"] = self._adaptive_blend(
            df["Close"].rolling(window=cfg.fast_window_floor).std(),
            df["Close"].rolling(window=cfg.slow_window_ceiling).std(),
            df["ER"],
        ).replace(0, np.nan)
        df["RSI_FAST"] = self._calculate_rsi(df["Close"], window=max(2, cfg.fast_window_floor // 2))
        df["RSI_SLOW"] = self._calculate_rsi(df["Close"], window=max(2, cfg.slow_window_ceiling // 8))
        df["RSI_ADAPTIVE"] = self._adaptive_blend(df["RSI_FAST"], df["RSI_SLOW"], df["ER"])
        df["HURST"] = rolling_hurst(df["Close"], cfg.base_lookback)

        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP_FAST"] = (typical_price * df["Volume"]).rolling(cfg.fast_window_floor).sum() / df["Volume"].rolling(cfg.fast_window_floor).sum().replace(0, np.nan)
        df["VWAP_SLOW"] = (typical_price * df["Volume"]).rolling(cfg.slow_window_ceiling).sum() / df["Volume"].rolling(cfg.slow_window_ceiling).sum().replace(0, np.nan)
        df["VWAP_ADAPTIVE"] = self._adaptive_blend(df["VWAP_FAST"], df["VWAP_SLOW"], df["ER"])
        df["VWAP_DISTANCE"] = ((df["Close"] - df["VWAP_ADAPTIVE"]) / df["VWAP_ADAPTIVE"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)

        df["TREND_SPREAD_FAST"] = (
            (df["Close"].rolling(cfg.fast_window_floor).mean() - df["Close"].rolling(cfg.fast_window_floor * 2).mean())
            / df["Close"].rolling(cfg.fast_window_floor * 2).mean().replace(0, np.nan)
        )
        df["TREND_SPREAD_SLOW"] = (
            (df["Close"].rolling(cfg.slow_window_ceiling // 2).mean() - df["Close"].rolling(cfg.slow_window_ceiling).mean())
            / df["Close"].rolling(cfg.slow_window_ceiling).mean().replace(0, np.nan)
        )
        df["TREND_SPREAD"] = self._adaptive_blend(df["TREND_SPREAD_FAST"], df["TREND_SPREAD_SLOW"], df["ER"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        df["PRICE_STRETCH"] = ((df["Close"] - df["ADAPTIVE_MEAN"]) / df["ADAPTIVE_MEAN"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        df["ORB_PRESSURE"] = ((df["Close"] - df["Open"]) / df["ATR"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        prev_close = df["Close"].shift(1)
        df["GAP_PCT"] = ((df["Open"] - prev_close) / prev_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        df["SPREAD_PROXY"] = ((df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        df["VW_SPREAD_PROXY"] = (df["SPREAD_PROXY"] / (df["Volume"] / df["VOL_MA"].replace(0, np.nan)).clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(df["SPREAD_PROXY"])

        if self.benchmark_df is not None:
            bench_close = self.benchmark_df["Close"].reindex(df.index).ffill()
            bench_returns = bench_close.pct_change().fillna(0)
            df["RS_20"] = df["Close"].pct_change(cfg.rs_window) - bench_close.pct_change(cfg.rs_window)
            df["BENCH_VOL"] = bench_returns.rolling(window=cfg.base_lookback, min_periods=max(20, cfg.base_lookback // 5)).std().fillna(0)
        else:
            df["RS_20"] = 0.0
            df["BENCH_VOL"] = df["RETURNS"].rolling(window=cfg.base_lookback, min_periods=max(20, cfg.base_lookback // 5)).std().fillna(0)

        df["ASSET_VOL"] = df["RETURNS"].rolling(window=cfg.base_lookback, min_periods=max(20, cfg.base_lookback // 5)).std().fillna(0)

        z_cols = {
            "hurst_z": "HURST",
            "trend_z": "TREND_SPREAD",
            "mean_z": "PRICE_STRETCH",
            "flow_z": "VWAP_DISTANCE",
            "breakout_z": "ORB_PRESSURE",
            "gap_z": "GAP_PCT",
            "spread_z": "VW_SPREAD_PROXY",
            "rs_z": "RS_20",
            "atr_z": "ATR_PCT",
        }
        for out_col, src_col in z_cols.items():
            df[out_col] = rolling_zscore(df[src_col].fillna(0), cfg.base_lookback).fillna(0)

        df["TREND_LOWER"], df["TREND_UPPER"] = self._rolling_percentile_flags(df["trend_z"])
        df["MEAN_LOWER"], df["MEAN_UPPER"] = self._rolling_percentile_flags(df["mean_z"])
        df["FLOW_LOWER"], df["FLOW_UPPER"] = self._rolling_percentile_flags(df["flow_z"])
        df["BREAKOUT_LOWER"], df["BREAKOUT_UPPER"] = self._rolling_percentile_flags(df["breakout_z"])
        df["HURST_LOWER"], df["HURST_UPPER"] = self._rolling_percentile_flags(df["hurst_z"])

        df["TREND_ALLOWED"] = True if self.is_index else (df["rs_z"] > cfg.z_trigger_level)
        df["GAP_FREEZE"] = self._forward_freeze_mask(df["gap_z"].abs() > cfg.gap_z_trigger_level, cfg.gap_freeze_bars)
        df["LIQUIDITY_OK"] = df["spread_z"] < cfg.spread_z_trigger_level
        df["NOISY_CHOP"] = (df["hurst_z"].abs() < cfg.z_trigger_level) & (df["trend_z"].abs() < cfg.z_trigger_level)

        vol_cluster_score = rolling_zscore(df["ASSET_VOL"].fillna(0), cfg.base_lookback).fillna(0)
        df["VOL_CLUSTER"] = np.select(
            [vol_cluster_score < -cfg.z_trigger_level, vol_cluster_score > cfg.z_trigger_level],
            ["LOW_VOL", "HIGH_VOL"],
            default="MID_VOL",
        )

        return df

    def _apply_regime_cooldown(self, candidate_strategy, hard_stop):
        active = []
        current = "mean_reversion"
        hold_count = self.config.regime_cooldown_bars

        for candidate, stop_now in zip(candidate_strategy, hard_stop):
            if stop_now:
                current = "no_trade"
                hold_count = 0
            elif current == "no_trade":
                current = candidate if candidate != "no_trade" else "mean_reversion"
                hold_count = 1
            elif candidate == current:
                hold_count += 1
            elif hold_count >= self.config.regime_cooldown_bars:
                current = candidate
                hold_count = 1
            else:
                hold_count += 1
            active.append(current)

        return pd.Series(active, index=candidate_strategy.index)

    def _compute_strategy_score_matrix(self, df):
        base_scores = pd.DataFrame(index=df.index)
        asset_returns = df["RETURNS"]

        for key in self.STRATEGY_KEYS:
            signals = STRATEGY_REGISTRY[key](self.df).reindex(df.index).fillna(0)
            strat_returns = signals.shift(1).fillna(0) * asset_returns
            mean_ret = strat_returns.rolling(window=self.config.score_lookback_bars, min_periods=max(10, self.config.score_lookback_bars // 5)).mean()
            std_ret = strat_returns.rolling(window=self.config.score_lookback_bars, min_periods=max(10, self.config.score_lookback_bars // 5)).std()
            base_scores[key] = (mean_ret / std_ret.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)

        rebalance_points = np.arange(len(base_scores)) % self.config.score_rebalance_bars == 0
        score_matrix = base_scores.loc[base_scores.index[rebalance_points]].reindex(base_scores.index).ffill().fillna(0)
        top_strategy = score_matrix.idxmax(axis=1)
        top_score = score_matrix.max(axis=1).clip(lower=0)

        boost = pd.DataFrame(0.0, index=df.index, columns=self.STRATEGY_KEYS)
        for key in self.STRATEGY_KEYS:
            boost[key] = np.where(top_strategy.eq(key), 1.0 + (top_score * self.config.score_boost_multiplier), 1.0)

        return score_matrix, top_strategy, boost

    def build_regime_frame(self):
        df = self._prepare_features()
        score_matrix, top_strategy, score_boost = self._compute_strategy_score_matrix(df)
        cfg = self.config

        trend_signal = ((df["hurst_z"] > np.maximum(df["HURST_UPPER"].fillna(cfg.z_trigger_level), cfg.z_trigger_level)) & (df["trend_z"] > np.maximum(df["TREND_UPPER"].fillna(cfg.z_trigger_level), cfg.z_trigger_level)) & df["TREND_ALLOWED"])
        mean_signal = ((df["mean_z"] < np.minimum(df["MEAN_LOWER"].fillna(-cfg.z_trigger_level), -cfg.z_trigger_level)) | (df["hurst_z"] < np.minimum(df["HURST_LOWER"].fillna(-cfg.z_trigger_level), -cfg.z_trigger_level)))
        breakout_signal = df["breakout_z"] > np.maximum(df["BREAKOUT_UPPER"].fillna(cfg.z_trigger_level), cfg.z_trigger_level)
        flow_signal = df["flow_z"].abs() > np.maximum(df["FLOW_UPPER"].fillna(cfg.z_trigger_level), cfg.z_trigger_level)
        event_signal = df["gap_z"].abs() > cfg.gap_z_trigger_level

        candidate = pd.Series("mean_reversion", index=df.index, dtype=object)
        candidate = candidate.mask(trend_signal, "trend_following")
        candidate = candidate.mask(flow_signal, "vwap")
        candidate = candidate.mask(breakout_signal, "opening_range_breakout")
        candidate = candidate.mask(event_signal & df["LIQUIDITY_OK"], "index_rebalancing")
        candidate = candidate.mask(df["NOISY_CHOP"] | (~mean_signal & ~trend_signal & ~breakout_signal & ~flow_signal), "no_trade")

        hard_stop = (~df["LIQUIDITY_OK"]) | df["GAP_FREEZE"]
        active_strategy = self._apply_regime_cooldown(candidate, hard_stop)

        raw_weights = pd.DataFrame(0.0, index=df.index, columns=self.STRATEGY_KEYS)
        raw_weights["trend_following"] = sigmoid(df["trend_z"] * cfg.trend_bias_scale) * trend_signal.astype(float)
        raw_weights["mean_reversion"] = sigmoid((-df["mean_z"]) * cfg.mean_bias_scale) * mean_signal.astype(float)
        raw_weights["opening_range_breakout"] = sigmoid(df["breakout_z"] * cfg.breakout_bias_scale) * breakout_signal.astype(float)
        raw_weights["vwap"] = sigmoid(df["flow_z"].abs() * cfg.flow_bias_scale) * flow_signal.astype(float)
        raw_weights["index_rebalancing"] = sigmoid(df["gap_z"].abs() * cfg.event_bias_scale) * event_signal.astype(float)

        active_mask = pd.DataFrame(False, index=df.index, columns=self.STRATEGY_KEYS)
        for key in self.STRATEGY_KEYS:
            active_mask[key] = active_strategy.eq(key)
        raw_weights = raw_weights.where(active_mask, raw_weights * 0.25)
        raw_weights = raw_weights.mul(score_boost, axis=0)
        raw_weights = raw_weights.where(~pd.DataFrame(np.repeat(hard_stop.to_numpy()[:, None], len(raw_weights.columns), axis=1), index=raw_weights.index, columns=raw_weights.columns), 0.0)

        weight_sum = raw_weights.sum(axis=1).replace(0, np.nan)
        weights = raw_weights.div(weight_sum, axis=0).fillna(0.0)

        relative_vol_ratio = (df["ASSET_VOL"] / df["BENCH_VOL"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        volatility_target_scale = (cfg.volatility_target / relative_vol_ratio).clip(lower=cfg.minimum_position_scale, upper=cfg.maximum_position_scale).fillna(cfg.minimum_position_scale)

        realized_edge = score_matrix.max(axis=1).clip(lower=0)
        realized_var = (df["ASSET_VOL"] ** 2).replace(0, np.nan)
        kelly_scale = ((realized_edge / realized_var) * cfg.kelly_fraction).clip(lower=cfg.minimum_position_scale, upper=cfg.maximum_position_scale).fillna(cfg.minimum_position_scale)
        position_scale = np.minimum(volatility_target_scale, kelly_scale)

        atr_ratio = (df["ATR_PCT"] / df["ATR_PCT"].rolling(window=cfg.base_lookback, min_periods=max(20, cfg.base_lookback // 5)).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        entry_buffer = df["ATR"].fillna(0) * cfg.slippage_atr_multiplier * atr_ratio
        switch_cost = weights.diff().abs().sum(axis=1).fillna(0) * (cfg.commission_pct + cfg.slippage_pct)
        expected_alpha = score_matrix.max(axis=1).clip(lower=0) * df["ATR_PCT"]
        switch_allowed = expected_alpha > switch_cost

        stop_mult = pd.Series(cfg.stop_base_multiplier, index=df.index)
        stop_mult = stop_mult.mask(active_strategy.eq("trend_following"), cfg.stop_trend_multiplier)
        stop_mult = stop_mult.mask(active_strategy.eq("mean_reversion"), cfg.stop_mean_reversion_multiplier)
        stop_mult = stop_mult.mask(active_strategy.eq("opening_range_breakout"), cfg.stop_breakout_multiplier)
        stop_mult = stop_mult.mask(active_strategy.eq("vwap"), cfg.stop_flow_multiplier)
        stop_mult = stop_mult.mask(active_strategy.eq("index_rebalancing"), cfg.stop_breakout_multiplier)
        stop_mult = stop_mult.mask(active_strategy.eq("no_trade"), 0.0)

        regime = pd.Series("FROZEN", index=df.index, dtype=object)
        regime = regime.mask(active_strategy.eq("trend_following"), "TRENDING")
        regime = regime.mask(active_strategy.eq("mean_reversion"), "MEAN_REVERTING")
        regime = regime.mask(active_strategy.eq("opening_range_breakout"), "BREAKOUT")
        regime = regime.mask(active_strategy.eq("vwap"), "FLOW")
        regime = regime.mask(active_strategy.eq("index_rebalancing"), "EVENT_DRIVEN")

        output = pd.DataFrame(
            {
                "regime": regime,
                "raw_candidate": candidate,
                "recommended_strategy": candidate,
                "active_strategy": active_strategy,
                "confidence": weights.max(axis=1).fillna(0),
                "hurst": df["HURST"],
                "efficiency_ratio": df["ER"],
                "adaptive_window": df["ADAPTIVE_WINDOW"],
                "relative_strength_20": df["RS_20"].fillna(0),
                "switch_allowed": switch_allowed.fillna(False),
                "expected_alpha": expected_alpha.fillna(0),
                "estimated_switch_cost": switch_cost.fillna(0),
                "stop_loss_multiplier": stop_mult.fillna(cfg.stop_base_multiplier),
                "position_scale": position_scale.fillna(cfg.minimum_position_scale),
                "gap_freeze": df["GAP_FREEZE"].fillna(False),
                "liquidity_ok": df["LIQUIDITY_OK"].fillna(False),
                "volatility_normalized": relative_vol_ratio.fillna(1.0),
                "entry_buffer": entry_buffer.fillna(0),
                "top_strategy_score": top_strategy,
                "execution_state": np.where((~df["GAP_FREEZE"]) & df["LIQUIDITY_OK"] & switch_allowed.fillna(False), "READY", "PAUSE"),
            },
            index=df.index,
        )

        for key in self.STRATEGY_KEYS:
            output[f"{key}_weight"] = weights[key]
            output[f"{key}_ir"] = score_matrix[key]

        return output


class ExecutionGuard:
    """
    Lightweight execution policy intended to run on a faster cadence than regime updates.
    """

    def __init__(self, regime_frame):
        self.regime_frame = regime_frame.copy()

    def build_execution_frame(self):
        frame = self.regime_frame.copy()
        frame["can_trade"] = frame["switch_allowed"] & frame["liquidity_ok"] & (~frame["gap_freeze"])
        frame["execution_state"] = np.where(frame["can_trade"], "READY", "PAUSE")
        return frame
