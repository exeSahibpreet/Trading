from .trend_following import strategy_trend_following
from .mean_reversion import strategy_mean_reversion
from .opening_range_breakout import strategy_opening_range_breakout
from .index_rebalancing import strategy_index_rebalancing
from .vwap_strategy import strategy_vwap

STRATEGY_REGISTRY = {
    "trend_following": strategy_trend_following,
    "mean_reversion": strategy_mean_reversion,
    "opening_range_breakout": strategy_opening_range_breakout,
    "index_rebalancing": strategy_index_rebalancing,
    "vwap": strategy_vwap,
}

STRATEGY_LABELS = {
    "trend_following": "Trend Following",
    "mean_reversion": "Mean Reversion",
    "opening_range_breakout": "Opening Range Breakout",
    "index_rebalancing": "Index Fund Rebalancing",
    "vwap": "VWAP-Based Strategy",
}


def get_strategy_signal(strategy_name, df, context=None):
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return STRATEGY_REGISTRY[strategy_name](df, context=context or {})
