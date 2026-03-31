class PortfolioManager:
    """
    Reads the adaptive module output and converts it into portfolio weights.
    """

    STRATEGY_KEYS = [
        "trend_following",
        "mean_reversion",
        "opening_range_breakout",
        "index_rebalancing",
        "vwap",
    ]

    def __init__(self, regimes_df):
        self.regimes_df = regimes_df

    def get_allocation(self, date):
        alloc = {key: 0.0 for key in self.STRATEGY_KEYS}

        if date not in self.regimes_df.index:
            alloc["mean_reversion"] = 0.5
            return alloc

        row = self.regimes_df.loc[date]
        if not bool(row.get("can_trade", True)):
            return alloc

        for key in self.STRATEGY_KEYS:
            weight_col = f"{key}_weight"
            alloc[key] = float(row.get(weight_col, 0.0))

        if row.get("active_strategy", "mean_reversion") == "no_trade":
            return alloc

        total = sum(alloc.values())
        if total <= 0:
            alloc["mean_reversion"] = 0.25
            return alloc

        position_scale = float(row.get("position_scale", 1.0))
        alloc = {key: (value / total) * position_scale for key, value in alloc.items()}

        return alloc
