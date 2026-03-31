import pandas as pd


class Backtester:
    def __init__(
        self,
        initial_capital=100000,
        risk_per_trade=0.02,
        brokerage_pct=0.0003,
        min_brokerage=20,
        slippage=0.001,
        mode="equity",
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.brokerage_pct = brokerage_pct
        self.min_brokerage = min_brokerage
        self.slippage = 0.002 if mode == "fno" else slippage
        self.leverage = 5.0 if mode == "fno" else 1.0

    def run(self, df, signals):
        signals = pd.Series(signals, index=df.index).fillna(0)

        cash = float(self.initial_capital)
        open_trade = None
        trades = []
        equity_curve = []
        equity_index = []

        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i - 1]
            desired_signal = int(signals.iloc[i - 1])
            open_px = float(df["Open"].iloc[i])
            close_px = float(df["Close"].iloc[i])
            atr = float(df["ATR"].iloc[i - 1]) if pd.notna(df["ATR"].iloc[i - 1]) else open_px * 0.03
            atr = atr if atr > 0 else open_px * 0.03

            if open_trade is not None:
                should_exit = desired_signal != open_trade["direction"]
                if should_exit:
                    exit_price = open_px * (1 - self.slippage) if open_trade["direction"] == 1 else open_px * (1 + self.slippage)
                    trade_val = open_trade["qty"] * exit_price
                    exit_brokerage = max(self.min_brokerage, trade_val * self.brokerage_pct)
                    pnl = (exit_price - open_trade["entry_price"]) * open_trade["qty"] * open_trade["direction"]
                    pnl -= (open_trade["entry_brokerage"] + exit_brokerage)
                    cash += open_trade["margin"] + pnl
                    open_trade.update(
                        {
                            "exit_date": date.strftime("%Y-%m-%d"),
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "reason": "SIGNAL_CHANGE",
                        }
                    )
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is None and desired_signal != 0:
                risk_budget = cash * self.risk_per_trade * self.leverage
                stop_distance = max(atr * 2, open_px * 0.01)
                qty = max(1, int(risk_budget / stop_distance))
                entry_price = open_px * (1 + self.slippage) if desired_signal == 1 else open_px * (1 - self.slippage)
                margin = (qty * entry_price) / self.leverage
                entry_brokerage = max(self.min_brokerage, qty * entry_price * self.brokerage_pct)

                if margin + entry_brokerage <= cash:
                    cash -= margin + entry_brokerage
                    open_trade = {
                        "date": prev_date.strftime("%Y-%m-%d"),
                        "direction": desired_signal,
                        "qty": qty,
                        "entry_price": entry_price,
                        "entry_brokerage": entry_brokerage,
                        "margin": margin,
                    }

            equity = cash
            if open_trade is not None:
                mark_pnl = (close_px - open_trade["entry_price"]) * open_trade["qty"] * open_trade["direction"]
                equity += open_trade["margin"] + mark_pnl

            equity_curve.append(equity)
            equity_index.append(date)

        if open_trade is not None and len(df) > 1:
            exit_date = df.index[-1]
            exit_price = float(df["Close"].iloc[-1])
            trade_val = open_trade["qty"] * exit_price
            exit_brokerage = max(self.min_brokerage, trade_val * self.brokerage_pct)
            pnl = (exit_price - open_trade["entry_price"]) * open_trade["qty"] * open_trade["direction"]
            pnl -= (open_trade["entry_brokerage"] + exit_brokerage)
            cash += open_trade["margin"] + pnl
            open_trade.update(
                {
                    "exit_date": exit_date.strftime("%Y-%m-%d"),
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": "END_OF_SIM",
                }
            )
            trades.append(open_trade)
            if equity_index:
                equity_curve[-1] = cash

        return pd.Series(equity_curve, index=equity_index), trades


class MultiStrategyBacktester:
    def __init__(self, mode="equity", initial_capital=100000):
        self.mode = mode
        self.initial_capital = initial_capital
        self.risk_pct = 0.01
        self.brokerage_pct = 0.0003
        self.min_brokerage = 20
        self.slippage = 0.002 if mode == "fno" else 0.001
        self.leverage = 5.0 if mode == "fno" else 1.0
        self.max_concurrent_trades = 3

    def run_portfolio(self, df, strategies_signals, portfolio_allocations, regime_data=None):
        sig_df = pd.DataFrame(strategies_signals).reindex(df.index).fillna(0)
        alloc_df = portfolio_allocations.reindex(df.index).fillna(0)
        regime_df = regime_data.reindex(df.index) if regime_data is not None else None

        cash = float(self.initial_capital)
        open_trades = []
        all_trades = {strategy: [] for strategy in sig_df.columns}
        combined_equity = []

        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i - 1]
            open_px = float(df["Open"].iloc[i])
            close_px = float(df["Close"].iloc[i])
            atr = float(df["ATR"].iloc[i - 1]) if pd.notna(df["ATR"].iloc[i - 1]) else open_px * 0.03
            atr = atr if atr > 0 else open_px * 0.03

            for trade in list(open_trades):
                expected_signal = int(sig_df.loc[prev_date, trade["strategy"]])
                allocation_live = float(alloc_df.loc[prev_date, trade["strategy"]])
                stop_hit = False
                exit_price = open_px

                if trade["direction"] == 1 and float(df["Low"].iloc[i]) <= trade["stop_loss"]:
                    stop_hit = True
                    exit_price = min(open_px, trade["stop_loss"]) * (1 - self.slippage)
                elif trade["direction"] == -1 and float(df["High"].iloc[i]) >= trade["stop_loss"]:
                    stop_hit = True
                    exit_price = max(open_px, trade["stop_loss"]) * (1 + self.slippage)

                if stop_hit or expected_signal != trade["direction"] or allocation_live <= 0:
                    if not stop_hit:
                        exit_price = open_px * (1 - self.slippage) if trade["direction"] == 1 else open_px * (1 + self.slippage)
                    trade_val = trade["qty"] * exit_price
                    exit_brokerage = max(self.min_brokerage, trade_val * self.brokerage_pct)
                    pnl = (exit_price - trade["entry_price"]) * trade["qty"] * trade["direction"]
                    pnl -= (trade["entry_brokerage"] + exit_brokerage)
                    cash += trade["margin"] + pnl
                    trade.update(
                        {
                            "exit_date": date.strftime("%Y-%m-%d"),
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "reason": "STOP_LOSS" if stop_hit else "ROUTER_SWITCH",
                        }
                    )
                    all_trades[trade["strategy"]].append(trade)
                    open_trades.remove(trade)

            candidates = []
            if len(open_trades) < self.max_concurrent_trades:
                for strategy in sig_df.columns:
                    if any(trade["strategy"] == strategy for trade in open_trades):
                        continue
                    signal = int(sig_df.loc[prev_date, strategy])
                    allocation = float(alloc_df.loc[prev_date, strategy])
                    if signal != 0 and allocation > 0:
                        candidates.append((strategy, signal, allocation))

            candidates.sort(key=lambda item: item[2], reverse=True)

            for strategy, signal, allocation in candidates:
                if len(open_trades) >= self.max_concurrent_trades:
                    break

                risk_budget = cash * self.risk_pct * self.leverage * allocation
                stop_multiplier = 2.0
                entry_buffer = 0.0
                if regime_df is not None and "stop_loss_multiplier" in regime_df.columns and prev_date in regime_df.index:
                    stop_multiplier = float(regime_df.loc[prev_date, "stop_loss_multiplier"])
                if regime_df is not None and "entry_buffer" in regime_df.columns and prev_date in regime_df.index:
                    entry_buffer = float(regime_df.loc[prev_date, "entry_buffer"])
                stop_distance = max(atr * stop_multiplier, open_px * 0.01)
                qty = max(1, int(risk_budget / stop_distance))
                directional_buffer = entry_buffer / max(open_px, 1e-9)
                entry_price = open_px * (1 + self.slippage + directional_buffer) if signal == 1 else open_px * (1 - self.slippage - directional_buffer)
                margin = (qty * entry_price) / self.leverage
                entry_brokerage = max(self.min_brokerage, qty * entry_price * self.brokerage_pct)

                if margin + entry_brokerage > cash:
                    continue

                cash -= margin + entry_brokerage
                open_trades.append(
                    {
                        "strategy": strategy,
                        "date": prev_date.strftime("%Y-%m-%d"),
                        "direction": signal,
                        "qty": qty,
                        "entry_price": entry_price,
                        "entry_brokerage": entry_brokerage,
                        "margin": margin,
                        "stop_loss": entry_price - stop_distance if signal == 1 else entry_price + stop_distance,
                    }
                )

            equity = cash
            for trade in open_trades:
                mark_pnl = (close_px - trade["entry_price"]) * trade["qty"] * trade["direction"]
                equity += trade["margin"] + mark_pnl

            combined_equity.append(equity)

        for trade in list(open_trades):
            exit_price = float(df["Close"].iloc[-1])
            trade_val = trade["qty"] * exit_price
            exit_brokerage = max(self.min_brokerage, trade_val * self.brokerage_pct)
            pnl = (exit_price - trade["entry_price"]) * trade["qty"] * trade["direction"]
            pnl -= (trade["entry_brokerage"] + exit_brokerage)
            trade.update(
                {
                    "exit_date": df.index[-1].strftime("%Y-%m-%d"),
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": "END_OF_SIM",
                }
            )
            all_trades[trade["strategy"]].append(trade)

        strategy_equities = {}
        combined_index = df.index[1:]

        for strategy in sig_df.columns:
            pnl_curve = pd.Series(float(self.initial_capital), index=combined_index)
            running = float(self.initial_capital)
            trades = sorted(all_trades[strategy], key=lambda item: item.get("exit_date", ""))
            trade_map = {trade["exit_date"]: trade["pnl"] for trade in trades if "exit_date" in trade}
            for dt in combined_index:
                running += trade_map.get(dt.strftime("%Y-%m-%d"), 0.0)
                pnl_curve.loc[dt] = running
            strategy_equities[strategy] = pnl_curve

        return pd.Series(combined_equity, index=combined_index), strategy_equities, all_trades
