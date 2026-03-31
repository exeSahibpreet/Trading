import pandas as pd
import numpy as np


def strategy_index_rebalancing(df, context=None):
    df_temp = df.copy()
    prev_close = df_temp["Close"].shift(1)
    df_temp["VOL_MA_20"] = df_temp["Volume"].rolling(window=20).mean()
    df_temp["PRICE_JUMP"] = ((df_temp["Close"] - prev_close) / prev_close.replace(0, np.nan)).fillna(0)
    df_temp["FLOW_SCORE"] = (df_temp["Volume"] / df_temp["VOL_MA_20"].replace(0, np.nan)).fillna(0)
    month_end_window = df_temp.index.to_series().dt.is_month_end | (df_temp.index.to_series().dt.day >= 25)

    signals = pd.Series(0, index=df_temp.index, dtype=float)
    current_pos = 0
    hold_bars = 0

    for i in range(len(df_temp)):
        row = df_temp.iloc[i]
        if pd.isna(row["VOL_MA_20"]):
            signals.iloc[i] = 0
            continue

        inclusion_flow = month_end_window.iloc[i] and row["FLOW_SCORE"] > 1.8 and row["PRICE_JUMP"] > 0.025
        deletion_flow = month_end_window.iloc[i] and row["FLOW_SCORE"] > 1.8 and row["PRICE_JUMP"] < -0.025

        if inclusion_flow:
            current_pos = 1
            hold_bars = 5
        elif deletion_flow:
            current_pos = -1
            hold_bars = 5
        elif hold_bars > 0:
            hold_bars -= 1
        else:
            current_pos = 0

        signals.iloc[i] = current_pos

    return signals.fillna(0)
