import pandas as pd
import numpy as np


def strategy_opening_range_breakout(df, context=None):
    df_temp = df.copy()
    daily_range = (df_temp["High"] - df_temp["Low"]).replace(0, np.nan)
    df_temp["ORB_HIGH"] = df_temp["Open"] + (daily_range * 0.35)
    df_temp["ORB_LOW"] = df_temp["Open"] - (daily_range * 0.35)
    df_temp["ROLL_HIGH"] = df_temp["High"].rolling(window=20).max().shift(1)
    df_temp["ROLL_LOW"] = df_temp["Low"].rolling(window=20).min().shift(1)
    df_temp["VOL_MA_20"] = df_temp["Volume"].rolling(window=20).mean().shift(1)

    signals = pd.Series(0, index=df_temp.index, dtype=float)
    current_pos = 0

    for i in range(len(df_temp)):
        row = df_temp.iloc[i]
        if pd.isna(row["ROLL_HIGH"]) or pd.isna(row["ROLL_LOW"]) or pd.isna(row["VOL_MA_20"]) or pd.isna(row["ORB_HIGH"]) or pd.isna(row["ORB_LOW"]):
            signals.iloc[i] = 0
            continue

        breakout_up = (
            row["Close"] > row["ORB_HIGH"]
            and row["High"] >= row["ROLL_HIGH"]
            and row["Volume"] > row["VOL_MA_20"] * 1.2
        )
        breakout_down = (
            row["Close"] < row["ORB_LOW"]
            and row["Low"] <= row["ROLL_LOW"]
            and row["Volume"] > row["VOL_MA_20"] * 1.2
        )

        if breakout_up:
            current_pos = 1
        elif breakout_down:
            current_pos = -1
        elif current_pos == 1 and row["Close"] < row["Open"]:
            current_pos = 0
        elif current_pos == -1 and row["Close"] > row["Open"]:
            current_pos = 0

        signals.iloc[i] = current_pos

    return signals.fillna(0)
