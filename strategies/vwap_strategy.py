import pandas as pd
import numpy as np


def strategy_vwap(df, context=None):
    df_temp = df.copy()
    typical_price = (df_temp["High"] + df_temp["Low"] + df_temp["Close"]) / 3
    volume_sum = df_temp["Volume"].rolling(window=20).sum().replace(0, np.nan)
    df_temp["VWAP_20"] = (typical_price * df_temp["Volume"]).rolling(window=20).sum() / volume_sum
    df_temp["VWAP_DISTANCE"] = ((df_temp["Close"] - df_temp["VWAP_20"]) / df_temp["VWAP_20"]).fillna(0)
    df_temp["VOL_MA_20"] = df_temp["Volume"].rolling(window=20).mean()

    prev_close = df_temp["Close"].shift(1)
    prev_vwap = df_temp["VWAP_20"].shift(1)

    bullish_cross = ((prev_close <= prev_vwap) & (df_temp["Close"] > df_temp["VWAP_20"])).fillna(False)
    bearish_cross = ((prev_close >= prev_vwap) & (df_temp["Close"] < df_temp["VWAP_20"])).fillna(False)
    volume_confirm = (df_temp["Volume"] > df_temp["VOL_MA_20"] * 1.1).fillna(False)

    signals = pd.Series(0, index=df_temp.index, dtype=float)
    current_pos = 0

    for i in range(len(df_temp)):
        row = df_temp.iloc[i]
        if pd.isna(row["VWAP_20"]) or pd.isna(row["VOL_MA_20"]):
            signals.iloc[i] = 0
            continue

        if bullish_cross.iloc[i] and volume_confirm.iloc[i]:
            current_pos = 1
        elif bearish_cross.iloc[i] and volume_confirm.iloc[i]:
            current_pos = -1
        elif current_pos == 1 and row["VWAP_DISTANCE"] < -0.005:
            current_pos = 0
        elif current_pos == -1 and row["VWAP_DISTANCE"] > 0.005:
            current_pos = 0
        elif abs(row["VWAP_DISTANCE"]) < 0.002:
            current_pos = 0

        signals.iloc[i] = current_pos

    return signals.fillna(0)
