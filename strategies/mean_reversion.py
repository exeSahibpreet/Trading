import pandas as pd
import numpy as np


def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def strategy_mean_reversion(df, context=None):
    df_temp = df.copy()
    df_temp["SMA_20"] = df_temp["Close"].rolling(window=20).mean()
    df_temp["STD_20"] = df_temp["Close"].rolling(window=20).std()
    df_temp["BB_UPPER"] = df_temp["SMA_20"] + (2 * df_temp["STD_20"])
    df_temp["BB_LOWER"] = df_temp["SMA_20"] - (2 * df_temp["STD_20"])
    df_temp["RSI_14"] = _rsi(df_temp["Close"], window=14)

    signals = pd.Series(0, index=df_temp.index, dtype=float)
    current_pos = 0

    for i in range(len(df_temp)):
        row = df_temp.iloc[i]
        if pd.isna(row["RSI_14"]) or pd.isna(row["SMA_20"]):
            signals.iloc[i] = 0
            continue

        if current_pos == 0:
            if row["RSI_14"] < 30 and row["Close"] <= row["BB_LOWER"]:
                current_pos = 1
            elif row["RSI_14"] > 70 and row["Close"] >= row["BB_UPPER"]:
                current_pos = -1
        elif current_pos == 1 and (row["Close"] >= row["SMA_20"] or row["RSI_14"] > 55):
            current_pos = 0
        elif current_pos == -1 and (row["Close"] <= row["SMA_20"] or row["RSI_14"] < 45):
            current_pos = 0

        signals.iloc[i] = current_pos

    return signals.fillna(0)
