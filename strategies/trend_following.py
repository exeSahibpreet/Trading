import pandas as pd


def strategy_trend_following(df, context=None):
    df_temp = df.copy()
    df_temp["SMA_50"] = df_temp["Close"].rolling(window=50).mean()
    df_temp["SMA_200"] = df_temp["Close"].rolling(window=200).mean()
    df_temp["TREND_SLOPE"] = df_temp["SMA_50"].diff(5)

    long_signal = (df_temp["SMA_50"] > df_temp["SMA_200"]) & (df_temp["TREND_SLOPE"] > 0)
    short_signal = (df_temp["SMA_50"] < df_temp["SMA_200"]) & (df_temp["TREND_SLOPE"] < 0)

    signals = pd.Series(0, index=df_temp.index, dtype=float)
    signals[long_signal] = 1
    signals[short_signal] = -1
    return signals.fillna(0)
