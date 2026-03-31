import pandas as pd
import numpy as np

def strategy_volatility_expansion(df):
    """The Squeeze: Trade BB expansion natively."""
    df_temp = df.copy()
    df_temp['SMA_20'] = df_temp['Close'].rolling(20).mean()
    df_temp['STD'] = df_temp['Close'].rolling(20).std()
    df_temp['UPPER_BB'] = df_temp['SMA_20'] + 2*df_temp['STD']
    df_temp['LOWER_BB'] = df_temp['SMA_20'] - 2*df_temp['STD']
    
    buy_sig = (df_temp['Close'] > df_temp['UPPER_BB']) & (df_temp['Close'].shift(1) <= df_temp['UPPER_BB'].shift(1))
    sell_sig = (df_temp['Close'] < df_temp['LOWER_BB']) & (df_temp['Close'].shift(1) >= df_temp['LOWER_BB'].shift(1))
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]: current_pos = 1
        elif sell_sig.iloc[i]: current_pos = -1
        signals.iloc[i] = current_pos
    return signals
