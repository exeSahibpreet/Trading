import pandas as pd
import numpy as np

def strategy_vix_scalper(df):
    """Bear Volatile: Mean reversion of ATR spikes."""
    df_temp = df.copy()
    df_temp['ATR_SMA'] = df_temp['ATR'].rolling(20).mean()
    df_temp['ATR_STD'] = df_temp['ATR'].rolling(20).std()
    
    atr_spike = df_temp['ATR'] > (df_temp['ATR_SMA'] + 2 * df_temp['ATR_STD'])
    buy_sig = atr_spike & (df_temp['Close'] < df_temp['Close'].rolling(5).mean())
    sell_sig = df_temp['Close'] > df_temp['Close'].rolling(10).mean()
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]: current_pos = 1
        elif sell_sig.iloc[i]: current_pos = 0
        signals.iloc[i] = current_pos
    return signals
