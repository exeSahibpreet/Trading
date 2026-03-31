import pandas as pd
import numpy as np

def strategy_grid_trader(df):
    """Sideways Quiet: Buy at lower band of 20 SMA, sell at upper."""
    df_temp = df.copy()
    df_temp['SMA_20'] = df_temp['Close'].rolling(20).mean()
    df_temp['UPPER'] = df_temp['SMA_20'] * 1.02
    df_temp['LOWER'] = df_temp['SMA_20'] * 0.98
    
    buy_sig = df_temp['Close'] <= df_temp['LOWER']
    sell_sig = df_temp['Close'] >= df_temp['UPPER']
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]: current_pos = 1
        elif sell_sig.iloc[i]: current_pos = -1
        signals.iloc[i] = current_pos
    return signals
