import pandas as pd
import numpy as np

def strategy_volume_profile(df):
    """Accumulation: Look for volume divergence efficiently matching sideway supports."""
    df_temp = df.copy()
    
    close_diff = df_temp['Close'].diff()
    direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
    df_temp['OBV'] = (direction * df_temp['Volume']).cumsum()
    df_temp['OBV_SMA'] = df_temp['OBV'].rolling(10).mean()
    
    df_temp['SMA_20'] = df_temp['Close'].rolling(20).mean()
    
    buy_sig = (df_temp['OBV'] > df_temp['OBV_SMA']) & (df_temp['Close'] < df_temp['SMA_20'])
    sell_sig = df_temp['Close'] > df_temp['SMA_20']
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]: current_pos = 1
        elif sell_sig.iloc[i]: current_pos = 0
        signals.iloc[i] = current_pos
    return signals
