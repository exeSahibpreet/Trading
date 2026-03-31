import pandas as pd
import numpy as np

def strategy_exhaustion_play(df):
    """Parabolic: Entries on severe 3 Deviation stretches mapping mean reversion."""
    df_temp = df.copy()
    df_temp['SMA_20'] = df_temp['Close'].rolling(20).mean()
    df_temp['STD'] = df_temp['Close'].rolling(20).std()
    df_temp['UPPER_3_BB'] = df_temp['SMA_20'] + 3*df_temp['STD']
    df_temp['LOWER_3_BB'] = df_temp['SMA_20'] - 3*df_temp['STD']
    
    sell_sig = df_temp['Close'] > df_temp['UPPER_3_BB']
    buy_sig = df_temp['Close'] < df_temp['LOWER_3_BB']
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]: current_pos = 1
        elif sell_sig.iloc[i]: current_pos = -1
        elif (current_pos == 1 and df_temp['Close'].iloc[i] >= df_temp['SMA_20'].iloc[i]) or \
             (current_pos == -1 and df_temp['Close'].iloc[i] <= df_temp['SMA_20'].iloc[i]):
            current_pos = 0
        signals.iloc[i] = current_pos
    return signals
