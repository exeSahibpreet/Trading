import pandas as pd
import numpy as np

def strategy_short_trend(df):
    """
    Bear Regimes: Shorting consecutive breakdowns below 20-day lows.
    """
    df_temp = df.copy()
    df_temp['Lowest_20'] = df_temp['Low'].rolling(20).min().shift(1)
    df_temp['SMA_200'] = df_temp['Close'].rolling(200).mean()
    
    short_sig = (df_temp['Close'] < df_temp['SMA_200']) & (df_temp['Close'] < df_temp['Lowest_20'])
    
    current_pos = 0
    signal_list = []
    
    for i in range(len(df_temp)):
        if current_pos == 0:
            if short_sig.iloc[i]:
                current_pos = -1
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)
