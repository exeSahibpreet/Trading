import pandas as pd
import numpy as np

def strategy_trend_follower(df):
    """
    Bull Quiet: EMA Crossover (20/50).
    V7 System assigns ATR Stop Losses natively inside the Event loop automatically.
    """
    df_temp = df.copy()
    df_temp['EMA_20'] = df_temp['Close'].ewm(span=20, adjust=False).mean()
    df_temp['EMA_50'] = df_temp['Close'].ewm(span=50, adjust=False).mean()
    
    buy_sig = df_temp['EMA_20'] > df_temp['EMA_50']
    sell_sig = df_temp['EMA_20'] < df_temp['EMA_50']
    
    current_pos = 0
    signal_list = []
    
    for i in range(len(df_temp)):
        if current_pos == 0:
            if buy_sig.iloc[i]:
                current_pos = 1
        elif current_pos == 1:
            if sell_sig.iloc[i]:
                current_pos = 0
                
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)
