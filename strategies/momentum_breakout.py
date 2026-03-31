import pandas as pd
import numpy as np

def strategy_momentum_breakout(df):
    """
    Bull Volatile: Donchian Channel breakouts + Volume Confirmation.
    """
    df_temp = df.copy()
    df_temp['Highest_20'] = df_temp['High'].rolling(20).max().shift(1)
    df_temp['Vol_MA'] = df_temp['Volume'].rolling(20).mean().shift(1)
    
    buy_sig = (df_temp['Close'] > df_temp['Highest_20']) & (df_temp['Volume'] > df_temp['Vol_MA'])
    
    current_pos = 0
    signal_list = []
    
    for i in range(len(df_temp)):
        if current_pos == 0:
            if buy_sig.iloc[i]:
                current_pos = 1
        # Will rely strictly globally tracking Dynamic Exits natively via the master state machine.
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)
