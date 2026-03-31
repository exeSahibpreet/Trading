import pandas as pd
import numpy as np

def strategy_advanced_mr_vwap(df):
    """Advanced MR: Z-Score based entries anchoring natively to Daily VWAP iterations."""
    df_temp = df.copy()
    
    typ_price = (df_temp['High'] + df_temp['Low'] + df_temp['Close']) / 3
    df_temp['VWAP'] = (typ_price * df_temp['Volume']).rolling(20).sum() / df_temp['Volume'].rolling(20).sum()
    
    df_temp['STD'] = df_temp['Close'].rolling(20).std()
    df_temp['Z_SCORE'] = (df_temp['Close'] - df_temp['VWAP']) / df_temp['STD']
    
    buy_sig = df_temp['Z_SCORE'] < -2.0
    sell_sig = df_temp['Z_SCORE'] > 2.0
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]: current_pos = 1
        elif sell_sig.iloc[i]: current_pos = -1
        elif (current_pos == 1 and df_temp['Z_SCORE'].iloc[i] >= 0) or \
             (current_pos == -1 and df_temp['Z_SCORE'].iloc[i] <= 0):
            current_pos = 0
        signals.iloc[i] = current_pos
    return signals
