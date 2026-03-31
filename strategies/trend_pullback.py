import pandas as pd
import numpy as np

def strategy_trend_pullback_v4(df):
    """
    Trade strictly when Close > SMA_200. Entry on pullback to EMA_20 with 40<=RSI<=50.
    Exit on trailing stop (crossing below EMA_20).
    """
    df_temp = df.copy()
    if len(df_temp) < 200:
        return pd.Series(0, index=df_temp.index)
        
    df_temp['SMA_200'] = df_temp['Close'].rolling(window=200).mean()
    df_temp['EMA_20'] = df_temp['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_temp['RSI'] = 100 - (100 / (1 + rs))
    
    buy_sig = (df_temp['Close'] > df_temp['SMA_200']) & \
              (df_temp['Low'] <= df_temp['EMA_20']) & \
              (df_temp['Close'] >= df_temp['EMA_20']) & \
              (df_temp['RSI'] >= 40) & (df_temp['RSI'] <= 50)
              
    sell_sig = df_temp['Close'] < df_temp['EMA_20']
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = 0
        signals.iloc[i] = current_pos
        
    return signals
