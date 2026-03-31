import pandas as pd
import numpy as np

def strategy_volatility_breakout_v4(df):
    """
    Volatility Breakout Strategy (ATR-based)
    Buy when price > previous close + (1 * ATR 14)
    Sell when price < previous close - (1 * ATR 14)
    """
    df_temp = df.copy()
    prev_close = df_temp['Close'].shift(1)
    atr = df_temp['ATR'].shift(1)
    
    buy_sig = df_temp['Close'] > (prev_close + atr)
    sell_sig = df_temp['Close'] < (prev_close - atr)
    
    signals = pd.Series(0, index=df_temp.index)
    current_pos = 0
    for i in range(len(df_temp)):
        if pd.isna(atr.iloc[i]):
            signals.iloc[i] = 0
            continue
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = -1 # Assuming short
        signals.iloc[i] = current_pos
        
    return signals
