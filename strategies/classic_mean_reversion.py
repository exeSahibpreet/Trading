import pandas as pd
import numpy as np

def _calc_adx(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    up = df['High'] - df['High'].shift(1)
    dn = df['Low'].shift(1) - df['Low']
    pdm = np.where((up > dn) & (up > 0), up, 0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0)
    tr_sm = tr.rolling(period).sum()
    pdi = 100 * (pd.Series(pdm, index=df.index).rolling(period).sum() / tr_sm)
    mdi = 100 * (pd.Series(mdm, index=df.index).rolling(period).sum() / tr_sm)
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return dx.rolling(period).mean()

def strategy_classic_mean_reversion(df):
    """
    Mean Reverting Stretch: RSI exhaustion + BB Touches
    V6 Circuit Breaker: Hard-disabled if ADX > 70th Percentile natively eliminating runaway trend destruction dynamically.
    """
    df_temp = df.copy()
    
    df_temp['ADX'] = _calc_adx(df_temp)
    # 252-day dynamic percentile baseline evaluating real-time threshold limits
    df_temp['ADX_Q70'] = df_temp['ADX'].rolling(window=252, min_periods=20).quantile(0.70)
    df_temp['ADX_Q70'] = df_temp['ADX_Q70'].fillna(25)
    
    df_temp['SMA_20'] = df_temp['Close'].rolling(window=20).mean()
    df_temp['STD_20'] = df_temp['Close'].rolling(window=20).std()
    df_temp['Lower'] = df_temp['SMA_20'] - (df_temp['STD_20'] * 2.5) 
    
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs = gain / loss
    df_temp['RSI_2'] = 100 - (100 / (1 + rs))
    
    # Execution blocks categorically require ADX explicitly under the runaway threshold recursively.
    buy_sig = (df_temp['RSI_2'] < 10) & (df_temp['Close'] <= df_temp['Lower']) & (df_temp['ADX'] <= df_temp['ADX_Q70'])
    sell_sig = df_temp['Close'] >= df_temp['SMA_20']
    
    current_pos = 0
    signal_list = []
    
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = 0
            
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)
