import pandas as pd
import numpy as np

# --- 1. MA Crossover ---
def strategy_ma_crossover(df, short_window=10, long_window=50):
    df_temp = df.copy()
    df_temp['SMA_short'] = df_temp['Close'].rolling(window=short_window).mean()
    df_temp['SMA_long'] = df_temp['Close'].rolling(window=long_window).mean()
    
    signals = pd.Series(0, index=df_temp.index)
    signals[df_temp['SMA_short'] > df_temp['SMA_long']] = 1
    signals[df_temp['SMA_short'] < df_temp['SMA_long']] = -1
    return signals

# --- 2. RSI Mean Reversion ---
def strategy_rsi_mean_reversion(df, period=14):
    df_temp = df.copy()
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    buy_sig = rsi < 30
    sell_sig = rsi > 70
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = -1
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 3. Breakout Strategy ---
def strategy_breakout(df, window=20):
    df_temp = df.copy()
    high_roll = df_temp['High'].rolling(window=window).max().shift(1)
    low_roll = df_temp['Low'].rolling(window=window).min().shift(1)
    vol_roll = df_temp['Volume'].rolling(window=window).mean().shift(1)
    
    buy_sig = (df_temp['Close'] > high_roll) & (df_temp['Volume'] > vol_roll)
    sell_sig = (df_temp['Close'] < low_roll) & (df_temp['Volume'] > vol_roll)
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = -1
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 4. Volume-Based ---
def strategy_volume_based(df, vol_window=20):
    df_temp = df.copy()
    vol_ma = df_temp['Volume'].rolling(window=vol_window).mean()
    price_up = df_temp['Close'] > df_temp['Close'].shift(1)
    price_down = df_temp['Close'] < df_temp['Close'].shift(1)
    high_vol = df_temp['Volume'] > vol_ma
    
    buy_sig = price_up & high_vol
    sell_sig = price_down & high_vol
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = -1
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 5. Pairs Trading ---
def strategy_pairs_trading(df_asset, df_bench=None, window=20):
    if df_bench is None or len(df_asset) != len(df_bench):
        return pd.Series(0, index=df_asset.index)
        
    ret_asset = np.log(df_asset['Close'] / df_asset['Close'].shift(1))
    ret_bench = np.log(df_bench['Close'] / df_bench['Close'].shift(1))
    spread = ret_asset - ret_bench
    
    if len(spread) == 0:
        return pd.Series(0, index=df_asset.index)
        
    mean_spread = spread.rolling(window=window).mean()
    std_spread = spread.rolling(window=window).std()
    
    z_score = (spread - mean_spread) / std_spread
    buy_sig = z_score < -2.0
    sell_sig = z_score > 2.0
    
    z_prev = z_score.shift(1)
    exit_sig = ((z_prev < 0) & (z_score > 0)) | ((z_prev > 0) & (z_score < 0))
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_asset)):
        if pd.isna(z_score.iloc[i]):
            signal_list.append(0)
            continue
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = -1
        elif exit_sig.iloc[i]:
            current_pos = 0
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_asset.index)

# --- 6. Trend + Pullback Strategy ---
def strategy_trend_pullback(df):
    """Buy strictly in uptrends, during oversold pullbacks."""
    df_temp = df.copy()
    df_temp['SMA_200'] = df_temp['Close'].rolling(window=200).mean()
    df_temp['EMA_20'] = df_temp['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_temp['RSI'] = 100 - (100 / (1 + rs))
    
    buy_sig = (df_temp['Close'] > df_temp['SMA_200']) & \
              (df_temp['Low'] <= df_temp['EMA_20']) & \
              (df_temp['RSI'] >= 40) & (df_temp['RSI'] <= 50)
              
    sell_sig = df_temp['Close'] < df_temp['EMA_20']
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = 0 
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 7. Volatility Breakout (ATR) ---
def strategy_volatility_breakout(df):
    """Enters long on price rising above previous close + ATR."""
    df_temp = df.copy()
    prev_close = df_temp['Close'].shift(1)
    atr = df_temp['ATR'].shift(1)
    
    buy_sig = df_temp['Close'] > (prev_close + atr)
    sell_sig = df_temp['Close'] < (prev_close - atr)
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if pd.isna(atr.iloc[i]):
            signal_list.append(0)
            continue
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = -1 # Or 0 if long only
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 8. Relative Strength Strategy ---
def strategy_relative_strength(df, all_dfs=None):
    """Evaluates trailing 30-day performance. Buys if ranked within top 3 of the universe."""
    if all_dfs is None or len(all_dfs) == 0:
        return pd.Series(0, index=df.index)
        
    df_temp = df.copy()
    df_temp['Ret30'] = df_temp['Close'].pct_change(periods=30)
    
    universe_rets = pd.DataFrame(index=df.index)
    for ticker, asset_df in all_dfs.items():
        reindexed = asset_df['Close'].reindex(df.index).ffill()
        universe_rets[ticker] = reindexed.pct_change(periods=30)
        
    current_pos = 0
    signal_list = []
    
    for i in range(len(df_temp)):
        row = universe_rets.iloc[i].dropna()
        if len(row) < 3 or pd.isna(df_temp['Ret30'].iloc[i]):
            signal_list.append(current_pos)
            continue
            
        threshold = row.nlargest(3).iloc[-1]
        my_ret = df_temp['Ret30'].iloc[i]
        
        if my_ret >= threshold and my_ret > 0:
            current_pos = 1
        else:
            current_pos = 0 
            
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 9. Filtered Mean Reversion ---
def strategy_filtered_mean_reversion(df):
    """Bollinger bands combined with oversold RSI and 20-MA exits."""
    df_temp = df.copy()
    
    df_temp['SMA_20'] = df_temp['Close'].rolling(window=20).mean()
    df_temp['STD_20'] = df_temp['Close'].rolling(window=20).std()
    df_temp['Upper'] = df_temp['SMA_20'] + (df_temp['STD_20'] * 2)
    df_temp['Lower'] = df_temp['SMA_20'] - (df_temp['STD_20'] * 2)
    
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_temp['RSI'] = 100 - (100 / (1 + rs))
    
    buy_sig = (df_temp['RSI'] < 25) & (df_temp['Close'] <= df_temp['Lower'] * 1.02)
    sell_sig = (df_temp['Close'] >= df_temp['SMA_20'])
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = 0
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)

# --- 10. Multi-Factor Strategy ---
def strategy_multi_factor(df, bench_df=None):
    """Combination of Trend (MA200), Pullback (RSI<40), Volume, and Macro Benchmark."""
    df_temp = df.copy()
    
    df_temp['SMA_200'] = df_temp['Close'].rolling(window=200).mean()
    
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_temp['RSI'] = 100 - (100 / (1 + rs))
    
    df_temp['SMA_Vol_20'] = df_temp['Volume'].rolling(window=20).mean()
    
    buy_sig = (df_temp['Close'] > df_temp['SMA_200']) & \
              (df_temp['RSI'] < 40) & \
              (df_temp['Volume'] > df_temp['SMA_Vol_20'])
              
    if bench_df is not None:
        bench_reindexed = bench_df['Close'].reindex(df_temp.index).ffill()
        bench_sma200 = bench_reindexed.rolling(window=200).mean()
        bench_bull = bench_reindexed > bench_sma200
        buy_sig = buy_sig & bench_bull

    sell_sig = df_temp['Close'] < df_temp['Close'].rolling(window=20).mean()
    
    current_pos = 0
    signal_list = []
    for i in range(len(df_temp)):
        if pd.isna(buy_sig.iloc[i]):
            signal_list.append(0)
            continue
        if buy_sig.iloc[i]:
            current_pos = 1
        elif sell_sig.iloc[i]:
            current_pos = 0
        signal_list.append(current_pos)
        
    return pd.Series(signal_list, index=df_temp.index)
