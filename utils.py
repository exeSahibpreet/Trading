import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, start='2014-01-01', end='2024-01-01'):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df.dropna(inplace=True)
        if len(df) == 0:
            return df
        df['ATR'] = calculate_atr(df)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window).mean()

def train_test_split(df, train_ratio=0.7):
    if len(df) == 0:
        return df, df
    n = int(len(df) * train_ratio)
    return df.iloc[:n], df.iloc[n:]

def calculate_drawdown_series(equity_curve):
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown

def calculate_recovery_metrics(equity_curve, max_dd, total_profit):
    peak = equity_curve.expanding(min_periods=1).max()
    dd_dollars = peak - equity_curve
    max_dd_dollars = dd_dollars.max()
    
    recovery_factor = total_profit / max_dd_dollars if max_dd_dollars > 0 else 0
    
    underwater = equity_curve < peak
    grouper = (~underwater).cumsum()
    consecutive_days = underwater.groupby(grouper).sum()
    max_recovery_days = int(consecutive_days.max()) if not consecutive_days.empty else 0
    
    return recovery_factor, max_recovery_days

def compute_metrics(equity_curve, trades):
    zero_metrics = {
        "final_capital": 0, "total_profit": 0, "return_pct": 0, "annual_return": 0,
        "num_trades": 0, "win_rate": 0, "max_drawdown": 0, "sharpe_ratio": 0, 
        "sortino_ratio": 0, "calmar_ratio": 0, "recovery_factor": 0, 
        "recovery_days": 0, "drawdown_series": []
    }
    if len(trades) == 0 or len(equity_curve) <= 1:
        return zero_metrics
    
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    total_profit = final - initial
    ret_pct = total_profit / initial * 100
    
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25 if days > 0 else 1
    annual_return = (ret_pct / 100) / years if years > 0 else 0
    
    profit_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    col_trades = [t for t in trades if t.get('type') == 'close' or 'exit_price' in t]
    win_rate = (profit_trades / len(col_trades)) * 100 if len(col_trades) > 0 else 0
    
    dd_series = calculate_drawdown_series(equity_curve)
    max_dd = dd_series.min() * 100
    
    rec_factor, rec_days = calculate_recovery_metrics(equity_curve, max_dd, total_profit)
    
    daily_returns = equity_curve.pct_change().dropna()
    std = daily_returns.std()
    sharpe = np.sqrt(252) * (daily_returns.mean() / std) if pd.notna(std) and std != 0 else 0
    
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = neg_returns.std()
    sortino = np.sqrt(252) * (daily_returns.mean() / downside_std) if pd.notna(downside_std) and downside_std != 0 else 0
    
    calmar = (annual_return * 100) / abs(max_dd) if max_dd < 0 else 0
    
    return {
        "final_capital": final,
        "total_profit": total_profit,
        "return_pct": ret_pct,
        "annual_return": annual_return * 100,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "recovery_factor": rec_factor,
        "recovery_days": rec_days,
        "drawdown_series": (dd_series * 100).tolist()
    }

def rank_strategies(results_list):
    """
    results_list: [{'strategy_name', 'train_metrics', 'test_metrics', 'equity', 'drawdown', 'dates'}]
    """
    if not results_list:
        return []

    for res in results_list:
        t_train = res['train_metrics']
        t_test = res['test_metrics']
        
        train_pnl = t_train['total_profit']
        test_pnl = t_test['total_profit']
        train_ann = t_train['annual_return']
        test_ann = t_test['annual_return']
        
        wfe = (test_ann / train_ann) if train_ann > 0 else 0
        wfe = max(0, wfe)
        res['wfe'] = wfe
        
        badges = []
        is_disqualified = False
        
        trades = t_test['num_trades']
        if trades < 30:
            badges.append("Low Sample Size")
            is_disqualified = True
            
        sharpe = t_test['sharpe_ratio']
        if sharpe < 0:
            badges.append("Unacceptable Risk")
            is_disqualified = True
        elif sharpe < 1.0:
            badges.append("Poor Risk-Adjusted")
            
        if train_pnl > 0 and test_pnl < 0:
            badges.append("FAILED OUT-OF-SAMPLE")
            is_disqualified = True
            
        if wfe < 0.5:
            badges.append("Overfitted")
            
        if t_test['max_drawdown'] < -30:
            badges.append("High Drawdown Risk")
            
        res['badges'] = badges
        res['is_disqualified'] = is_disqualified

    calmars = np.array([r['test_metrics']['calmar_ratio'] for r in results_list])
    sharpes = np.array([r['test_metrics']['sharpe_ratio'] for r in results_list])
    wfes = np.array([r['wfe'] for r in results_list])
    dds = np.array([r['test_metrics']['max_drawdown'] for r in results_list])
    rec_factors = np.array([r['test_metrics']['recovery_factor'] for r in results_list])
    wins = np.array([r['test_metrics']['win_rate'] for r in results_list])
    trades = np.array([r['test_metrics']['num_trades'] for r in results_list])

    def minmax(arr):
        if len(arr) == 0: return arr
        if arr.max() == arr.min(): return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    norm_calmar = minmax(np.clip(calmars, -2, 5))
    norm_sharpe = minmax(np.clip(sharpes, -1, 3))
    norm_wfe = minmax(np.clip(wfes, 0, 1.5))
    norm_dd = minmax(np.clip(dds, -50, 0))
    norm_rec = minmax(np.clip(rec_factors, 0, 10))
    norm_win = minmax(wins)
    norm_trades = minmax(np.clip(trades, 0, 100))

    w_calmar = 0.35
    w_sharpe = 0.15
    w_wfe = 0.15
    w_dd = 0.15
    w_rec = 0.10
    w_win = 0.05
    w_trades = 0.05
    
    scores = (w_calmar * norm_calmar) + (w_sharpe * norm_sharpe) + (w_wfe * norm_wfe) + \
             (w_dd * norm_dd) + (w_rec * norm_rec) + (w_win * norm_win) + (w_trades * norm_trades)
    scores = scores * 10
    
    for i, res in enumerate(results_list):
        final_score = scores[i]
        if res['is_disqualified']:
            final_score *= 0.25 
        if res['test_metrics']['max_drawdown'] < -50:
            final_score *= 0.5 
        res['score'] = round(final_score, 2)
        
    return sorted(results_list, key=lambda x: x['score'], reverse=True)
