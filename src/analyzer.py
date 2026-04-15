import pandas as pd
import numpy as np
from scipy import stats

def load_stock_data(file_path='data/stock_timeseries.csv'):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_returns(df):
    df = df.sort_values(['stock_symbol', 'date'])
    
    df['daily_return'] = df.groupby('stock_symbol')['close_price'].pct_change() * 100
    
    df['log_return'] = df.groupby('stock_symbol')['close_price'].apply(
        lambda x: np.log(x / x.shift(1))
    ).reset_index(level=0, drop=True)
    
    return df

def calculate_volatility(df, window=5):
    df = df.sort_values(['stock_symbol', 'date'])
    
    rolling_std = df.groupby('stock_symbol')['daily_return'].transform(
        lambda x: x.rolling(window=window).std()
    )
    
    daily_vol = rolling_std
    annual_factor = np.sqrt(252)
    
    scaled_vol = daily_vol * annual_factor
    df['volatility'] = scaled_vol
    
    return df

def calculate_moving_averages(df, windows=[5, 10, 20]):
    df = df.sort_values(['stock_symbol', 'date'])
    
    for window in windows:
        df[f'ma_{window}'] = df.groupby('stock_symbol')['close_price'].transform(
            lambda x: x.rolling(window=window).mean()
        )
    
    return df

def calculate_rsi(df, window=14):
    df = df.sort_values(['stock_symbol', 'date'])
    
    delta = df.groupby('stock_symbol')['close_price'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = df.groupby('stock_symbol').apply(
        lambda x: gain.loc[x.index].ewm(alpha=1/window, adjust=False).mean()
    ).reset_index(level=0, drop=True)
    
    avg_loss = df.groupby('stock_symbol').apply(
        lambda x: loss.loc[x.index].ewm(alpha=1/window, adjust=False).mean()
    ).reset_index(level=0, drop=True)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.sort_values(['stock_symbol', 'date'])
    
    ema_fast = df.groupby('stock_symbol')['close_price'].transform(
        lambda x: x.ewm(span=fast, adjust=False).mean()
    )
    
    ema_slow = df.groupby('stock_symbol')['close_price'].transform(
        lambda x: x.ewm(span=slow, adjust=False).mean()
    )
    
    df['macd'] = ema_fast - ema_slow
    
    df['macd_signal'] = df.groupby('stock_symbol')['macd'].transform(
        lambda x: x.ewm(span=signal, adjust=False).mean()
    )
    
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    df = df.sort_values(['stock_symbol', 'date'])
    
    df['bb_middle'] = df.groupby('stock_symbol')['close_price'].transform(
        lambda x: x.rolling(window=window).mean()
    )
    
    df['bb_std'] = df.groupby('stock_symbol')['close_price'].transform(
        lambda x: x.rolling(window=window).std()
    )
    
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * num_std)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * num_std)
    
    df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def analyze_stock_performance(df):
    stocks = df['stock_symbol'].unique()
    performance = {}
    
    for stock in stocks:
        stock_data = df[df['stock_symbol'] == stock]
        
        total_return = ((stock_data['close_price'].iloc[-1] / stock_data['close_price'].iloc[0]) - 1) * 100
        
        daily_returns = stock_data['daily_return'].dropna()
        volatility = daily_returns.std()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        max_return = daily_returns.max()
        min_return = daily_returns.min()
        
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        
        performance[stock] = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_daily_return': max_return,
            'min_daily_return': min_return,
            'positive_days': positive_days,
            'total_days': total_days,
            'win_rate': (positive_days / total_days) * 100 if total_days > 0 else 0
        }
    
    return performance

def calculate_correlation_matrix(df):
    pivot_df = df.pivot_table(index='date', columns='stock_symbol', values='close_price')
    correlation_matrix = pivot_df.corr()
    
    return correlation_matrix

def detect_trend(df):
    df = df.sort_values(['stock_symbol', 'date'])
    
    def linear_trend(prices):
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend': 'upward' if slope > 0 else 'downward'
        }
    
    trend_results = df.groupby('stock_symbol').apply(
        lambda x: linear_trend(x['close_price'].values)
    )
    
    return trend_results

def calculate_beta(df, market_col='close_price'):
    stocks = df['stock_symbol'].unique()
    betas = {}
    
    for stock in stocks:
        if stock == 'AAPL':
            continue
        
        stock_data = df[df['stock_symbol'] == stock]
        market_data = df[df['stock_symbol'] == 'AAPL']
        
        merged = stock_data.merge(market_data, on='date', suffixes=('', '_market'))
        
        if len(merged) > 2:
            covariance = np.cov(merged['daily_return'], merged['daily_return_market'])[0][1]
            market_variance = np.var(merged['daily_return_market'])
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            betas[stock] = beta
    
    return betas

def perform_comprehensive_analysis(df):
    df = calculate_returns(df)
    df = calculate_volatility(df)
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    
    results = {
        'processed_data': df,
        'performance': analyze_stock_performance(df),
        'correlations': calculate_correlation_matrix(df),
        'trends': detect_trend(df),
        'betas': calculate_beta(df)
    }
    
    return results

if __name__ == '__main__':
    df = load_stock_data()
    print(f"Loaded {len(df)} records from {df['stock_symbol'].nunique()} stocks")
    
    results = perform_comprehensive_analysis(df)
    
    print("\nStock Performance:")
    for stock, perf in results['performance'].items():
        print(f"  {stock}: Return={perf['total_return']:.2f}%, Volatility={perf['volatility']:.2f}%")
