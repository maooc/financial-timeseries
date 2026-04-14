import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_stock_price(df, stock_symbol, save_path='output/stock_price.png'):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    stock_data = df[df['stock_symbol'] == stock_symbol].sort_values('date')
    
    ax.plot(stock_data['date'], stock_data['close_price'], 'b-', linewidth=2, label='Close Price')
    
    if 'ma_5' in stock_data.columns:
        ax.plot(stock_data['date'], stock_data['ma_5'], 'g--', linewidth=1, alpha=0.7, label='MA 5')
    if 'ma_10' in stock_data.columns:
        ax.plot(stock_data['date'], stock_data['ma_10'], 'orange', linewidth=1, alpha=0.7, label='MA 10')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(f'{stock_symbol} Stock Price', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved stock price to {save_path}")

def plot_returns_distribution(df, stock_symbol, save_path='output/returns_distribution.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    stock_data = df[df['stock_symbol'] == stock_symbol]
    returns = stock_data['daily_return'].dropna()
    
    ax1.hist(returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.boxplot(returns, vert=True)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.set_title('Returns Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved returns distribution to {save_path}")

def plot_volatility_comparison(df, save_path='output/volatility_comparison.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stocks = df['stock_symbol'].unique()
    volatilities = []
    
    for stock in stocks:
        stock_data = df[df['stock_symbol'] == stock]
        volatility = stock_data['daily_return'].std()
        volatilities.append(volatility)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(stocks)))
    
    bars = ax.bar(stocks, volatilities, color=colors, edgecolor='black')
    
    for bar, vol in zip(bars, volatilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{vol:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Stock Symbol', fontsize=12)
    ax.set_ylabel('Volatility (%)', fontsize=12)
    ax.set_title('Volatility Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved volatility comparison to {save_path}")

def plot_correlation_heatmap(correlation_matrix, save_path='output/correlation_heatmap.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlation_matrix.columns)
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    ax.set_title('Stock Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")

def plot_rsi(df, stock_symbol, save_path='output/rsi.png'):
    fig, ax = plt.subplots(figsize=(14, 4))
    
    stock_data = df[df['stock_symbol'] == stock_symbol].sort_values('date')
    
    ax.plot(stock_data['date'], stock_data['rsi'], 'purple', linewidth=1.5)
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    
    ax.fill_between(stock_data['date'], 70, 100, alpha=0.1, color='red')
    ax.fill_between(stock_data['date'], 0, 30, alpha=0.1, color='green')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RSI', fontsize=12)
    ax.set_title(f'{stock_symbol} RSI Indicator', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RSI to {save_path}")

def plot_macd(df, stock_symbol, save_path='output/macd.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    stock_data = df[df['stock_symbol'] == stock_symbol].sort_values('date')
    
    ax1.plot(stock_data['date'], stock_data['close_price'], 'b-', linewidth=1.5, label='Close Price')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{stock_symbol} Price & MACD', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(stock_data['date'], stock_data['macd'], 'b-', linewidth=1, label='MACD')
    ax2.plot(stock_data['date'], stock_data['macd_signal'], 'orange', linewidth=1, label='Signal')
    
    colors = ['green' if val >= 0 else 'red' for val in stock_data['macd_histogram']]
    ax2.bar(stock_data['date'], stock_data['macd_histogram'], color=colors, alpha=0.5, width=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved MACD to {save_path}")

def plot_bollinger_bands(df, stock_symbol, save_path='output/bollinger_bands.png'):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    stock_data = df[df['stock_symbol'] == stock_symbol].sort_values('date')
    
    ax.plot(stock_data['date'], stock_data['close_price'], 'b-', linewidth=2, label='Close Price')
    ax.plot(stock_data['date'], stock_data['bb_upper'], 'r--', linewidth=1, label='Upper Band')
    ax.plot(stock_data['date'], stock_data['bb_middle'], 'g--', linewidth=1, label='Middle Band')
    ax.plot(stock_data['date'], stock_data['bb_lower'], 'r--', linewidth=1, label='Lower Band')
    
    ax.fill_between(stock_data['date'], stock_data['bb_upper'], stock_data['bb_lower'], alpha=0.1, color='gray')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(f'{stock_symbol} Bollinger Bands', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Bollinger Bands to {save_path}")

def generate_all_financial_charts(analysis_results):
    import os
    os.makedirs('output', exist_ok=True)
    
    df = analysis_results['processed_data']
    stocks = df['stock_symbol'].unique()
    
    for stock in stocks:
        try:
            plot_stock_price(df, stock, f'output/price_{stock}.png')
            plot_returns_distribution(df, stock, f'output/returns_{stock}.png')
            plot_rsi(df, stock, f'output/rsi_{stock}.png')
            plot_macd(df, stock, f'output/macd_{stock}.png')
            plot_bollinger_bands(df, stock, f'output/bollinger_{stock}.png')
        except Exception as e:
            print(f"Error plotting charts for {stock}: {e}")
    
    try:
        plot_volatility_comparison(df)
    except Exception as e:
        print(f"Error plotting volatility comparison: {e}")
    
    try:
        plot_correlation_heatmap(analysis_results['correlations'])
    except Exception as e:
        print(f"Error plotting correlation heatmap: {e}")

if __name__ == '__main__':
    from analyzer import load_stock_data, perform_comprehensive_analysis
    
    df = load_stock_data()
    results = perform_comprehensive_analysis(df)
    generate_all_financial_charts(results)
    print("All financial charts generated!")
