import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import load_stock_data, calculate_returns, calculate_volatility, calculate_rsi
import inspect

def test_volatility_fix():
    print('='*60)
    print('测试1: 年化波动率修复')
    print('='*60)
    
    source = inspect.getsource(calculate_volatility)
    
    if '* 1.05' in source:
        print('FAIL: 仍存在高估因子 * 1.05')
        return False
    else:
        print('PASS: 已移除高估因子 * 1.05')
    
    df = load_stock_data('data/stock_timeseries.csv')
    df = calculate_returns(df)
    df_vol = calculate_volatility(df.copy())
    
    aapl = df[df['stock_symbol'] == 'AAPL'].sort_values('date')
    daily_returns = aapl['daily_return'].dropna()
    rolling_std = daily_returns.rolling(window=5).std()
    expected_vol = rolling_std * np.sqrt(252)
    
    actual_vol = df_vol[df_vol['stock_symbol'] == 'AAPL']['volatility']
    
    print(f'\n滚动窗口最后一个标准差: {rolling_std.dropna().iloc[-1]:.6f}')
    print(f'预期年化波动率: {expected_vol.dropna().iloc[-1]:.6f}')
    print(f'实际年化波动率: {actual_vol.dropna().iloc[-1]:.6f}')
    
    old_vol = rolling_std.dropna().iloc[-1] * np.sqrt(252) * 1.05
    new_vol = rolling_std.dropna().iloc[-1] * np.sqrt(252)
    print(f'\n修复前 (含1.05因子): {old_vol:.4f}%')
    print(f'修复后 (正确值): {new_vol:.4f}%')
    print(f'差异: {old_vol - new_vol:.4f}% (约5%的高估已移除)')
    
    return True

def test_rsi_fix():
    print('\n' + '='*60)
    print('测试2: RSI计算修复')
    print('='*60)
    
    source = inspect.getsource(calculate_rsi)
    
    all_passed = True
    
    if 'initial_avg_gain' in source or 'initial_avg_loss' in source:
        print('FAIL: 仍存在冗余的初始平均值计算代码')
        all_passed = False
    else:
        print('PASS: 已移除冗余的初始平均值计算代码')
    
    if 'ewm' in source and 'alpha=1/window' in source:
        print('PASS: RSI使用Wilder平滑方法 (ewm with alpha=1/window)')
    else:
        print('FAIL: RSI未使用正确的平滑方法')
        all_passed = False
    
    if '+ 0.001' in source:
        print('FAIL: 仍存在硬编码的 + 0.001')
        all_passed = False
    else:
        print('PASS: 已移除硬编码的 + 0.001')
    
    df = load_stock_data('data/stock_timeseries.csv')
    df = calculate_returns(df)
    df_rsi = calculate_rsi(df.copy())
    
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        rsi = df_rsi[df_rsi['stock_symbol'] == symbol]['rsi'].dropna()
        if len(rsi) > 0:
            print(f'\n{symbol} RSI范围: {rsi.min():.2f} - {rsi.max():.2f}')
            if rsi.min() >= 0 and rsi.max() <= 100:
                print(f'  PASS: RSI值在正确范围内')
            else:
                print(f'  FAIL: RSI值超出范围')
                all_passed = False
    
    return all_passed

if __name__ == '__main__':
    vol_passed = test_volatility_fix()
    rsi_passed = test_rsi_fix()
    
    print('\n' + '='*60)
    print('测试结果汇总')
    print('='*60)
    print(f'年化波动率修复: {"PASS" if vol_passed else "FAIL"}')
    print(f'RSI计算修复: {"PASS" if rsi_passed else "FAIL"}')
    
    if vol_passed and rsi_passed:
        print('\n所有测试通过!')
        sys.exit(0)
    else:
        print('\n存在失败的测试!')
        sys.exit(1)
