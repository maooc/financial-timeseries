import pandas as pd
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.analyzer import load_stock_data, perform_comprehensive_analysis
from src.visualizer import generate_all_financial_charts

def generate_report(analysis_results, output_file='output/financial_report.txt'):
    os.makedirs('output', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Financial Time Series Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        performance = analysis_results['performance']
        f.write("STOCK PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        for stock, perf in performance.items():
            f.write(f"\n{stock}:\n")
            f.write(f"  Total Return: {perf['total_return']:.2f}%\n")
            f.write(f"  Volatility: {perf['volatility']:.2f}%\n")
            f.write(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Daily Return: {perf['max_daily_return']:.2f}%\n")
            f.write(f"  Min Daily Return: {perf['min_daily_return']:.2f}%\n")
            f.write(f"  Win Rate: {perf['win_rate']:.1f}%\n")
        
        f.write("\n\nCORRELATION MATRIX\n")
        f.write("-" * 40 + "\n")
        corr = analysis_results['correlations']
        f.write(corr.to_string())
        
        f.write("\n\nTREND ANALYSIS\n")
        f.write("-" * 40 + "\n")
        trends = analysis_results['trends']
        for stock, trend in trends.items():
            f.write(f"\n{stock}:\n")
            f.write(f"  Trend: {trend['trend']}\n")
            f.write(f"  Slope: {trend['slope']:.4f}\n")
            f.write(f"  R-squared: {trend['r_squared']:.4f}\n")
        
        f.write("\n\nBETA VALUES\n")
        f.write("-" * 40 + "\n")
        betas = analysis_results['betas']
        for stock, beta in betas.items():
            f.write(f"  {stock}: {beta:.4f}\n")
    
    print(f"Report saved to {output_file}")

def main():
    print("Financial Time Series Analysis System v1.0")
    print("=" * 40)
    
    os.makedirs('output', exist_ok=True)
    
    df = load_stock_data('data/stock_timeseries.csv')
    print(f"Loaded {len(df)} records")
    print(f"Stocks: {df['stock_symbol'].unique().tolist()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print("\nPerforming analysis...")
    results = perform_comprehensive_analysis(df)
    
    print("\nStock Performance:")
    for stock, perf in results['performance'].items():
        print(f"  {stock}: Return={perf['total_return']:.2f}%, Volatility={perf['volatility']:.2f}%")
    
    print("\nCorrelation Matrix:")
    print(results['correlations'].round(2))
    
    print("\nBeta Values:")
    for stock, beta in results['betas'].items():
        print(f"  {stock}: {beta:.4f}")
    
    print("\nGenerating charts...")
    generate_all_financial_charts(results)
    
    print("\nGenerating report...")
    generate_report(results)
    
    print("\nAnalysis complete!")
    print("Output files saved to 'output/' directory")

if __name__ == '__main__':
    main()
