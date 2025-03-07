"""
Data Quality Testing and Repair Script

This script checks the quality of cached financial data and attempts to repair issues.
Run this if you're experiencing NaN values or other data quality problems.
"""

import os
import pandas as pd
import numpy as np
from cache_manager import DataCache
import argparse
import yfinance as yf
from datetime import datetime, timedelta

def test_cached_data(cache_dir: str = "data_cache", verbose: bool = False, repair: bool = False):
    """Test the quality of cached data and optionally repair it"""
    cache = DataCache(cache_dir)
    
    # Get cache status
    status = cache.get_cache_status()
    
    print("\n===== Data Cache Quality Report =====")
    print(f"Cache created: {status.get('cache_created', 'Unknown')}")
    print(f"Date range: {status.get('oldest_data', 'Unknown')} to {status.get('newest_data', 'Unknown')}")
    print(f"Price tickers: {status.get('price_ticker_count', 0)}")
    print(f"Cache size: {status.get('price_cache_size_mb', 0):.2f} MB")
    
    # Load price data cache
    try:
        with open(cache.price_cache_file, 'rb') as f:
            import pickle
            price_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
        return
    
    print(f"\nAnalyzing {len(price_data)} tickers...")
    
    # Check each ticker's data quality
    good_tickers = []
    bad_tickers = []
    repaired_tickers = []
    
    for ticker, prices in price_data.items():
        nan_count = prices.isna().sum()
        nan_pct = (nan_count / len(prices)) * 100
        
        if verbose:
            print(f"\n{ticker}:")
            print(f"  Data points: {len(prices)}")
            print(f"  NaN values: {nan_count} ({nan_pct:.2f}%)")
            if not prices.empty:
                valid_prices = prices.dropna()
                if not valid_prices.empty:
                    print(f"  Date range: {valid_prices.index[0].date()} to {valid_prices.index[-1].date()}")
                    print(f"  Price range: {valid_prices.min():.2f} to {valid_prices.max():.2f}")
        
        if nan_pct == 0:
            good_tickers.append(ticker)
        else:
            if nan_pct > 50:  # More than 50% NaN
                bad_tickers.append((ticker, nan_pct))
                if repair:
                    repaired_tickers.append(ticker)
                    # Try to fetch fresh data
                    try:
                        print(f"Repairing {ticker} data...")
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365*5)  # 5 years
                        
                        stock = yf.Ticker(ticker)
                        new_prices = stock.history(start=start_date, end=end_date)['Close']
                        
                        if len(new_prices) > 10:  # We got some data
                            new_prices.index = new_prices.index.tz_localize(None)
                            cache.save_price_data(ticker, new_prices)
                            print(f"✅ Successfully repaired {ticker} with {len(new_prices)} data points")
                        else:
                            print(f"❌ Failed to repair {ticker} - insufficient data")
                    except Exception as e:
                        print(f"❌ Error repairing {ticker}: {e}")
            else:
                if repair:
                    # Just clean up the existing data by filling NaN values
                    clean_prices = prices.ffill().bfill()
                    cache.save_price_data(ticker, clean_prices)
                    repaired_tickers.append(ticker)
                    print(f"✅ Cleaned {ticker} data by filling {nan_count} NaN values")
    
    print("\n===== Summary =====")
    print(f"Total tickers: {len(price_data)}")
    print(f"Good data: {len(good_tickers)} tickers")
    print(f"Problematic data: {len(bad_tickers)} tickers")
    
    if bad_tickers:
        print("\nTop 5 worst tickers:")
        for ticker, nan_pct in sorted(bad_tickers, key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {ticker}: {nan_pct:.1f}% NaN values")
    
    if repair:
        print(f"\nRepaired {len(repaired_tickers)} tickers")

def main():
    parser = argparse.ArgumentParser(description="Test and repair data quality in cache")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--repair", "-r", action="store_true", help="Attempt to repair bad data")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Cache directory")
    args = parser.parse_args()
    
    test_cached_data(args.cache_dir, args.verbose, args.repair)

if __name__ == "__main__":
    main()
