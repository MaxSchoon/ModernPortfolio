"""
Financial Data Cache Tools - Lightweight Utilities

This module provides essential tools for working with the CSV cache:
- Quick cache inspection
- Simple data validation
- Common data quality fixes

For more advanced maintenance, use cache_maintenance.py
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import sys
from typing import List, Dict, Any, Optional

from csv_cache_manager import CSVDataCache
from utils import (
    validate_ticker, format_ticker, 
    print_info, print_success, print_warning, print_error
)

# Default cache directory - consistent with ModernPortfolio.py
DEFAULT_CACHE_DIR = "data_cache"

def list_cached_tickers(cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """List all tickers in the cache"""
    cache = CSVDataCache(cache_dir)
    price_dir = os.path.join(cache_dir, "prices")
    
    if not os.path.exists(price_dir):
        print_error(f"Cache directory not found: {price_dir}")
        return
    
    files = glob.glob(os.path.join(price_dir, "*_price.csv"))
    tickers = [os.path.basename(f).replace("_price.csv", "") for f in files]
    
    print_success(f"Found {len(tickers)} tickers in cache:")
    
    # Sort and display tickers in columns
    tickers.sort()
    cols = 4
    for i in range(0, len(tickers), cols):
        row = tickers[i:i+cols]
        print("  ".join(ticker.ljust(10) for ticker in row))
    
    # Show cache status
    status = cache.get_cache_status()
    print_info("Cache Status:")
    print(f"- Price data: {status.get('price_ticker_count', 0)} tickers ({status.get('price_cache_size_mb', 0):.2f} MB)")
    print(f"- Dividend data: {status.get('div_ticker_count', 0)} tickers ({status.get('div_cache_size_mb', 0):.2f} MB)")
    print(f"- Date range: {status.get('oldest_data', 'N/A')} to {status.get('newest_data', 'N/A')}")

def inspect_ticker_data(ticker: str, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Inspect detailed data for a specific ticker"""
    cache = CSVDataCache(cache_dir)
    
    # Get price data
    price_data = cache.get_price_data(ticker)
    div_data = cache.get_div_data(ticker)
    
    if price_data is None:
        print_error(f"No price data found for {ticker}")
        return
    
    print_info(f"=== {ticker} Data Inspection ===")
    
    # Basic statistics
    print_info("Price data:")
    print(f"- Data points: {len(price_data)}")
    print(f"- Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    print(f"- Price range: ${price_data.min():.2f} to ${price_data.max():.2f}")
    print(f"- Current price: ${price_data.iloc[-1]:.2f}")
    
    # Check for NaN values
    nan_count = price_data.isna().sum()
    nan_pct = (nan_count / len(price_data)) * 100
    if nan_count > 0:
        print_warning(f"NaN values: {nan_count} ({nan_pct:.2f}%)")
    else:
        print(f"- NaN values: {nan_count} (0.00%)")
    
    # Dividend information
    if div_data is not None and not div_data.empty:
        div_payments = div_data[div_data > 0]
        if len(div_payments) > 0:
            print_info("Dividend data:")
            print(f"- Dividend payments: {len(div_payments)}")
            print(f"- Average dividend: ${div_payments.mean():.4f}")
            print(f"- Last dividend: ${div_payments.iloc[-1] if len(div_payments) > 0 else 0:.4f}")
            
            # Calculate yield
            annual_div = div_data.resample('Y').sum().mean()
            current_price = price_data.iloc[-1]
            div_yield = (annual_div / current_price) * 100
            print(f"- Estimated annual yield: {div_yield:.2f}%")
        else:
            print("\nNo dividend payments found")

def fix_nan_issues(ticker: str, cache_dir: str = DEFAULT_CACHE_DIR) -> bool:
    """
    Fix NaN issues in a ticker's price data
    
    Parameters:
        ticker: The ticker to fix
        cache_dir: Cache directory
        
    Returns:
        True if fixed successfully, False otherwise
    """
    cache = CSVDataCache(cache_dir)
    
    try:
        # Get price data
        price_data = cache.get_price_data(ticker)
        
        if price_data is None:
            print_error(f"No price data found for {ticker}")
            return False
        
        # Check for NaN values
        nan_count = price_data.isna().sum()
        
        if nan_count == 0:
            print_success(f"No NaN values found in {ticker} price data")
            return True
        
        # Fix NaN values with forward/backward fill
        clean_prices = price_data.ffill().bfill()
        
        # Save back to cache
        cache.save_price_data(ticker, clean_prices)
        
        print_success(f"Fixed {nan_count} NaN values in {ticker} price data")
        return True
        
    except Exception as e:
        print_error(f"Error fixing {ticker}: {str(e)}")
        return False

def validate_cached_ticker(ticker: str, cache_dir: str = DEFAULT_CACHE_DIR, verbose: bool = True) -> bool:
    """
    Validate a cached ticker's data
    
    Parameters:
        ticker: The ticker to validate
        cache_dir: Cache directory
        verbose: Whether to print details
        
    Returns:
        True if valid, False otherwise
    """
    cache = CSVDataCache(cache_dir)
    
    try:
        # Get price data
        price_data = cache.get_price_data(ticker)
        
        if price_data is None:
            if verbose:
                print_error(f"No price data found for {ticker}")
            return False
        
        # Check for minimum data points
        if len(price_data) < 100:
            if verbose:
                print_error(f"{ticker}: Insufficient data points ({len(price_data)} < 100)")
            return False
        
        # Check for NaN values
        nan_count = price_data.isna().sum()
        nan_pct = (nan_count / len(price_data)) * 100
        
        if nan_pct > 10:
            if verbose:
                print_error(f"{ticker}: Too many NaN values ({nan_pct:.1f}%)")
            return False
        
        # Check date range
        date_range = (price_data.index[-1] - price_data.index[0]).days
        if date_range < 30:
            if verbose:
                print_error(f"{ticker}: Date range too small ({date_range} days)")
            return False
        
        if verbose:
            print_success(f"{ticker}: Valid with {len(price_data)} data points over {date_range} days")
        
        return True
        
    except Exception as e:
        if verbose:
            print_error(f"Error validating {ticker}: {str(e)}")
        return False

if __name__ == "__main__":
    # Simple CLI for basic operations
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic cache tools")
    parser.add_argument('command', choices=['list', 'inspect', 'fix', 'validate'],
                        help="Command to execute")
    parser.add_argument('--ticker', '-t', help="Ticker symbol")
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help="Cache directory")
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_cached_tickers(args.cache_dir)
    elif args.command == 'inspect':
        if not args.ticker:
            print_error("Ticker required for inspect command")
            sys.exit(1)
        inspect_ticker_data(args.ticker, args.cache_dir)
    elif args.command == 'fix':
        if not args.ticker:
            print_error("Ticker required for fix command")
            sys.exit(1)
        fix_nan_issues(args.ticker, args.cache_dir)
    elif args.command == 'validate':
        if not args.ticker:
            print_error("Ticker required for validate command")
            sys.exit(1)
        validate_cached_ticker(args.ticker, args.cache_dir)