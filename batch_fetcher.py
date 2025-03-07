"""
Batch Data Fetcher for Yahoo Finance

This script fetches Yahoo Finance data in small batches with longer delays
between requests to avoid rate limiting.
"""

import yfinance as yf
import pandas as pd
import time
import os
import argparse
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from csv_cache_manager import CSVDataCache
from utils import load_tickers, format_ticker  # Import utility functions

class BatchFetcher:
    """Fetches financial data in batches with rate limiting to avoid API restrictions"""
    
    def __init__(self, cache_dir: str = "csv_data_cache", batch_size: int = 3, 
                 delay_min: float = 2.0, delay_max: float = 5.0,
                 retry_count: int = 3, years: int = 5):
        """
        Initialize the batch fetcher
        
        Parameters:
            cache_dir: Directory to store cached data
            batch_size: Number of tickers to fetch in each batch
            delay_min: Minimum delay between batches in seconds
            delay_max: Maximum delay between batches in seconds
            retry_count: Number of times to retry failed requests
            years: Number of years of historical data to fetch
        """
        self.cache = CSVDataCache(cache_dir)
        self.batch_size = batch_size
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.retry_count = retry_count
        self.years = years
        
        # Stats
        self.success_count = 0
        self.fail_count = 0
        self.cached_count = 0
    
    def fetch_ticker(self, ticker: str, use_cache: bool = True) -> Tuple[bool, str]:
        """
        Fetch data for a single ticker with retries
        
        Returns:
            Tuple of (success, message)
        """
        if ticker in ['CASH', 'TBILLS']:
            return True, f"✅ {ticker}: Special asset, no need to fetch data"
        
        # Check cache first if enabled
        if use_cache:
            cached_data = self.cache.get_price_data(ticker)
            if cached_data is not None:
                # Extra check for data quality - ensure we're not saving NaN values
                nan_pct = cached_data.isna().mean() * 100
                if nan_pct > 50:  # If more than 50% is NaN, consider it bad data
                    print(f"⚠️ {ticker}: Cached data has {nan_pct:.1f}% NaN values - will fetch fresh data")
                else:
                    self.cached_count += 1
                    return True, f"✅ {ticker}: Using cached data ({len(cached_data)} points, {nan_pct:.1f}% NaN)"
        
        formatted_ticker = format_ticker(ticker)  # Use the utility function
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*self.years)
        
        for attempt in range(self.retry_count):
            try:
                print(f"Fetching {ticker} (attempt {attempt+1}/{self.retry_count})...")
                
                stock = yf.Ticker(formatted_ticker)
                
                # Try to get basic info first - lightweight check
                try:
                    info = stock.info
                    if 'regularMarketPrice' not in info and 'currentPrice' not in info:
                        if attempt < self.retry_count - 1:
                            print(f"⚠️ No market data found for {ticker}, retrying...")
                            time.sleep(random.uniform(1, 3))
                            continue
                        return False, f"❌ {ticker}: No market data available"
                        
                    # Save info to cache
                    self.cache.save_info_data(ticker, info)
                    
                except Exception as e:
                    print(f"⚠️ Error getting info for {ticker}: {str(e)}")
                    # Continue anyway - info isn't critical
                
                # Get historical prices
                prices = stock.history(start=start_date, end=end_date)['Close']
                
                if len(prices) < 10:
                    if attempt < self.retry_count - 1:
                        print(f"⚠️ Insufficient data for {ticker}, retrying...")
                        time.sleep(random.uniform(1, 3))
                        continue
                    return False, f"❌ {ticker}: Insufficient price data ({len(prices)} days)"
                
                # Add explicit data quality check before saving to cache
                if len(prices) > 0:
                    # Check for NaN values
                    nan_pct = prices.isna().mean() * 100
                    if nan_pct > 80:  # Too many NaN values
                        return False, f"❌ {ticker}: Data quality too low ({nan_pct:.1f}% NaN values)"
                        
                    # Fill NaN values before saving
                    prices = prices.ffill().bfill()
                    
                    # Save clean data to cache
                    self.cache.save_price_data(ticker, prices)
                
                # Get dividends
                try:
                    dividends = stock.dividends
                    
                    # Normalize data
                    prices.index = prices.index.tz_localize(None)
                    
                    # Create dividend series
                    div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                    if not dividends.empty:
                        dividends.index = dividends.index.tz_localize(None)
                        common_dates = dividends.index.intersection(div_series.index)
                        if not common_dates.empty:
                            div_series.loc[common_dates] = dividends[dividends.index.isin(common_dates)].astype('float64')
                    
                    # Save to cache
                    self.cache.save_price_data(ticker, prices)
                    self.cache.save_div_data(ticker, div_series)
                    
                    self.success_count += 1
                    return True, f"✅ {ticker}: Found {len(prices)} days of data and {len(dividends)} dividends"
                    
                except Exception as e:
                    print(f"⚠️ Error processing dividends for {ticker}: {str(e)}")
                    # Continue with just price data
                    
                    # Normalize and save price data
                    prices.index = prices.index.tz_localize(None)
                    self.cache.save_price_data(ticker, prices)
                    
                    # Create empty dividend series
                    div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                    self.cache.save_div_data(ticker, div_series)
                    
                    self.success_count += 1
                    return True, f"✅ {ticker}: Found {len(prices)} days of data (no dividends)"
                    
            except Exception as e:
                delay = random.uniform(2, 6) * (attempt + 1)  # Increasing delay with each attempt
                if attempt < self.retry_count - 1:
                    print(f"⚠️ Failed to fetch {ticker} (attempt {attempt+1}): {str(e)}")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    self.fail_count += 1
                    return False, f"❌ {ticker}: Failed after {self.retry_count} attempts - {str(e)}"
                    
        return False, f"❌ {ticker}: Failed after {self.retry_count} attempts"
    
    def fetch_batch(self, tickers: List[str], use_cache: bool = True) -> Dict[str, str]:
        """
        Fetch data for a batch of tickers
        
        Returns:
            Dictionary of {ticker: status_message}
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            success, message = self.fetch_ticker(ticker, use_cache)
            results[ticker] = message
            
            # Add delay between requests (but not after the last one)
            if i < len(tickers) - 1:
                delay = random.uniform(self.delay_min, self.delay_max)
                print(f"Waiting {delay:.1f} seconds before next request...")
                time.sleep(delay)
                
        return results
    
    def fetch_all(self, tickers: List[str], use_cache: bool = True) -> Dict[str, str]:
        """
        Fetch all tickers in batches
        
        Returns:
            Dictionary of {ticker: status_message}
        """
        print(f"\n===== Fetching {len(tickers)} tickers in batches of {self.batch_size} =====")
        print(f"Using cache: {use_cache}")
        
        all_results = {}
        batch_count = (len(tickers) + self.batch_size - 1) // self.batch_size
        
        # Reset stats
        self.success_count = 0
        self.fail_count = 0
        self.cached_count = 0
        
        start_time = time.time()
        
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            
            print(f"\n----- Batch {batch_num}/{batch_count}: {', '.join(batch)} -----")
            
            # Process batch
            results = self.fetch_batch(batch, use_cache)
            all_results.update(results)
            
            # Add longer delay between batches
            if batch_num < batch_count:
                batch_delay = random.uniform(self.delay_max, self.delay_max * 2)
                print(f"\nBatch complete. Waiting {batch_delay:.1f} seconds before next batch...")
                time.sleep(batch_delay)
                
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("\n===== Fetch Summary =====")
        print(f"Completed in {elapsed_time:.1f} seconds")
        print(f"Tickers processed: {len(tickers)}")
        print(f"Successfully fetched: {self.success_count}")
        print(f"Used from cache: {self.cached_count}")
        print(f"Failed: {self.fail_count}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Fetch Yahoo Finance data in batches to avoid rate limiting')
    parser.add_argument('--tickers', help='Comma-separated list of tickers to fetch', type=str)
    parser.add_argument('--file', help='CSV file containing tickers', type=str, default="tickers.csv")
    parser.add_argument('--batch-size', help='Batch size for fetching', type=int, default=3)
    parser.add_argument('--delay-min', help='Minimum delay between requests (seconds)', type=float, default=2.0)
    parser.add_argument('--delay-max', help='Maximum delay between requests (seconds)', type=float, default=5.0)
    parser.add_argument('--retry', help='Number of retry attempts', type=int, default=3)
    parser.add_argument('--years', help='Years of data to fetch', type=int, default=5)
    parser.add_argument('--no-cache', help='Disable cache', action='store_true')
    parser.add_argument('--cache-dir', help='Cache directory', type=str, default="data_cache")
    parser.add_argument('--clear-cache', help='Clear cache before starting', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = BatchFetcher(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        retry_count=args.retry,
        years=args.years
    )
    
    # Show cache status
    cache_status = fetcher.cache.get_cache_status()
    print("\nCache Status:")
    print(f"- Cache created: {cache_status.get('cache_created', 'N/A')}")
    print(f"- Price data: {cache_status.get('price_ticker_count', 0)} tickers ({cache_status.get('price_cache_size_mb', 0)} MB)")
    print(f"- Data range: {cache_status.get('oldest_data', 'N/A')} to {cache_status.get('newest_data', 'N/A')}")
    
    # Clear cache if requested
    if args.clear_cache:
        print("\nClearing cache...")
        fetcher.cache.clear_cache()
    
    # Get tickers
    tickers = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        file_path = args.file
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)
        tickers = load_tickers(file_path)  # Use the imported function
    
    # Fetch data
    results = fetcher.fetch_all(tickers, not args.no_cache)
    
    # Show results
    print("\n===== Results =====")
    for ticker, status in results.items():
        print(f"{ticker}: {status}")

if __name__ == "__main__":
    main()
