"""
Financial Data Fetcher

This module combines ticker validation, data fetching, and cache management
to provide a unified interface for obtaining financial data.
"""

import yfinance as yf
import pandas as pd
import os
import argparse
import time
import sys
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from tqdm import tqdm  # For progress bars
import time
import itertools
import colorama
from colorama import Fore, Style

from src.cache.csv_cache_manager import CSVDataCache
from src.utils.utils import (
    load_tickers, 
    format_ticker, 
    validate_ticker, 
    log_progress,
    validate_prices
)

# Initialize colorama for colored terminal output
colorama.init()

class DataFetcher:
    """
    Fetches financial data with built-in caching, validation, and rate limiting
    """
    
    def __init__(self, cache_dir: str = "data_cache", 
                 batch_size: int = 50,  # Increased from 5 to 50
                 delay_between_batches: float = 5.0,  # Simplified to constant 5 seconds
                 retry_count: int = 3, 
                 years: int = 5,
                 max_workers: int = 10,
                 throttle_calls: int = 20,  # Increased to match larger batch size
                 throttle_period: int = 20):  # Reduced from 60 to 20 seconds
        """
        Initialize the data fetcher
        
        Parameters:
            cache_dir: Directory to store cached data
            batch_size: Number of tickers to fetch in each batch
            delay_between_batches: Delay between batches in seconds (fixed at 5 seconds)
            retry_count: Number of times to retry failed requests
            years: Number of years of historical data to fetch
            max_workers: Maximum number of parallel workers for fetching
            throttle_calls: Maximum API calls in throttle period
            throttle_period: Period in seconds for throttling (default: 20 seconds)
        """
        self.cache = CSVDataCache(cache_dir)
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.retry_count = retry_count
        self.years = years
        self.max_workers = max_workers
        self.throttle_calls = throttle_calls
        self.throttle_period = throttle_period
        
        # Stats
        self.success_count = 0
        self.fail_count = 0
        self.cached_count = 0
        
        # Throttling queue
        self.api_call_times = []
        
        # Add a flag to prevent multiple throttle messages
        self.currently_throttling = False
    
    def _throttle_api_calls(self):
        """Throttle API calls to avoid rate limiting"""
        # Skip if already throttling
        if self.currently_throttling:
            return
            
        current_time = time.time()
        
        # Remove old timestamps outside the throttle window
        self.api_call_times = [t for t in self.api_call_times 
                             if current_time - t < self.throttle_period]
        
        # Check if we need to throttle
        if len(self.api_call_times) >= self.throttle_calls:
            # Calculate sleep time to stay within limits
            oldest_call = min(self.api_call_times)
            sleep_time = self.throttle_period - (current_time - oldest_call) + 1
            
            # Cap maximum wait time to 10 seconds
            sleep_time = min(sleep_time, 10.0)
            
            if sleep_time > 0:
                try:
                    self.currently_throttling = True
                    self._animated_sleep(sleep_time, "Throttling API calls")
                finally:
                    self.currently_throttling = False
        
        # Add current timestamp
        self.api_call_times.append(time.time())
    
    def _animated_sleep(self, seconds: float, message: str = "Waiting"):
        """Display an animated spinner while sleeping on a single line"""
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        end_time = time.time() + seconds
        start_time = time.time()
        total_seconds = seconds
        
        try:
            while time.time() < end_time:
                elapsed = time.time() - start_time
                remaining = max(0, end_time - time.time())
                progress = min(1.0, elapsed / total_seconds)
                
                # Create a progress bar
                bar_length = 20
                block = int(round(bar_length * progress))
                progress_bar = f"[{Fore.GREEN}{'█' * block}{Fore.YELLOW}{'-' * (bar_length - block)}{Style.RESET_ALL}]"
                
                # Format the status line
                percentage = int(progress * 100)
                status = f"{Fore.CYAN}{message}{Style.RESET_ALL} {progress_bar} {percentage}% {Fore.BLUE}{next(spinner)}{Style.RESET_ALL} {remaining:.1f}s remaining"
                
                # Clear the line and write the new status
                sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the line
                sys.stdout.write(status)
                sys.stdout.flush()
                time.sleep(0.1)
            
            # Show completion status
            progress_bar = f"[{Fore.GREEN}{'█' * bar_length}{Style.RESET_ALL}]"
            completion = f"\r{Fore.CYAN}{message}{Style.RESET_ALL} {progress_bar} {Fore.GREEN}100% ✓ Complete{Style.RESET_ALL}{' ' * 20}"
            sys.stdout.write(completion)
            sys.stdout.flush()
            print()  # Move to the next line after completion
            
        except KeyboardInterrupt:
            sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the line
            sys.stdout.write(f"{Fore.RED}Wait interrupted!{Style.RESET_ALL}")
            sys.stdout.flush()
            print()
            raise
    
    def fetch_ticker(self, ticker: str, use_cache: bool = True) -> Tuple[bool, str]:
        """
        Fetch data for a single ticker with validation and retries
        
        Parameters:
            ticker: The ticker symbol to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            Tuple of (success, message)
        """
        # First validate the ticker
        is_valid, corrected_ticker, _ = validate_ticker(ticker, verbose=False)
        
        if not is_valid:
            return False, f"❌ {ticker} is not valid"
        
        # If the ticker was corrected, update ticker
        if ticker != corrected_ticker:
            ticker = corrected_ticker
        
        # Apply additional formatting if needed
        formatted_ticker = format_ticker(ticker)
        if formatted_ticker != ticker:
            ticker = formatted_ticker
        
        # Handle special assets
        if ticker in ['CASH', 'TBILLS']:
            try:
                # Generate synthetic data
                synthetic_data = self.cache._get_synthetic_asset_data(ticker)
                return True, f"✅ {ticker}: Special asset, using synthetic data ({len(synthetic_data)} points)"
            except Exception as e:
                return False, f"❌ {ticker}: Error generating synthetic data - {str(e)}"
        
        # Check cache first if enabled
        if use_cache:
            cached_data = self.cache.get_price_data(ticker)
            if cached_data is not None and len(cached_data) > 100:
                # Extra check for data quality - ensure we're not using bad cached data
                nan_pct = cached_data.isna().mean() * 100
                if nan_pct > 50:  # If more than 50% is NaN, consider it bad data
                    print(f"{Fore.YELLOW}⚠️ {ticker}: Cached data has {nan_pct:.1f}% NaN values - will fetch fresh data{Style.RESET_ALL}")
                else:
                    self.cached_count += 1
                    return True, f"✅ {ticker}: Using cached data ({len(cached_data)} points, {nan_pct:.1f}% NaN)"
        
        # Apply rate throttling
        self._throttle_api_calls()
        
        # Fetch from Yahoo Finance with retries
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*self.years)
        
        for attempt in range(self.retry_count):
            try:
                stock = yf.Ticker(ticker)
                
                # Try to get basic info first - lightweight check
                try:
                    info = stock.info
                    if 'regularMarketPrice' not in info and 'currentPrice' not in info:
                        if attempt < self.retry_count - 1:
                            delay = 2  # Simple retry delay
                            time.sleep(delay)
                            continue
                        return False, f"❌ {ticker}: No market data available"
                    
                    # Save info to cache
                    self.cache.save_info_data(ticker, info)
                    
                except Exception as e:
                    # Continue anyway - info isn't critical
                    pass
                
                # Get price data
                prices = stock.history(start=start_date, end=end_date)['Close']
                
                if len(prices) < 100:
                    if attempt < self.retry_count - 1:
                        delay = 2  # Simple retry delay
                        time.sleep(delay)
                        continue
                    return False, f"❌ {ticker}: Insufficient price data ({len(prices)} days)"
                
                # Normalize and clean data
                prices.index = prices.index.tz_localize(None)
                
                # Check data quality
                validation = validate_prices(prices)
                if not validation['valid']:
                    if attempt < self.retry_count - 1:
                        delay = 2  # Simple retry delay
                        time.sleep(delay)
                        continue
                    return False, f"❌ {ticker}: {validation['error']}"
                
                # Save price data
                self.cache.save_price_data(ticker, prices)
                
                # Get dividend data
                try:
                    dividends = stock.dividends
                    if not dividends.empty:
                        dividends.index = dividends.index.tz_localize(None)
                        div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                        common_dates = dividends.index.intersection(div_series.index)
                        
                        if common_dates.size > 0:
                            div_series.loc[common_dates] = dividends[dividends.index.isin(common_dates)].astype('float64')
                        self.cache.save_div_data(ticker, div_series)
                        
                        self.success_count += 1
                        return True, f"✅ {ticker}: Fetched {len(prices)} price points and {len(dividends)} dividends"
                    else:
                        div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                        self.cache.save_div_data(ticker, div_series)
                        
                        self.success_count += 1
                        return True, f"✅ {ticker}: Fetched {len(prices)} price points (no dividends)"
                
                except Exception as e:
                    # If dividend fetching fails, we can still use the price data
                    div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                    self.cache.save_div_data(ticker, div_series)
                    
                    self.success_count += 1
                    return True, f"✅ {ticker}: Fetched {len(prices)} price points (dividend error: {str(e)})"
            
            except Exception as e:
                if attempt < self.retry_count - 1:
                    delay = 2  # Simple retry delay
                    time.sleep(delay)
                else:
                    self.fail_count += 1
                    return False, f"❌ {ticker}: Failed after {self.retry_count} attempts - {str(e)}"
        
        return False, f"❌ {ticker}: Failed after {self.retry_count} attempts"
    
    def fetch_batch(self, tickers: List[str], use_cache: bool = True) -> Dict[str, str]:
        """
        Fetch data for a batch of tickers
        
        Parameters:
            tickers: List of ticker symbols to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary of {ticker: status_message}
        """
        results = {}
        
        # First check cache for all tickers to avoid unnecessary API calls
        if use_cache:
            for ticker in tickers:
                # Check if special asset or in cache
                if ticker in ['CASH', 'TBILLS']:
                    success, message = self.fetch_ticker(ticker, use_cache=True)
                    results[ticker] = message
                    continue
                    
                cached_data = self.cache.get_price_data(ticker)
                if cached_data is not None and len(cached_data) > 100:
                    # Check quality
                    nan_pct = cached_data.isna().mean() * 100
                    if nan_pct <= 50:
                        self.cached_count += 1
                        results[ticker] = f"✅ {ticker}: Using cached data ({len(cached_data)} points, {nan_pct:.1f}% NaN)"
        
        # Identify tickers that need fetching
        tickers_to_fetch = [t for t in tickers if t not in results]
        
        # Use parallel fetching for remaining tickers
        if tickers_to_fetch:
            print(f"\n{Fore.CYAN}Fetching {len(tickers_to_fetch)} tickers in parallel with {self.max_workers} workers...{Style.RESET_ALL}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a dictionary mapping futures to tickers
                future_to_ticker = {
                    executor.submit(self.fetch_ticker, ticker, False): ticker 
                    for ticker in tickers_to_fetch
                }
                
                # Process completed futures with a progress bar
                with tqdm(total=len(tickers_to_fetch), desc="Fetching", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as progress_bar:
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        try:
                            success, message = future.result()
                            results[ticker] = message
                            
                            # Update progress bar with color based on success
                            if success:
                                desc = f"{Fore.GREEN}✓ {ticker}{Style.RESET_ALL}"
                            else:
                                desc = f"{Fore.RED}✗ {ticker}{Style.RESET_ALL}"
                            progress_bar.set_description(desc)
                            progress_bar.update(1)
                            
                        except Exception as e:
                            results[ticker] = f"❌ {ticker}: Error in thread: {str(e)}"
                            progress_bar.update(1)
        
        return results
    
    def fetch_all(self, tickers: List[str], use_cache: bool = True) -> Dict[str, str]:
        """
        Fetch all tickers in batches
        
        Parameters:
            tickers: List of ticker symbols to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary of {ticker: status_message}
        """
        print(f"\n{Fore.GREEN}{'='*20} FINANCIAL DATA FETCHER {'='*20}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Fetching {len(tickers)} tickers in batches of {self.batch_size}{Style.RESET_ALL}")
        print(f"Cache: {'Enabled' if use_cache else 'Disabled'}, Workers: {self.max_workers}, Years: {self.years}")
        
        all_results = {}
        batch_count = (len(tickers) + self.batch_size - 1) // self.batch_size
        
        # Reset stats
        self.success_count = 0
        self.fail_count = 0
        self.cached_count = 0
        
        start_time = time.time()
        
        # Create progress bar for the entire process
        with tqdm(total=len(tickers), desc="Overall Progress", 
                 position=0, leave=True,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as overall_progress:
            
            # Process batches
            for i in range(0, len(tickers), self.batch_size):
                batch = tickers[i:i+self.batch_size]
                batch_num = i // self.batch_size + 1
                
                print(f"\n{Fore.BLUE}Batch {batch_num}/{batch_count}: Processing {len(batch)} tickers{Style.RESET_ALL}")
                
                # Process batch
                batch_start = time.time()
                results = self.fetch_batch(batch, use_cache)
                all_results.update(results)
                batch_time = time.time() - batch_start
                
                # Update overall progress
                overall_progress.update(len(batch))
                
                # Print batch summary
                success_count = sum(1 for msg in results.values() if msg.startswith("✅"))
                fail_count = len(batch) - success_count
                print(f"Batch {batch_num} complete: {success_count} succeeded, {fail_count} failed, took {batch_time:.1f}s")
                
                # Add delay between batches (except after the last one)
                if batch_num < batch_count:
                    self._animated_sleep(self.delay_between_batches, f"Waiting before batch {batch_num+1}")
                
        elapsed_time = time.time() - start_time
        
        # Print colorful summary
        print(f"\n{Fore.GREEN}{'='*20} FETCH SUMMARY {'='*20}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Completed in {elapsed_time:.1f} seconds{Style.RESET_ALL}")
        print(f"Tickers processed: {len(tickers)}")
        print(f"{Fore.GREEN}Successfully fetched: {self.success_count}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Used from cache: {self.cached_count}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.fail_count}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
        
        return all_results
    
    def process_csv_file(self, filepath: str, output_filepath: str = None, 
                      use_cache: bool = True) -> None:
        """
        Process a CSV file containing tickers
        
        Parameters:
            filepath: Path to the input CSV file
            output_filepath: Path to save corrected CSV (if None, will use "tickers_corrected.csv")
            use_cache: Whether to use cached data if available
        """
        if not os.path.exists(filepath):
            print(f"{Fore.RED}❌ File not found: {filepath}{Style.RESET_ALL}")
            return
        
        if output_filepath is None:
            output_filepath = "tickers_corrected.csv"
        
        # Read the CSV file
        try:
            print(f"{Fore.CYAN}Reading tickers from {filepath}...{Style.RESET_ALL}")
            tickers = load_tickers(filepath)
            df = None
            
            # Try to detect delimiter
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    delimiter = ';'
                elif ',' in first_line:
                    delimiter = ','
                else:
                    delimiter = ';'  # Default
            
            # Read the original DataFrame to preserve other columns
            df = pd.read_csv(filepath, sep=delimiter)
            
            # Find the ticker column
            ticker_column = None
            for col in df.columns:
                if col.lower() == 'ticker':
                    ticker_column = col
                    break
                    
            if ticker_column is None:
                print(f"{Fore.RED}❌ CSV columns: {', '.join(df.columns)}")
                print("❌ CSV file must contain a 'ticker' column{Style.RESET_ALL}")
                return
            
            # Add status column if it doesn't exist
            status_column = 'status'
            if status_column not in df.columns:
                df[status_column] = 'Unknown'
            
            # Process all tickers
            results = self.fetch_all(tickers, use_cache)
            
            # Update the dataframe
            updated_count = 0
            for i, ticker in enumerate(df[ticker_column]):
                if ticker in results:
                    # Extract status from the result message
                    if results[ticker].startswith('✅'):
                        df.loc[i, status_column] = 'Success'
                    elif results[ticker].startswith('⚠️'):
                        df.loc[i, status_column] = 'Warning'
                    else:
                        df.loc[i, status_column] = 'Failed'
                    
                    # If ticker was corrected, update it
                    if '->' in results[ticker]:
                        original, corrected = results[ticker].split('->')[0:2]
                        original = original.split(' ')[1].strip()
                        corrected = corrected.split(':')[0].strip()
                        if original != corrected:
                            df.loc[i, ticker_column] = corrected
                            updated_count += 1
            
            # Save the corrected CSV
            df.to_csv(output_filepath, sep=delimiter, index=False)
            print(f"\n{Fore.GREEN}✅ Saved processed tickers to {output_filepath}")
            print(f"Updated {updated_count} ticker symbols{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing CSV file: {str(e)}{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="Fetch and validate financial data")
    parser.add_argument('--file', '-f', help='CSV file containing tickers', default='tickers.csv')
    parser.add_argument('--output', '-o', help='Output file for processed tickers', default='tickers_corrected.csv')
    parser.add_argument('--tickers', '-t', help='Comma-separated list of tickers to fetch', type=str)
    parser.add_argument('--years', '-y', type=int, default=5, help='Years of historical data to fetch')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size for fetching')  # Increased default
    parser.add_argument('--workers', '-w', type=int, default=3, help='Number of worker threads')
    parser.add_argument('--delay', '-d', type=float, default=5.0, help='Delay between batches in seconds')  # Fixed delay
    parser.add_argument('--retry', '-r', type=int, default=3, help='Number of retry attempts')
    parser.add_argument('--no-cache', help='Disable cache', action='store_true')
    parser.add_argument('--cache-dir', help='Cache directory', type=str, default="data_cache")
    parser.add_argument('--clear-cache', help='Clear cache before starting', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = DataFetcher(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        retry_count=args.retry,
        years=args.years,
        max_workers=args.workers
    )
    
    # Show cache status
    cache_status = fetcher.cache.get_cache_status()
    print(f"\n{Fore.BLUE}Cache Status:{Style.RESET_ALL}")
    print(f"- Price data: {cache_status.get('price_ticker_count', 0)} tickers ({cache_status.get('price_cache_size_mb', 0):.2f} MB)")
    print(f"- Date range: {cache_status.get('oldest_data', 'N/A')} to {cache_status.get('newest_data', 'N/A')}")
    
    # Clear cache if requested
    if args.clear_cache:
        print(f"\n{Fore.YELLOW}Clearing cache...{Style.RESET_ALL}")
        fetcher.cache.clear_cache()
    
    # Get tickers - either from command line or file
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
        results = fetcher.fetch_all(tickers, not args.no_cache)
        
        # Show results
        print(f"\n{Fore.BLUE}{'='*20} RESULTS {'='*20}{Style.RESET_ALL}")
        for ticker, status in results.items():
            if status.startswith("✅"):
                print(f"{Fore.GREEN}{ticker}: {status}{Style.RESET_ALL}")
            elif status.startswith("⚠️"):
                print(f"{Fore.YELLOW}{ticker}: {status}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}{ticker}: {status}{Style.RESET_ALL}")
    else:
        # Process CSV file
        fetcher.process_csv_file(args.file, args.output, not args.no_cache)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}Process interrupted by user{Style.RESET_ALL}")
        sys.exit(1)

