"""
Financial Data Cache Maintenance Tool

This tool provides advanced maintenance operations for the CSV data cache:
- Ticker validation and correction
- Cache inspection and reporting
- Data quality testing and repair
- Parallel batch operations
- Cache format conversion (pickle to CSV)

All functions work with the standard CSVDataCache from csv_cache_manager.py
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta
import time
import sys
import re
import json
import shutil
import argparse
import concurrent.futures
from typing import List, Dict, Tuple, Any, Optional

try:
    from tqdm import tqdm
    import yfinance as yf
    import matplotlib.pyplot as plt
    from colorama import Fore, Style, init
    
    # Initialize colorama for colored terminal output
    init()
    HAS_COLOR = True
except ImportError:
    # Create dummy tqdm if not available
    def tqdm(iterable, **kwargs):
        return iterable
    HAS_COLOR = False

# Import required utilities
from csv_cache_manager import CSVDataCache
from utils import (
    validate_ticker, format_ticker, print_info, print_success, 
    print_warning, print_error
)

# Default cache directory
DEFAULT_CACHE_DIR = "data_cache"

#==============================================================================
# TICKER VALIDATION FUNCTIONS
#==============================================================================

def validate_ticker_list(tickers: List[str], verbose: bool = True) -> Dict[str, str]:
    """
    Validate a list of tickers against Yahoo Finance
    
    Parameters:
        tickers: List of ticker symbols
        verbose: Whether to print validation details
        
    Returns:
        Dictionary of {ticker: status_message}
    """
    results = {}
    
    print_info(f"Validating {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        # Skip special assets
        if ticker in ['CASH', 'TBILLS']:
            results[ticker] = f"✅ {ticker}: Special asset"
            if verbose:
                print(results[ticker])
            continue
        
        # Validate ticker
        is_valid, corrected, message = validate_ticker(ticker, verbose=False)
        
        if is_valid:
            if ticker != corrected:
                results[ticker] = f"🔄 {ticker} -> {corrected}: {message}"
            else:
                results[ticker] = f"✅ {ticker}: {message}"
        else:
            results[ticker] = f"❌ {ticker}: {message}"
        
        if verbose:
            print(results[ticker])
        
        # Add delay to avoid rate limiting
        if i < len(tickers) - 1:
            time.sleep(1)
    
    return results

def validate_csv_tickers(filepath: str, output_filepath: str = None) -> None:
    """
    Validate and correct tickers in a CSV file
    
    Parameters:
        filepath: Path to the input CSV file
        output_filepath: Path to save corrected CSV
    """
    if not os.path.exists(filepath):
        print_error(f"File not found: {filepath}")
        return
    
    if output_filepath is None:
        base, ext = os.path.splitext(filepath)
        output_filepath = f"{base}_corrected{ext}"
    
    # Read the CSV file
    try:
        # Try to detect delimiter
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if ';' in first_line:
                delimiter = ';'
                print_info(f"Detected semicolon delimiter in {filepath}")
            elif ',' in first_line:
                delimiter = ','
                print_info(f"Detected comma delimiter in {filepath}")
            else:
                delimiter = ';'  # Default
                print_info(f"Using default semicolon delimiter for {filepath}")
        
        df = pd.read_csv(filepath, sep=delimiter)
        
        # Check column names (case insensitive)
        ticker_column = None
        for col in df.columns:
            if col.lower() == 'ticker':
                ticker_column = col
                break
                
        if ticker_column is None:
            print_error(f"CSV columns: {', '.join(df.columns)}")
            print_error("CSV file must contain a 'ticker' column")
            return
            
        tickers = df[ticker_column].tolist()
        print_info(f"Found {len(tickers)} tickers in the CSV file")
        
        # Validate and correct each ticker
        validation_results = validate_ticker_list(tickers, verbose=True)
        
        # Extract corrected tickers
        corrected_tickers = []
        for ticker in tickers:
            result = validation_results.get(ticker, "")
            if "->" in result:
                # Extract corrected ticker from the result
                corrected = result.split("->")[1].split(":")[0].strip()
                corrected_tickers.append(corrected)
            else:
                corrected_tickers.append(ticker)
        
        # Update the dataframe with corrected tickers
        df[ticker_column] = corrected_tickers
        
        # Add status column if it doesn't exist
        status_column = 'status'
        if status_column not in df.columns:
            df[status_column] = 'Unknown'
        
        # Update status
        for i, ticker in enumerate(tickers):
            result = validation_results.get(ticker, "")
            if "✅" in result:
                df.loc[i, status_column] = 'Valid'
            elif "🔄" in result:
                df.loc[i, status_column] = 'Corrected'
            else:
                df.loc[i, status_column] = 'Invalid'
        
        # Save the corrected CSV
        df.to_csv(output_filepath, sep=delimiter, index=False)
        print_success(f"Saved corrected tickers to {output_filepath}")
        
    except Exception as e:
        print_error(f"Error processing CSV file: {str(e)}")

#==============================================================================
# CACHE INSPECTION FUNCTIONS
#==============================================================================

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
    if div_data is not None:
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
    
    # Show sample data
    print_info("Sample price data (first 5 rows):")
    temp_df = pd.DataFrame({'price': price_data})
    if div_data is not None:
        temp_df['dividend'] = div_data
    print(temp_df.head())
    
    # Show sample data
    print_info("Sample price data (last 5 rows):")
    print(temp_df.tail())
    
    # Cache file info
    price_file = os.path.join(cache_dir, "prices", f"{ticker}_price.csv")
    if os.path.exists(price_file):
        file_size = os.path.getsize(price_file) / 1024
        last_modified = datetime.fromtimestamp(os.path.getmtime(price_file))
        print_info("Cache file info:")
        print(f"- File: {price_file}")
        print(f"- Size: {file_size:.1f} KB")
        print(f"- Last modified: {last_modified}")

def check_all_tickers_data_quality(cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Check data quality for all cached tickers"""
    cache = CSVDataCache(cache_dir)
    price_dir = os.path.join(cache_dir, "prices")
    
    if not os.path.exists(price_dir):
        print_error(f"Cache directory not found: {price_dir}")
        return
    
    files = glob.glob(os.path.join(price_dir, "*_price.csv"))
    
    print_info(f"Checking data quality for {len(files)} cached tickers...")
    
    good_tickers = []
    bad_tickers = []
    
    for i, file in enumerate(files):
        ticker = os.path.basename(file).replace("_price.csv", "")
        sys.stdout.write(f"\rChecking {i+1}/{len(files)}: {ticker}" + " " * 20)
        sys.stdout.flush()
        
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            
            if 'Close' in df.columns:
                column = 'Close'
            elif 'price' in df.columns:
                column = 'price'
            else:
                column = df.columns[0]  # Take the first column
            
            # Check for NaN values
            nan_count = df[column].isna().sum()
            nan_pct = nan_count / len(df) * 100
            
            if nan_pct > 10:
                bad_tickers.append((ticker, f"{nan_pct:.1f}% NaN"))
            elif len(df) < 100:
                bad_tickers.append((ticker, f"Only {len(df)} data points"))
            else:
                good_tickers.append(ticker)
                
        except Exception as e:
            bad_tickers.append((ticker, f"Error: {str(e)}"))
    
    print("\n")  # Clear the line after progress updates
    print_success(f"Good data: {len(good_tickers)} tickers")
    print_error(f"Problematic data: {len(bad_tickers)} tickers")
    
    if bad_tickers:
        print_info("Problematic tickers:")
        for ticker, issue in bad_tickers:
            print(f"- {ticker}: {issue}")
            
    # Save results to file
    results_file = os.path.join(cache_dir, "data_quality_report.txt")
    with open(results_file, 'w') as f:
        f.write(f"Data Quality Report - {datetime.now()}\n\n")
        f.write(f"Good data: {len(good_tickers)} tickers\n")
        f.write(f"Problematic data: {len(bad_tickers)} tickers\n\n")
        
        f.write("Problematic tickers:\n")
        for ticker, issue in bad_tickers:
            f.write(f"- {ticker}: {issue}\n")
    
    print_success(f"Report saved to {results_file}")

def plot_ticker(ticker: str, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Plot price data for a ticker"""
    try:
        cache = CSVDataCache(cache_dir)
        price_data = cache.get_price_data(ticker)
        
        if price_data is None:
            print_error(f"No data found for {ticker}")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(price_data.index, price_data.values)
        plt.title(f"{ticker} Price History")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        
        # Add annotations for start/end prices
        start_price = price_data.iloc[0]
        end_price = price_data.iloc[-1]
        plt.annotate(f"${start_price:.2f}", xy=(price_data.index[0], start_price),
                    xytext=(10, 10), textcoords="offset points")
        plt.annotate(f"${end_price:.2f}", xy=(price_data.index[-1], end_price),
                    xytext=(-40, 10), textcoords="offset points")
        
        # Save to file
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, f"{ticker}_price.png")
        plt.savefig(output_file)
        
        print_success(f"Plot saved to {output_file}")
        plt.show()
        
    except ImportError:
        print_error(f"Matplotlib not installed. Install it with 'pip install matplotlib'")
    except Exception as e:
        print_error(f"Error plotting data: {str(e)}")

#==============================================================================
# DATA QUALITY FIX FUNCTIONS
#==============================================================================

def inspect_csv_file(filepath: str) -> pd.DataFrame:
    """Inspect a CSV file to diagnose issues"""
    print_info(f"Inspecting file: {filepath}")
    
    try:
        # Try reading without parsing dates first to see raw data
        raw_df = pd.read_csv(filepath)
        print_info("Raw data (first 5 rows):")
        print(raw_df.head())
        
        # Check column names
        print_info(f"Columns: {raw_df.columns.tolist()}")
        
        # Check for date/index column type
        index_col = raw_df.columns[0]  # Usually the first column is the index/date
        print_info(f"Index column '{index_col}' data type: {raw_df[index_col].dtype}")
        
        # Try to parse with correct settings
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print_info("Parsed data with index_col=0, parse_dates=True (first 5 rows):")
        print(df.head())
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        nan_pct = df.isna().mean().mean() * 100
        print_info(f"NaN values in parsed data: {nan_count} ({nan_pct:.2f}%)")
        
        # Check data type of values
        if 'Close' in df.columns:
            col = 'Close'
        elif 'price' in df.columns:
            col = 'price'
        elif len(df.columns) > 0:
            col = df.columns[0]
        else:
            col = None
        
        if col:
            print_info(f"{col} column data type: {df[col].dtype}")
            print_info(f"{col} range: {df[col].min()} to {df[col].max()}")
        
        return df
    except Exception as e:
        print_error(f"Error inspecting file: {type(e).__name__}: {str(e)}")
        return None

def fix_csv_file(filepath: str, output_filepath: str = None) -> bool:
    """Fix issues in a CSV file and save a corrected version"""
    if output_filepath is None:
        # Create a new filename with _fixed suffix
        base, ext = os.path.splitext(filepath)
        output_filepath = f"{base}_fixed{ext}"
    
    try:
        print_info(f"Fixing file: {filepath}")
        print_info(f"Output will be saved to: {output_filepath}")
        
        # Read the file with flexible parsing
        df = pd.read_csv(filepath)
        
        # Ensure the first column is interpreted as dates
        date_col = df.columns[0]
        
        # Convert date strings to datetime objects
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            print_warning(f"Error converting dates: {type(e).__name__}: {str(e)}")
            print_info("Attempting to fix date format...")
            
            # Try multiple date formats
            for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                    if not df[date_col].isna().all():
                        print_success(f"Successfully parsed dates using format: {date_format}")
                        break
                except:
                    continue
        
        # Set the date column as index
        df.set_index(date_col, inplace=True)
        
        # Determine price column name
        price_col = None
        if 'Close' in df.columns:
            price_col = 'Close'
        elif 'price' in df.columns:
            price_col = 'price'
        else:
            price_col = df.columns[0]  # Assume first column is price
        
        # Ensure price column is properly typed
        if price_col:
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            
        # Check for dividend column
        div_col = None
        if 'Dividend' in df.columns:
            div_col = 'Dividend'
        elif 'dividend' in df.columns:
            div_col = 'dividend'
        
        if div_col:
            df[div_col] = pd.to_numeric(df[div_col], errors='coerce')
        
        # Sort index to ensure chronological order
        df.sort_index(inplace=True)
        
        # Fill any NaN values
        nan_before = df.isna().sum().sum()
        if nan_before > 0:
            # Forward/backward fill price column
            if price_col:
                price_nans = df[price_col].isna().sum()
                if price_nans > 0:
                    df[price_col] = df[price_col].ffill().bfill()
                    print_success(f"Filled {price_nans} NaN values in {price_col} column")
            
            # Fill NaN in dividend column with 0
            if div_col and div_col in df.columns:
                div_nans = df[div_col].isna().sum()
                if div_nans > 0:
                    df[div_col] = df[div_col].fillna(0)
                    print_success(f"Filled {div_nans} NaN values in {div_col} column with 0")
        
        # Save fixed file
        df.to_csv(output_filepath)
        
        # Validate the fix
        fixed_df = pd.read_csv(output_filepath, index_col=0, parse_dates=True)
        nan_after = fixed_df.isna().sum().sum()
        print_info(f"NaN values after fixing: {nan_after}")
            
        print_success(f"File fixed and saved to {output_filepath}")
        return True
    except Exception as e:
        print_error(f"Error fixing file: {type(e).__name__}: {str(e)}")
        return False

def test_and_repair_cached_data(cache_dir: str = DEFAULT_CACHE_DIR, verbose: bool = False, repair: bool = False) -> None:
    """Test the quality of cached data and optionally repair issues"""
    cache = CSVDataCache(cache_dir)
    
    # Get cache status
    status = cache.get_cache_status()
    
    print_info("===== Data Cache Quality Report =====")
    print(f"Date range: {status.get('oldest_data', 'Unknown')} to {status.get('newest_data', 'Unknown')}")
    print(f"Price tickers: {status.get('price_ticker_count', 0)}")
    print(f"Cache size: {status.get('price_cache_size_mb', 0):.2f} MB")
    
    # Get all price files
    price_dir = os.path.join(cache_dir, "prices")
    if not os.path.exists(price_dir):
        print_error(f"Price directory not found: {price_dir}")
        return
        
    price_files = glob.glob(os.path.join(price_dir, "*_price.csv"))
    tickers = [os.path.basename(f).split("_price")[0] for f in price_files]
    
    print_info(f"Analyzing {len(tickers)} tickers...")
    
    # Check each ticker's data quality
    good_tickers = []
    bad_tickers = []
    repaired_tickers = []
    
    for i, ticker in enumerate(tickers):
        sys.stdout.write(f"\rProcessing {i+1}/{len(tickers)}: {ticker}" + " " * 20)
        sys.stdout.flush()
            
        # Load data for this ticker
        prices = cache.get_price_data(ticker, max_age_days=365)  # Accept data up to a year old
        
        if prices is None:
            bad_tickers.append((ticker, "Failed to load"))
            continue
            
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
            if repair:
                if nan_pct > 50:  # Severe issues - refetch
                    try:
                        print(f"\n{ticker} (high NaN count: {nan_pct:.1f}%)")
                        
                        # Try to fetch fresh data
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365*5)  # 5 years
                        
                        stock = yf.Ticker(ticker)
                        new_prices = stock.history(start=start_date, end=end_date)['Close']
                        
                        if len(new_prices) > 10:  # We got some data
                            new_prices.index = new_prices.index.tz_localize(None)
                            cache.save_price_data(ticker, new_prices)
                            
                            # Also try to get dividends
                            try:
                                dividends = stock.dividends
                                if not dividends.empty:
                                    dividends.index = dividends.index.tz_localize(None)
                                    div_series = pd.Series(0.0, index=new_prices.index)
                                    common_dates = dividends.index.intersection(div_series.index)
                                    
                                    if common_dates.size > 0:
                                        div_series.loc[common_dates] = dividends[dividends.index.isin(common_dates)]
                                    
                                    cache.save_div_data(ticker, div_series)
                            except Exception as e:
                                print_error(f"Couldn't save dividend data for {ticker}: {e}")
                                
                            repaired_tickers.append(ticker)
                            print_success(f"Successfully repaired {ticker} with {len(new_prices)} data points")
                        else:
                            print_error(f"Failed to repair {ticker} - insufficient data")
                    
                    except Exception as e:
                        print_error(f"Error repairing {ticker}: {e}")
                        
                else:  # Minor issues - just clean up
                    try:
                        clean_prices = prices.ffill().bfill()
                        cache.save_price_data(ticker, clean_prices)
                        repaired_tickers.append(ticker)
                        print_success(f"\nFixed {ticker} by filling {nan_count} NaN values")
                    except Exception as e:
                        print_error(f"\nError fixing {ticker}: {e}")
                
            bad_tickers.append((ticker, f"{nan_pct:.1f}% NaN"))
    
    print("\n")  # Clear the progress line
    print_success(f"Good data: {len(good_tickers)} tickers")
    print_error(f"Problematic data: {len(bad_tickers)} tickers")
    if repair:
        print_success(f"Repaired: {len(repaired_tickers)} tickers")
    
    if bad_tickers:
        print_warning("Problematic tickers:")
        for ticker, issue in bad_tickers[:10]:  # Show first 10
            print(f"- {ticker}: {issue}")
        if len(bad_tickers) > 10:
            print(f"... and {len(bad_tickers) - 10} more")
    
    # Save results to file
    results_file = os.path.join(cache_dir, "data_quality_report.txt")
    with open(results_file, 'w') as f:
        f.write(f"Data Quality Report - {datetime.now()}\n\n")
        f.write(f"Good data: {len(good_tickers)} tickers\n")
        f.write(f"Problematic data: {len(bad_tickers)} tickers\n")
        if repair:
            f.write(f"Repaired: {len(repaired_tickers)} tickers\n")
        f.write("\nProblematic tickers:\n")
        for ticker, issue in bad_tickers:
            f.write(f"- {ticker}: {issue}\n")
    
    print_success(f"Report saved to {results_file}")

#==============================================================================
# CACHE CONVERSION FUNCTION
#==============================================================================

def convert_pickle_to_csv(old_cache_dir: str, new_cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """
    Convert data from old pickle cache format to new CSV format
    """
    try:
        import pickle
        
        # Check if old cache exists
        if not os.path.exists(old_cache_dir):
            print_error(f"Old cache directory not found: {old_cache_dir}")
            return
            
        # Create CSV cache
        csv_cache = CSVDataCache(new_cache_dir)
        
        # Check for pickle files
        price_files = glob.glob(os.path.join(old_cache_dir, "*.pkl"))
        if not price_files:
            print_error(f"No pickle files found in {old_cache_dir}")
            return
            
        print_info(f"Found {len(price_files)} pickle files to convert")
        
        converted_count = 0
        for pkl_file in price_files:
            ticker = os.path.basename(pkl_file).split('.')[0]
            print(f"Converting {ticker}...")
            
            try:
                # Load pickle data
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Check expected format
                if isinstance(data, dict) and 'prices' in data:
                    prices = data.get('prices')
                    dividends = data.get('dividends')
                    
                    # Convert and save
                    if isinstance(prices, pd.Series) and not prices.empty:
                        csv_cache.save_price_data(ticker, prices)
                        
                        if isinstance(dividends, pd.Series) and not dividends.empty:
                            csv_cache.save_div_data(ticker, dividends)
                            
                        converted_count += 1
                        print_success(f"Successfully converted {ticker}")
                    else:
                        print_warning(f"{ticker}: Invalid price data format")
                else:
                    print_warning(f"{ticker}: Unexpected data format")
                    
            except Exception as e:
                print_error(f"Error converting {ticker}: {str(e)}")
        
        if converted_count > 0:
            print_success(f"Successfully converted {converted_count} out of {len(price_files)} tickers")
            
            # Print status of the new cache
            status = csv_cache.get_cache_status()
            print_info("New CSV Cache Status:")
            print(f"- Price data: {status.get('price_ticker_count', 0)} tickers ({status.get('price_cache_size_mb', 0):.2f} MB)")
            print(f"- Date range: {status.get('oldest_data', 'N/A')} to {status.get('newest_data', 'N/A')}")
        else:
            print_warning("No tickers were converted")
            
    except ImportError:
        print_error("Pickle module not available")
    except Exception as e:
        print_error(f"Error during conversion: {str(e)}")

#==============================================================================
# BATCH OPERATIONS
#==============================================================================

def batch_validate_and_fix(cache_dir: str = DEFAULT_CACHE_DIR, parallel: bool = True) -> None:
    """
    Validate and fix all cached tickers in batch mode with parallel processing
    """
    cache = CSVDataCache(cache_dir)
    price_dir = os.path.join(cache_dir, "prices")
    
    if not os.path.exists(price_dir):
        print_error(f"Cache directory not found: {price_dir}")
        return
    
    # Get all tickers
    price_files = glob.glob(os.path.join(price_dir, "*_price.csv"))
    tickers = [os.path.basename(f).replace("_price.csv", "") for f in price_files]
    
    if not tickers:
        print_warning("No tickers found in cache")
        return
    
    print_info(f"Starting batch validation for {len(tickers)} tickers...")
    
    # Create a backup first
    backup_dir = os.path.join(cache_dir, "backup_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy essential files
    for file in price_files:
        ticker = os.path.basename(file).replace("_price.csv", "")
        backup_file = os.path.join(backup_dir, f"{ticker}_price.csv")
        shutil.copy2(file, backup_file)
    
    print_success(f"Created backup in {backup_dir}")
    
    # Process function for a single ticker
    def process_ticker(ticker):
        result = {"ticker": ticker, "status": "unknown", "message": ""}
        
        try:
            # Get price data
            prices = cache.get_price_data(ticker)
            
            if prices is None:
                result["status"] = "error"
                result["message"] = "No data available"
                return result
            
            # Check for NaN values
            nan_count = prices.isna().sum()
            nan_pct = (nan_count / len(prices)) * 100
            
            if nan_pct > 0:
                # Try to fix
                clean_prices = prices.ffill().bfill()
                cache.save_price_data(ticker, clean_prices)
                
                if nan_pct > 50:
                    result["status"] = "warning"
                    result["message"] = f"Fixed, but had {nan_pct:.1f}% NaN values"
                else:
                    result["status"] = "fixed"
                    result["message"] = f"Fixed {nan_count} NaN values"
            else:
                result["status"] = "good"
                result["message"] = "No issues found"
                
        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            
        return result
    
    # Process tickers in parallel or sequential
    results = []
    
    if parallel and concurrent.futures is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as executor:
            future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
            
            with tqdm(total=len(tickers), desc="Processing tickers") as pbar:
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "ticker": ticker, 
                            "status": "error", 
                            "message": f"Thread error: {str(e)}"
                        })
                    pbar.update(1)
    else:
        for ticker in tqdm(tickers, desc="Processing tickers"):
            results.append(process_ticker(ticker))
    
    # Summarize results
    status_counts = {"good": 0, "fixed": 0, "warning": 0, "error": 0}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
    
    print("\nProcessing Summary:")
    print_success(f"Good: {status_counts['good']}")
    print_info(f"Fixed: {status_counts['fixed']}")
    print_warning(f"Warning: {status_counts['warning']}")
    print_error(f"Error: {status_counts['error']}")
    
    # Save detailed results
    report_file = os.path.join(cache_dir, "batch_validation_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Batch Validation Report - {datetime.now()}\n\n")
        f.write(f"Summary:\n")
        f.write(f"- Good: {status_counts['good']}\n")
        f.write(f"- Fixed: {status_counts['fixed']}\n")
        f.write(f"- Warning: {status_counts['warning']}\n")
        f.write(f"- Error: {status_counts['error']}\n\n")
        
        f.write("Detailed Results:\n")
        for result in results:
            f.write(f"{result['ticker']}: {result['status']} - {result['message']}\n")
    
    print_success(f"Detailed report saved to {report_file}")

#==============================================================================
# MAIN FUNCTION
#==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Financial Data Cache Maintenance Tool"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List cache contents
    list_parser = subparsers.add_parser('list', help='List all cached tickers')
    list_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
    
    # Inspect ticker data
    inspect_parser = subparsers.add_parser('inspect', help='Inspect data for a specific ticker')
    inspect_parser.add_argument('ticker', help='Ticker symbol to inspect')
    inspect_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
    
    # Plot ticker data
    plot_parser = subparsers.add_parser('plot', help='Plot price data for a ticker')
    plot_parser.add_argument('ticker', help='Ticker symbol to plot')
    plot_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
    
    # Validate tickers
    validate_parser = subparsers.add_parser('validate', help='Validate ticker symbols')
    validate_parser.add_argument('--tickers', nargs='+', help='List of tickers to validate')
    validate_parser.add_argument('--file', '-f', help='CSV file containing tickers to validate')
    validate_parser.add_argument('--output', '-o', help='Output file for corrected tickers')
    
    # Check data quality
    quality_parser = subparsers.add_parser('quality', help='Check data quality')
    quality_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
    quality_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # Test and repair data
    repair_parser = subparsers.add_parser('repair', help='Test and repair data issues')
    repair_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
    repair_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    repair_parser.add_argument('--file', help='Fix a specific CSV file')
    repair_parser.add_argument('--output', help='Output file for fixed CSV')
    repair_parser.add_argument('--batch', '-b', action='store_true', help='Process all cached files')
    repair_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    
    # Convert cache format
    convert_parser = subparsers.add_parser('convert', help='Convert from pickle to CSV cache')
    convert_parser.add_argument('--old-cache', required=True, help='Old pickle cache directory')
    convert_parser.add_argument('--new-cache', default=DEFAULT_CACHE_DIR, help='New CSV cache directory')
    
    # Delete cache
    clear_parser = subparsers.add_parser('clear', help='Clear cache files')
    clear_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompting')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_cached_tickers(args.cache_dir)
        
    elif args.command == 'inspect':
        inspect_ticker_data(args.ticker, args.cache_dir)
        
    elif args.command == 'plot':
        plot_ticker(args.ticker, args.cache_dir)
        
    elif args.command == 'validate':
        if args.tickers:
            validate_ticker_list(args.tickers)
        elif args.file:
            validate_csv_tickers(args.file, args.output)
        else:
            print_warning("Please specify either --tickers or --file")
            
    elif args.command == 'quality':
        check_all_tickers_data_quality(args.cache_dir)
        
    elif args.command == 'repair':
        if args.file:
            if not inspect_csv_file(args.file):
                print_warning("Inspection failed, attempting fix anyway")
            fix_csv_file(args.file, args.output)
        elif args.batch:
            batch_validate_and_fix(args.cache_dir, not args.no_parallel)
        else:
            test_and_repair_cached_data(args.cache_dir, args.verbose, repair=True)
            
    elif args.command == 'convert':
        convert_pickle_to_csv(args.old_cache, args.new_cache)
        
    elif args.command == 'clear':
        if args.confirm or input("Are you sure you want to clear the cache? (y/n): ").lower() == 'y':
            cache = CSVDataCache(args.cache_dir)
            cache.clear_cache()
            print_success("Cache cleared")
        else:
            print("Cache clearing aborted")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)
