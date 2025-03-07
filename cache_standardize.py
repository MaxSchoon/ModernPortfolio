"""
Cache Standardization Utility

This script standardizes the date formats in cached CSV files to ensure
consistent parsing and alignment across different tickers.
"""

import os
import glob
import pandas as pd
from datetime import datetime
import argparse
import sys
from typing import List, Dict, Any, Optional

try:
    from colorama import Fore, Style, init
    init()
    HAS_COLOR = True
    
    def print_info(msg):
        print(f"{Fore.BLUE}{msg}{Style.RESET_ALL}")
    
    def print_success(msg):
        print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")
    
    def print_warning(msg):
        print(f"{Fore.YELLOW}⚠️ {msg}{Style.RESET_ALL}")
    
    def print_error(msg):
        print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")
        
except ImportError:
    HAS_COLOR = False
    
    def print_info(msg):
        print(f"INFO: {msg}")
    
    def print_success(msg):
        print(f"SUCCESS: {msg}")
    
    def print_warning(msg):
        print(f"WARNING: {msg}")
    
    def print_error(msg):
        print(f"ERROR: {msg}")

def scan_cache_directory(cache_dir: str) -> Dict[str, List[str]]:
    """
    Scan cache directory for CSV files
    
    Returns:
        Dictionary with price and dividend file lists
    """
    price_dir = os.path.join(cache_dir, "prices")
    div_dir = os.path.join(cache_dir, "dividends")
    
    price_files = []
    if os.path.exists(price_dir):
        price_files = glob.glob(os.path.join(price_dir, "*_price.csv"))
    
    div_files = []
    if os.path.exists(div_dir):
        div_files = glob.glob(os.path.join(div_dir, "*_div.csv"))
    
    return {
        "price_files": price_files,
        "div_files": div_files
    }

def analyze_date_formats(files: List[str]) -> Dict[str, Any]:
    """Analyze date formats in CSV files"""
    formats = {}
    issues = []
    total_files = len(files)
    
    for i, file in enumerate(files):
        ticker = os.path.basename(file).split('_')[0]
        sys.stdout.write(f"\rAnalyzing file {i+1}/{total_files}: {ticker}")
        sys.stdout.flush()
        
        try:
            # Read first few rows to analyze date format
            df = pd.read_csv(file, nrows=5)
            
            if df.empty:
                issues.append(f"{ticker}: Empty file")
                continue
                
            # Get the first column (usually the date index)
            date_col = df.columns[0]
            first_date = df.iloc[0, 0]
            
            # Try to determine format
            date_format = None
            try:
                pd.to_datetime(first_date)  # Try pandas automatic parsing
                date_format = "auto-detect"
            except:
                # Try some common formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        datetime.strptime(str(first_date), fmt)
                        date_format = fmt
                        break
                    except:
                        continue
            
            if date_format:
                formats[file] = date_format
            else:
                issues.append(f"{ticker}: Could not determine date format for '{first_date}'")
                
        except Exception as e:
            issues.append(f"{ticker}: Error - {str(e)}")
    
    print()  # New line after progress indicator
    
    return {
        "formats": formats,
        "issues": issues
    }

def standardize_csv_file(file_path: str, date_format: str = "ISO") -> bool:
    """
    Standardize date format in a CSV file
    
    Parameters:
        file_path: Path to the CSV file
        date_format: Format to use for dates ('ISO' for YYYY-MM-DD)
        
    Returns:
        True if successful, False otherwise
    """
    ticker = os.path.basename(file_path).split('_')[0]
    
    try:
        # Read file with pandas
        df = pd.read_csv(file_path)
        
        if df.empty:
            print_warning(f"{ticker}: Empty file, skipping")
            return False
            
        # Check if first column is a date column
        date_col = df.columns[0]
        
        # Try to convert to datetime with error handling
        try:
            # First, make a backup of the original file
            backup_path = file_path + ".bak"
            if not os.path.exists(backup_path):
                # Copy file to backup
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
            
            # Convert dates to datetime objects with flexible parsing
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Check for NaT values after conversion
            nat_count = df[date_col].isna().sum()
            if nat_count > 0:
                print_warning(f"{ticker}: {nat_count} date values could not be parsed")
            
            # Remove timezone information if present
            if hasattr(df[date_col].dtype, 'tz') and df[date_col].dtype.tz is not None:
                df[date_col] = df[date_col].dt.tz_localize(None)
            
            # Sort by date to ensure chronological order
            df = df.sort_values(by=date_col)
            
            # Overwrite file with standardized ISO format
            df.to_csv(file_path, index=False, date_format="%Y-%m-%d")
            
            return True
            
        except Exception as e:
            print_error(f"{ticker}: Failed to convert dates - {str(e)}")
            return False
            
    except Exception as e:
        print_error(f"{ticker}: Error processing file - {str(e)}")
        return False
        
def remove_future_dates(file_path: str) -> int:
    """
    Remove dates beyond current date from CSV file
    
    Returns:
        Number of rows removed
    """
    ticker = os.path.basename(file_path).split('_')[0]
    
    try:
        # Read file with pandas
        df = pd.read_csv(file_path)
        
        if df.empty:
            return 0
            
        # Check if first column is a date column
        date_col = df.columns[0]
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Get current date
        current_date = pd.Timestamp.now().normalize()
        
        # Get rows with future dates
        future_rows = df[df[date_col] > current_date]
        future_count = len(future_rows)
        
        if future_count > 0:
            # Remove future dates
            df = df[df[date_col] <= current_date]
            
            # Save back to file
            df.to_csv(file_path, index=False, date_format="%Y-%m-%d")
            
        return future_count
        
    except Exception as e:
        print_error(f"{ticker}: Error removing future dates - {str(e)}")
        return 0

def standardize_cache(cache_dir: str, remove_future: bool = True) -> Dict[str, Any]:
    """
    Standardize all CSV files in cache
    
    Parameters:
        cache_dir: Cache directory
        remove_future: Whether to remove future dates
        
    Returns:
        Statistics about the standardization
    """
    print_info(f"Scanning cache directory: {cache_dir}")
    
    # Scan cache
    files = scan_cache_directory(cache_dir)
    price_files = files["price_files"]
    div_files = files["div_files"]
    
    print_info(f"Found {len(price_files)} price files and {len(div_files)} dividend files")
    
    # Analyze formats
    print_info("\nAnalyzing price file date formats...")
    price_analysis = analyze_date_formats(price_files)
    
    print_info("\nAnalyzing dividend file date formats...")
    div_analysis = analyze_date_formats(div_files)
    
    # Print issues
    if price_analysis["issues"]:
        print_warning(f"\n{len(price_analysis['issues'])} price files have format issues:")
        for issue in price_analysis["issues"][:10]:
            print(f"  - {issue}")
        if len(price_analysis["issues"]) > 10:
            print(f"  - ... and {len(price_analysis['issues']) - 10} more")
    
    if div_analysis["issues"]:
        print_warning(f"\n{len(div_analysis['issues'])} dividend files have format issues:")
        for issue in div_analysis["issues"][:10]:
            print(f"  - {issue}")
        if len(div_analysis["issues"]) > 10:
            print(f"  - ... and {len(div_analysis['issues']) - 10} more")
    
    # Confirm standardization
    total_files = len(price_files) + len(div_files)
    print_info(f"\nReady to standardize {total_files} files to ISO date format (YYYY-MM-DD)")
    
    # Statistics
    stats = {
        "price_processed": 0,
        "div_processed": 0,
        "price_errors": 0,
        "div_errors": 0,
        "future_dates_removed": 0,
    }
    
    # Process price files
    print_info("\nStandardizing price files...")
    for i, file in enumerate(price_files):
        ticker = os.path.basename(file).split('_')[0]
        sys.stdout.write(f"\rProcessing {i+1}/{len(price_files)}: {ticker}")
        sys.stdout.flush()
        
        if standardize_csv_file(file):
            stats["price_processed"] += 1
            
            # Remove future dates if requested
            if remove_future:
                removed = remove_future_dates(file)
                stats["future_dates_removed"] += removed
        else:
            stats["price_errors"] += 1
    
    print()  # New line after progress indicator
    
    # Process dividend files
    print_info("\nStandardizing dividend files...")
    for i, file in enumerate(div_files):
        ticker = os.path.basename(file).split('_')[0]
        sys.stdout.write(f"\rProcessing {i+1}/{len(div_files)}: {ticker}")
        sys.stdout.flush()
        
        if standardize_csv_file(file):
            stats["div_processed"] += 1
            
            # Remove future dates if requested
            if remove_future:
                removed = remove_future_dates(file)
                stats["future_dates_removed"] += removed
        else:
            stats["div_errors"] += 1
    
    print()  # New line after progress indicator
    
    # Print summary
    print_info("\n" + "="*50)
    print_success("Cache Standardization Complete")
    print_info("="*50)
    print(f"Price files processed: {stats['price_processed']}/{len(price_files)}")
    print(f"Dividend files processed: {stats['div_processed']}/{len(div_files)}")
    
    if stats["price_errors"] > 0 or stats["div_errors"] > 0:
        print_warning(f"Errors encountered: {stats['price_errors'] + stats['div_errors']} files could not be standardized")
    
    if stats["future_dates_removed"] > 0:
        print_warning(f"Removed {stats['future_dates_removed']} future dates from files")
    
    return stats

def verify_cache_alignment(cache_dir: str, sample_count: int = 5) -> bool:
    """
    Verify that dates in the cache are aligned
    
    Parameters:
        cache_dir: Cache directory
        sample_count: Number of files to sample
        
    Returns:
        True if aligned, False if issues detected
    """
    price_dir = os.path.join(cache_dir, "prices")
    
    if not os.path.exists(price_dir):
        print_error(f"Price directory not found: {price_dir}")
        return False
        
    # Get list of price files
    price_files = glob.glob(os.path.join(price_dir, "*_price.csv"))
    
    if not price_files:
        print_error("No price files found in cache")
        return False
        
    # Select a sample of files
    if len(price_files) > sample_count:
        import random
        sample_files = random.sample(price_files, sample_count)
    else:
        sample_files = price_files
        
    print_info(f"Verifying date alignment using {len(sample_files)} sample files...")
    
    # Load data from each file
    sample_data = {}
    date_ranges = {}
    
    for file in sample_files:
        ticker = os.path.basename(file).split('_')[0]
        
        try:
            # Read file
            df = pd.read_csv(file)
            
            if df.empty:
                print_warning(f"{ticker}: Empty file, skipping")
                continue
                
            # Check first column (date column)
            date_col = df.columns[0]
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Count NaT values
            nat_count = df[date_col].isna().sum()
            if nat_count > 0:
                print_warning(f"{ticker}: {nat_count} dates could not be parsed")
                
            # Get date range
            if not df[date_col].isna().all():
                start_date = df[date_col].min()
                end_date = df[date_col].max()
                date_ranges[ticker] = (start_date, end_date)
                print(f"{ticker}: {start_date.date()} to {end_date.date()} ({len(df)} days)")
            
        except Exception as e:
            print_error(f"{ticker}: Error - {str(e)}")
    
    # Check for alignment issues
    issues = []
    
    # Check for future dates
    current_date = pd.Timestamp.now().normalize()
    future_tickers = []
    
    for ticker, (start, end) in date_ranges.items():
        if end > current_date:
            future_tickers.append((ticker, end.date()))
            
    if future_tickers:
        issues.append(f"{len(future_tickers)} tickers have future dates:")
        for ticker, end_date in future_tickers[:5]:
            issues.append(f"  - {ticker}: {end_date}")
        if len(future_tickers) > 5:
            issues.append(f"  - ... and {len(future_tickers) - 5} more")
    
    # Report results
    if issues:
        print_warning("\nAlignment issues detected:")
        for issue in issues:
            print_warning(f"  {issue}")
        return False
    else:
        print_success("\nAll sampled files are properly aligned!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Standardize date formats in cached CSV files")
    parser.add_argument('--cache-dir', default='data_cache', help='Cache directory')
    parser.add_argument('--verify', action='store_true', help='Verify cache alignment without making changes')
    parser.add_argument('--keep-future', action='store_true', help='Keep future dates (default is to remove them)')
    parser.add_argument('--sample', type=int, default=5, help='Number of files to sample during verification')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_cache_alignment(args.cache_dir, args.sample)
    else:
        standardize_cache(args.cache_dir, not args.keep_future)
        
        # Verify after standardization
        print_info("\nVerifying cache after standardization...")
        verify_cache_alignment(args.cache_dir, args.sample)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)
