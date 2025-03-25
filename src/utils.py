"""
Utility functions for Modern Portfolio Optimizer

This module contains common helper functions used across different scripts.
"""

import os
import pandas as pd
import re
from typing import Any, List, Dict, Optional, Tuple

def load_tickers(csv_path: str) -> List[str]:
    """
    Load tickers from CSV file with robust error handling
    
    Parameters:
        csv_path: Path to the CSV file containing tickers
        
    Returns:
        List of ticker symbols
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tickers file not found: {csv_path}")
    
    try:
        print(f"Reading ticker data from: {csv_path}")
        
        # Try to detect delimiter
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            if ';' in first_line:
                delimiter = ';'
                print("Detected semicolon delimiter")
            elif ',' in first_line:
                delimiter = ','
                print("Detected comma delimiter")
            else:
                delimiter = ';'  # Default
                print("Using default semicolon delimiter")
        
        df = pd.read_csv(csv_path, sep=delimiter)
        
        # Check column names (case insensitive)
        ticker_column = None
        for col in df.columns:
            if col.lower() == 'ticker':
                ticker_column = col
                break
                
        if ticker_column is None:
            print(f"Columns in CSV: {', '.join(df.columns)}")
            raise ValueError("CSV file must contain a 'ticker' column")
            
        tickers = df[ticker_column].tolist()
        print(f"Found {len(tickers)} tickers in the CSV file")
        
        # Print first few tickers
        sample = tickers[:min(5, len(tickers))]
        print(f"Sample tickers: {', '.join(sample)}")
        
        return tickers
    except Exception as e:
        print(f"Error reading tickers file: {str(e)}")
        raise

def get_exchange_suffix(country_code: str) -> Optional[str]:
    """
    Get the exchange suffix for a country code
    
    Parameters:
        country_code: Two-letter country code (e.g., 'NL', 'US', 'GB')
        
    Returns:
        Exchange suffix or None if not found
    """
    exchange_map = {
        'NL': '.AS',  # Netherlands (Amsterdam)
        'GB': '.L',   # United Kingdom (London)
        'FR': '.PA',  # France (Paris)
        'DE': '.DE',  # Germany (Frankfurt)
        'IT': '.MI',  # Italy (Milan)
        'ES': '.MC',  # Spain (Madrid)
        'CH': '.SW',  # Switzerland (Swiss Exchange)
        'SE': '.ST',  # Sweden (Stockholm)
        'NO': '.OL',  # Norway (Oslo)
        'DK': '.CO',  # Denmark (Copenhagen)
        'BE': '.BR',  # Belgium (Brussels)
        'AT': '.VI',  # Austria (Vienna)
        'PT': '.LS',  # Portugal (Lisbon)
        'GR': '.AT',  # Greece (Athens)
        'AU': '.AX',  # Australia (Australian Exchange)
        'CA': '.TO',  # Canada (Toronto)
        'JP': '.T',   # Japan (Tokyo)
        'HK': '.HK',  # Hong Kong
    }
    
    return exchange_map.get(country_code.upper())

def is_valid_ticker_format(ticker: str) -> bool:
    """
    Check if a ticker is properly formatted
    
    Parameters:
        ticker: The ticker to check
        
    Returns:
        True if the ticker appears to be properly formatted
    """
    # Special assets are always valid
    if ticker in ['CASH', 'TBILLS']:
        return True
        
    # Check for common issues
    if ticker.lower() in ['danaos', 'flow', 'hal']:
        return False  # These need correction
        
    # Check if it has a valid exchange suffix
    if '.' in ticker:
        suffix = ticker.split('.')[-1]
        valid_suffixes = ['AS', 'L', 'PA', 'DE', 'MI', 'MC', 'SW', 'ST', 
                          'OL', 'CO', 'BR', 'VI', 'LS', 'AT', 'AX', 'TO', 
                          'T', 'HK']
        if suffix in valid_suffixes:
            return True
    
    # US stocks don't need suffix
    return True

def validate_ticker(ticker: str, verbose: bool = True) -> Tuple[bool, str, str]:
    """
    Validate a ticker symbol and provide correction if needed
    
    Parameters:
        ticker: Ticker symbol to validate
        verbose: Whether to print validation details
        
    Returns:
        Tuple of (is_valid, corrected_ticker, message)
    """
    if ticker is None or not isinstance(ticker, str):
        return False, ticker, "Invalid ticker type"
        
    # Normalize by removing whitespace and converting to uppercase
    corrected = ticker.strip().upper()
    
    # Handle special assets
    if corrected in ['CASH', 'TBILLS']:
        return True, corrected, "Special asset"
    
    # Handle empty tickers
    if not corrected:
        return False, ticker, "Empty ticker"
    
    # Handle special cases
    if corrected.lower() == 'danaos':
        corrected = 'DAC'
    elif corrected.lower() == 'flow':
        corrected = 'FLOW.AS'
    elif corrected.lower() == 'hal':
        corrected = 'HAL.AS'
        
    # Check if the ticker contains invalid characters
    if not re.match(r'^[\w\.-]+$', corrected):
        return False, ticker, "Invalid characters in ticker"
        
    # For international exchanges, ensure correct suffix format
    if '.' in corrected:
        suffix = corrected.split('.')[-1]
        valid_suffixes = ['AS', 'L', 'PA', 'DE', 'MI', 'MC', 'SW', 'ST', 
                          'OL', 'CO', 'BR', 'VI', 'LS', 'AT', 'AX', 'TO', 
                          'T', 'HK']
        if suffix not in valid_suffixes:
            if verbose:
                print(f"Warning: '{ticker}' has unknown exchange suffix '{suffix}'")
    
    if verbose:
        if ticker != corrected:
            print(f"Corrected '{ticker}' to '{corrected}'")
    
    return True, corrected, "Valid ticker"

def validate_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Validate a dataframe for portfolio optimization calculations"""
    report = {
        "name": name,
        "is_valid": True,
        "issues": [],
        "shape": df.shape,
        "index_type": str(type(df.index))
    }
    
    # Check for empty dataframe
    if df.empty:
        report["is_valid"] = False
        report["issues"].append("DataFrame is empty")
        return report
        
    # Check for datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        report["is_valid"] = False
        report["issues"].append("Index is not DatetimeIndex")
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        report["issues"].append(f"NaN values found in columns: {', '.join(nan_cols)}")
        report["nan_columns"] = nan_cols
        
        # Count NaN values per column
        nan_counts = df.isna().sum().to_dict()
        report["nan_counts"] = nan_counts
    
    return report

def get_date_range(ticker: str, price_data: pd.Series) -> str:
    """
    Get a formatted date range string for a ticker's price data
    
    Parameters:
        ticker: The ticker symbol
        price_data: Series containing price data
    
    Returns:
        Formatted date range string
    """
    if price_data is None or price_data.empty:
        return f"{ticker}: No data available"
        
    start_date = price_data.index[0].strftime('%Y-%m-%d')
    end_date = price_data.index[-1].strftime('%Y-%m-%d')
    days = len(price_data)
    
    return f"{ticker}: {days} days from {start_date} to {end_date}"

def validate_prices(price_data: pd.Series, min_points: int = 100) -> Dict[str, Any]:
    """
    Validate price data quality
    
    Parameters:
        price_data: Series containing price data
        min_points: Minimum number of data points required
        
    Returns:
        Dictionary with validation results
    """
    if price_data is None or price_data.empty:
        return {
            'valid': False,
            'error': 'No data available',
            'nan_pct': 100
        }
        
    # Check data length
    if len(price_data) < min_points:
        return {
            'valid': False,
            'error': f'Insufficient data points ({len(price_data)} < {min_points})',
            'nan_pct': 0
        }
        
    # Check for NaN values
    nan_count = price_data.isna().sum()
    nan_pct = (nan_count / len(price_data)) * 100
    
    if nan_pct > 50:
        return {
            'valid': False,
            'error': f'Too many NaN values ({nan_pct:.1f}%)',
            'nan_pct': nan_pct
        }
        
    # Data is valid
    return {
        'valid': True,
        'nan_pct': nan_pct,
        'data_points': len(price_data),
        'start_date': price_data.index[0],
        'end_date': price_data.index[-1],
        'min_price': price_data.min(),
        'max_price': price_data.max()
    }

def log_progress(message: str, title: bool = False, error: bool = False) -> None:
    """
    Print a formatted progress log message
    
    Parameters:
        message: The message to print
        title: If True, formats as a section title
        error: If True, formats as an error message
    """
    if title:
        print(f"\n{'='*10} {message} {'='*10}")
    elif error:
        print(f"❌ {message}")
    else:
        print(f"- {message}")

###############################################################################
# Helper Functions for Display
###############################################################################
def format_ticker(ticker: str) -> str:
    """
    Format ticker symbol for Yahoo Finance API
    
    Handles special cases and international exchanges
    
    Parameters:
        ticker: The ticker symbol to format
        
    Returns:
        Properly formatted ticker symbol for Yahoo Finance
    """
    # Handle special case for Danaos (DAC) which might be in the ticker list as 'Danaos'
    if ticker.lower() == 'danaos':
        return 'DAC'
        
    # Handle Flow Traders which should be FLOW.AS
    if ticker.lower() == 'flow':
        return 'FLOW.AS'
        
    # Handle HAL Trust which should be HAL.AS
    if ticker.lower() == 'hal':
        return 'HAL.AS'
        
    # For US stocks, nothing changes for most tickers
    if '.' not in ticker and '-' not in ticker:
        return ticker
        
    # Handle stocks with special characters
    if '-' in ticker:  # Stocks like BRK-B need to be formatted as BRK-B
        return ticker
        
    # Handle international exchanges
    if ticker.endswith('.AS'):  # Amsterdam
        return ticker
    elif ticker.endswith('.L'):  # London
        return ticker
    elif ticker.endswith('.PA'):  # Paris 
        return ticker
    elif ticker.endswith('.DE'):  # Germany
        return ticker
    elif ticker.endswith('.MI'):  # Milan
        return ticker
    
    # For other formats, leave as is
    return ticker

def print_info(message: str) -> None:
    """Print an informational message"""
    try:
        from colorama import Fore, Style
        print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")
    except ImportError:
        print(f"INFO: {message}")

def print_success(message: str) -> None:
    """Print a success message"""
    try:
        from colorama import Fore, Style
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
    except ImportError:
        print(f"SUCCESS: {message}")

def print_warning(message: str) -> None:
    """Print a warning message"""
    try:
        from colorama import Fore, Style
        print(f"{Fore.YELLOW}⚠️ {message}{Style.RESET_ALL}")
    except ImportError:
        print(f"WARNING: {message}")
        
def print_error(message: str) -> None:
    """Print an error message"""
    try:
        from colorama import Fore, Style
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
    except ImportError:
        print(f"ERROR: {message}")

def validate_prices_df(df: pd.DataFrame, ticker: str = None) -> Dict[str, Any]:
    """
    Validate a dataframe of price data
    
    Parameters:
        df: DataFrame with price data
        ticker: Optional ticker symbol for reporting
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check if dataframe is empty
    if df is None or df.empty:
        result['valid'] = False
        result['errors'].append(f"No data available for {ticker if ticker else 'unknown ticker'}")
        return result
        
    # Check index type
    if not isinstance(df.index, pd.DatetimeIndex):
        result['valid'] = False
        result['errors'].append(f"Index is not DatetimeIndex (got {type(df.index).__name__})")
        
    # Check for sufficient data points
    if len(df) < 100:
        result['valid'] = False
        result['errors'].append(f"Insufficient data points: {len(df)} (minimum 100)")
    
    # Find the price column
    price_col = None
    if 'Close' in df.columns:
        price_col = 'Close'
    elif 'price' in df.columns:
        price_col = 'price'
    elif len(df.columns) > 0:
        price_col = df.columns[0]
        result['warnings'].append(f"Using column '{price_col}' as price data")
    else:
        result['valid'] = False
        result['errors'].append("No columns found in dataframe")
        return result
    
    # Check for NaN values
    nan_count = df[price_col].isna().sum()
    nan_pct = (nan_count / len(df)) * 100
    
    if nan_count > 0:
        if nan_pct > 20:
            result['valid'] = False
            result['errors'].append(f"Too many NaN values: {nan_count} ({nan_pct:.1f}%)")
        else:
            result['warnings'].append(f"Contains {nan_count} NaN values ({nan_pct:.1f}%)")
    
    # Check data range
    if not df.index.empty:
        start_date = df.index[0]
        end_date = df.index[-1]
        date_range = (end_date - start_date).days
        
        result['info'].append(f"Date range: {start_date.date()} to {end_date.date()} ({date_range} days)")
        
        # Check freshness
        days_old = (pd.Timestamp.now() - end_date).days
        if days_old > 30:
            result['warnings'].append(f"Data is {days_old} days old")
    
    return result
