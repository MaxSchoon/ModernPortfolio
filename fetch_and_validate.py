"""
Ticker Fetching and Validation Tool

This script performs comprehensive validation of tickers and pre-fetches data.
It combines ticker validation, data fetching, and cache management.
"""

import yfinance as yf
import pandas as pd
import os
import argparse
import time
import sys
from typing import List, Dict, Tuple
from csv_cache_manager import CSVDataCache
from ticker_validator import validate_ticker
from utils import format_ticker  # Import the utility function

def validate_and_fetch(ticker: str, cache: CSVDataCache, years: int = 5) -> Tuple[bool, str]:
    """
    Validate a ticker and fetch its data
    
    Returns:
        Tuple of (success, message)
    """
    # First validate the ticker
    is_valid, corrected_ticker, _ = validate_ticker(ticker, verbose=False)
    
    if not is_valid:
        return False, f"❌ {ticker} is not valid"
    
    # If the ticker was corrected, print a message
    if ticker != corrected_ticker:
        print(f"🔄 Corrected {ticker} to {corrected_ticker}")
        ticker = corrected_ticker
    
    # Use the utility function here if needed for additional formatting
    formatted_ticker = format_ticker(ticker)
    if formatted_ticker != ticker:
        print(f"🔄 Further formatted {ticker} to {formatted_ticker}")
        ticker = formatted_ticker
    
    try:
        print(f"Fetching data for {ticker}...")
        
        # Check cache first
        cached_data = cache.get_price_data(ticker)
        if cached_data is not None and len(cached_data) > 100:
            print(f"✅ {ticker}: Using cached data with {len(cached_data)} points")
            return True, f"Using cached data ({len(cached_data)} points)"
        
        # Fetch from Yahoo Finance
        stock = yf.Ticker(ticker)
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365*years)
        
        # Get price data
        prices = stock.history(start=start_date, end=end_date)['Close']
        
        if len(prices) < 100:
            return False, f"❌ {ticker}: Insufficient data points ({len(prices)})"
        
        # Normalize and save price data
        prices.index = prices.index.tz_localize(None)
        cache.save_price_data(ticker, prices)
        
        # Get dividend data
        try:
            dividends = stock.dividends
            if not dividends.empty:
                dividends.index = dividends.index.tz_localize(None)
                div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                common_dates = dividends.index.intersection(div_series.index)
                
                if common_dates.size > 0:
                    div_series.loc[common_dates] = dividends[dividends.index.isin(common_dates)].astype('float64')
                cache.save_div_data(ticker, div_series)
                
                return True, f"✅ {ticker}: Fetched {len(prices)} price points and {len(dividends)} dividends"
            else:
                div_series = pd.Series(0.0, index=prices.index, dtype='float64')
                cache.save_div_data(ticker, div_series)
                
                return True, f"✅ {ticker}: Fetched {len(prices)} price points (no dividends)"
        
        except Exception as e:
            # If dividend fetching fails, we can still use the price data
            div_series = pd.Series(0.0, index=prices.index, dtype='float64')
            cache.save_div_data(ticker, div_series)
            
            return True, f"✅ {ticker}: Fetched {len(prices)} price points (dividend error: {str(e)})"
    
    except Exception as e:
        return False, f"❌ {ticker}: Error fetching data - {str(e)}"

def process_csv_file(filepath: str, output_filepath: str = None, 
                    years: int = 5, delay: float = 1.0) -> None:
    """
    Process a CSV file containing tickers
    
    Parameters:
        filepath: Path to the input CSV file
        output_filepath: Path to save corrected CSV (if None, will use "tickers_corrected.csv")
        years: Number of years of historical data to fetch
        delay: Delay between API requests in seconds
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    if output_filepath is None:
        output_filepath = "tickers_corrected.csv"
    
    # Create cache manager
    cache = CSVDataCache()
    
    # Read the CSV file
    try:
        # Try to detect delimiter
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if ';' in first_line:
                delimiter = ';'
                print(f"Detected semicolon delimiter in {filepath}")
            elif ',' in first_line:
                delimiter = ','
                print(f"Detected comma delimiter in {filepath}")
            else:
                delimiter = ';'  # Default
                print(f"Using default semicolon delimiter for {filepath}")
        
        df = pd.read_csv(filepath, sep=delimiter)
        
        # Check column names (case insensitive)
        ticker_column = None
        for col in df.columns:
            if (col.lower() == 'ticker'):
                ticker_column = col
                break
                
        if ticker_column is None:
            print(f"❌ CSV columns: {', '.join(df.columns)}")
            print("❌ CSV file must contain a 'ticker' column")
            return
            
        tickers = df[ticker_column].tolist()
        print(f"Found {len(tickers)} tickers in the CSV file")
        
        # Add status column if it doesn't exist
        status_column = 'status'
        if status_column not in df.columns:
            df[status_column] = 'Unknown'
        
        # Process each ticker
        corrected_tickers = []
        results = []
        
        print("\nProcessing tickers (this may take a while)...")
        for i, ticker in enumerate(tickers):
            print(f"\nTicker {i+1}/{len(tickers)}: {ticker}")
            
            # Skip special assets like CASH and TBILLS
            if ticker in ['CASH', 'TBILLS']:
                corrected_tickers.append(ticker)
                df.loc[i, status_column] = 'Special asset'
                results.append(f"✅ {ticker}: Special asset")
                continue
                
            # Validate and fetch ticker data
            is_valid, corrected, _ = validate_ticker(ticker, verbose=False)
            if is_valid:
                success, message = validate_and_fetch(corrected, cache, years)
                if success:
                    corrected_tickers.append(corrected)
                    df.loc[i, status_column] = 'Fetched'
                    results.append(f"✅ {ticker} -> {corrected}: {message}")
                else:
                    corrected_tickers.append(corrected)
                    df.loc[i, status_column] = 'Fetch failed'
                    results.append(f"⚠️ {ticker} -> {corrected}: {message}")
            else:
                corrected_tickers.append(ticker)  # Keep original for invalid tickers
                df.loc[i, status_column] = 'Invalid'
                results.append(f"❌ {ticker}: Invalid ticker")
            
            # Add delay to avoid rate limiting
            if i < len(tickers) - 1:  # No need to delay after the last ticker
                print(f"Waiting {delay} seconds before next ticker...")
                time.sleep(delay)
        
        # Update the dataframe with corrected tickers
        df[ticker_column] = corrected_tickers
        
        # Save the corrected CSV
        df.to_csv(output_filepath, sep=delimiter, index=False)
        print(f"\n✅ Saved processed tickers to {output_filepath}")
        
        # Print results
        print("\nProcessing Results:")
        for result in results:
            print(result)
            
    except Exception as e:
        print(f"❌ Error processing CSV file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Validate tickers and fetch data")
    parser.add_argument('--file', '-f', help='CSV file containing tickers', default='tickers.csv')
    parser.add_argument('--output', '-o', help='Output file for processed tickers', default='tickers_corrected.csv')
    parser.add_argument('--years', '-y', type=int, default=5, help='Years of historical data to fetch')
    parser.add_argument('--delay', '-d', type=float, default=2.0, help='Delay between API requests in seconds')
    
    args = parser.parse_args()
    process_csv_file(args.file, args.output, args.years, args.delay)

if __name__ == "__main__":
    main()
