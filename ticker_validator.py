"""
Ticker Validator and Corrector

This script validates ticker symbols and fixes common format issues.
It can also check connectivity to Yahoo Finance for individual tickers.
"""

import pandas as pd
import yfinance as yf
import os
import argparse
import time
import sys
from typing import List, Dict, Tuple
from utils import format_ticker, is_valid_ticker_format  # Import utility functions

def validate_ticker(ticker: str, verbose: bool = True) -> Tuple[bool, str, str]:
    """
    Validate a ticker symbol against Yahoo Finance
    
    Returns:
        Tuple of (is_valid, corrected_ticker, message)
    """
    original_ticker = ticker
    
    # Use the utility function for formatting
    corrected_ticker = format_ticker(ticker)
    
    # If the ticker was corrected, note it
    if original_ticker != corrected_ticker:
        if verbose:
            print(f"🔄 Corrected {original_ticker} to {corrected_ticker}")
    
    # Check if the ticker exists
    try:
        stock = yf.Ticker(corrected_ticker)
        data = stock.history(period="1d")
        
        if data.empty:
            if verbose:
                print(f"❌ {original_ticker}: No data found for {corrected_ticker}")
            return False, corrected_ticker, "No data available"
        
        if 'Close' in data.columns:
            price = data['Close'].iloc[-1]
            if verbose:
                if original_ticker != corrected_ticker:
                    print(f"✅ {original_ticker}: Corrected to {corrected_ticker}, current price: ${price:.2f}")
                else:
                    print(f"✅ {original_ticker}: Valid ticker, current price: ${price:.2f}")
            return True, corrected_ticker, f"Valid ticker, price: ${price:.2f}"
        else:
            if verbose:
                print(f"❌ {original_ticker}: Data format issue")
            return False, corrected_ticker, "Data format issue"
    
    except Exception as e:
        if verbose:
            print(f"❌ {original_ticker}: Error - {str(e)}")
        return False, corrected_ticker, f"Error: {str(e)}"

def validate_csv_tickers(filepath: str, output_filepath: str = None) -> None:
    """
    Validate and correct tickers in a CSV file
    
    Parameters:
        filepath: Path to the input CSV file
        output_filepath: Path to save corrected CSV (if None, will use "tickers_corrected.csv")
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    if output_filepath is None:
        output_filepath = "tickers_corrected.csv"
    
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
            if col.lower() == 'ticker':
                ticker_column = col
                break
                
        if ticker_column is None:
            print(f"❌ CSV columns: {', '.join(df.columns)}")
            print("❌ CSV file must contain a 'ticker' column")
            return
            
        tickers = df[ticker_column].tolist()
        print(f"Found {len(tickers)} tickers in the CSV file")
        
        # Validate and correct each ticker
        corrected_tickers = []
        results = []
        
        print("\nValidating tickers (this may take a while)...")
        for ticker in tickers:
            # Skip special assets like CASH and TBILLS
            if ticker in ['CASH', 'TBILLS']:
                corrected_tickers.append(ticker)
                results.append(f"✅ {ticker}: Special asset")
                continue
                
            # Validate and correct ticker
            is_valid, corrected, message = validate_ticker(ticker, verbose=False)
            corrected_tickers.append(corrected)
            
            if is_valid:
                if ticker != corrected:
                    results.append(f"🔄 {ticker} -> {corrected}: {message}")
                else:
                    results.append(f"✅ {ticker}: {message}")
            else:
                results.append(f"❌ {ticker}: {message}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        # Update the dataframe with corrected tickers
        df[ticker_column] = corrected_tickers
        
        # Save the corrected CSV
        df.to_csv(output_filepath, sep=delimiter, index=False)
        print(f"\n✅ Saved corrected tickers to {output_filepath}")
        
        # Print results
        print("\nValidation Results:")
        for result in results:
            print(result)
            
    except Exception as e:
        print(f"❌ Error processing CSV file: {str(e)}")

def validate_ticker_list(tickers: List[str]) -> None:
    """Validate a list of tickers"""
    for ticker in tickers:
        validate_ticker(ticker)
        time.sleep(1)  # Add delay to avoid rate limiting

def main():
    parser = argparse.ArgumentParser(description="Validate and correct ticker symbols")
    parser.add_argument('tickers', nargs='*', help='List of tickers to validate')
    parser.add_argument('--file', '-f', help='CSV file containing tickers to validate')
    parser.add_argument('--output', '-o', help='Output file for corrected tickers')
    
    args = parser.parse_args()
    
    if args.tickers:
        validate_ticker_list(args.tickers)
    elif args.file:
        validate_csv_tickers(args.file, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
