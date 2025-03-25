"""
Yahoo Finance Test Script

This script tests connectivity to Yahoo Finance and validates specific tickers.
Run this before the main portfolio optimization to check if your tickers are accessible.
"""

import yfinance as yf
import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta

def test_yahoo_connectivity():
    """Test general connectivity to Yahoo Finance"""
    print("\n===== Testing Yahoo Finance Connectivity =====")
    try:
        # Try to fetch a well-known ticker
        test_ticker = "AAPL"
        print(f"Fetching test ticker: {test_ticker}...")
        
        stock = yf.Ticker(test_ticker)
        data = stock.history(period="1d")
        
        if data.empty:
            print("❌ Error: Connected to Yahoo Finance but received empty data")
            return False
                
        print(f"✅ Successfully connected to Yahoo Finance API")
        print(f"   Latest price for {test_ticker}: ${data['Close'].iloc[-1]:.2f}")
        return True
            
    except Exception as e:
        print(f"❌ Failed to connect to Yahoo Finance: {str(e)}")
        return False

def test_specific_ticker(ticker, years=1):
    """Test if a specific ticker can be fetched from Yahoo Finance"""
    print(f"\n----- Testing ticker: {ticker} -----")
    
    # Format ticker if needed
    formatted_ticker = ticker
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    try:
        print(f"Fetching data for {formatted_ticker}...")
        stock = yf.Ticker(formatted_ticker)
        
        # Try to get basic info
        try:
            info = stock.info
            if 'regularMarketPrice' in info:
                print(f"✅ Current price: ${info['regularMarketPrice']:.2f}")
            else:
                print("⚠️ No current price available")
        except Exception as e:
            print(f"⚠️ Could not retrieve ticker info: {str(e)}")
        
        # Try to get historical prices
        prices = stock.history(start=start_date, end=end_date)['Close']
        
        if prices.empty:
            print("❌ No historical price data available")
            return False
        
        print(f"✅ Found {len(prices)} days of price data")
        print(f"   Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${prices.min():.2f} to ${prices.max():.2f}")
        
        # Try to get dividend data
        dividends = stock.dividends
        if not dividends.empty:
            print(f"✅ Found {len(dividends)} dividend payments")
            print(f"   Latest dividend: ${dividends.iloc[-1]:.4f}")
        else:
            print("ℹ️ No dividend data available")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fetch data for {ticker}: {str(e)}")
        return False

def load_and_test_tickers(csv_path, test_count=None):
    """Load tickers from CSV and test each one"""
    if not os.path.exists(csv_path):
        print(f"❌ Tickers file not found: {csv_path}")
        return
    
    try:
        # Try to detect delimiter
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            if ';' in first_line:
                delimiter = ';'
                print(f"Detected semicolon delimiter in {csv_path}")
            elif ',' in first_line:
                delimiter = ','
                print(f"Detected comma delimiter in {csv_path}")
            else:
                delimiter = ';'  # Default
                print(f"Using default semicolon delimiter for {csv_path}")
        
        df = pd.read_csv(csv_path, sep=delimiter)
        
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
        print(f"📋 Found {len(tickers)} tickers in the CSV file")
        
        # Test all or a subset of tickers
        test_tickers = tickers[:test_count] if test_count else tickers
        
        success_count = 0
        for ticker in test_tickers:
            if ticker in ['CASH', 'TBILLS']:
                print(f"\n----- {ticker}: Special asset, skipping test -----")
                success_count += 1
                continue
                
            success = test_specific_ticker(ticker)
            if success:
                success_count += 1
            
            # Add a delay between requests
            if ticker != test_tickers[-1]:
                time.sleep(1)
        
        print(f"\n===== Results: {success_count}/{len(test_tickers)} tickers passed the test =====")
        
    except Exception as e:
        print(f"❌ Error processing tickers file: {str(e)}")

def main():
    # Test general connectivity first
    if not test_yahoo_connectivity():
        print("\n❌ Failed basic Yahoo Finance connectivity test. Please check your internet connection.")
        return
    
    # Get CSV file path
    csv_path = os.path.join(os.path.dirname(__file__), 'tickers.csv')
    
    # Process command line args
    if len(sys.argv) > 1:
        # If specific tickers are provided as arguments
        for ticker in sys.argv[1:]:
            test_specific_ticker(ticker)
    else:
        # Otherwise test all tickers in the CSV
        load_and_test_tickers(csv_path)

if __name__ == "__main__":
    main()
