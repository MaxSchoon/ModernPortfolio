"""
CSV Cache Inspector Tool

This tool allows inspecting and debugging the CSV cache data
"""

import os
import pandas as pd
import glob
import argparse
from csv_cache_manager import CSVDataCache

def list_cached_tickers(cache_dir: str = "csv_data_cache") -> None:
    """List all tickers in the CSV cache"""
    price_dir = os.path.join(cache_dir, "prices")
    
    if not os.path.exists(price_dir):
        print(f"❌ Cache directory not found: {price_dir}")
        return
    
    files = glob.glob(os.path.join(price_dir, "*.csv"))
    tickers = [os.path.splitext(os.path.basename(f))[0] for f in files]
    
    print(f"\nFound {len(tickers)} tickers in cache:")
    
    # Sort and display tickers in columns
    tickers.sort()
    cols = 4
    for i in range(0, len(tickers), cols):
        row = tickers[i:i+cols]
        print("  ".join(ticker.ljust(10) for ticker in row))

def inspect_ticker(ticker: str, cache_dir: str = "csv_data_cache") -> None:
    """Inspect detailed data for a specific ticker"""
    cache = CSVDataCache(cache_dir)
    cache.inspect_ticker_data(ticker)

def check_all_tickers_data_quality(cache_dir: str = "csv_data_cache") -> None:
    """Check data quality for all cached tickers"""
    price_dir = os.path.join(cache_dir, "prices")
    
    if not os.path.exists(price_dir):
        print(f"❌ Cache directory not found: {price_dir}")
        return
    
    files = glob.glob(os.path.join(price_dir, "*.csv"))
    
    print(f"\nChecking data quality for {len(files)} cached tickers...")
    
    good_tickers = []
    bad_tickers = []
    
    for file in files:
        ticker = os.path.splitext(os.path.basename(file))[0]
        
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            
            # Check for NaN values
            nan_count = df['price'].isna().sum()
            nan_pct = nan_count / len(df) * 100
            
            if nan_pct > 10:
                bad_tickers.append((ticker, f"{nan_pct:.1f}% NaN"))
            elif len(df) < 100:
                bad_tickers.append((ticker, f"Only {len(df)} data points"))
            else:
                good_tickers.append(ticker)
                
        except Exception as e:
            bad_tickers.append((ticker, f"Error: {str(e)}"))
    
    print(f"\n✅ Good data: {len(good_tickers)} tickers")
    print(f"❌ Problematic data: {len(bad_tickers)} tickers")
    
    if bad_tickers:
        print("\nProblematic tickers:")
        for ticker, issue in bad_tickers:
            print(f"- {ticker}: {issue}")

def plot_ticker(ticker: str, cache_dir: str = "csv_data_cache") -> None:
    """Plot price data for a ticker"""
    try:
        import matplotlib.pyplot as plt
        
        cache = CSVDataCache(cache_dir)
        filename = cache.get_ticker_filename(ticker)
        filepath = os.path.join(cache_dir, "prices", f"{filename}.csv")
        
        if not os.path.exists(filepath):
            print(f"❌ No data found for {ticker}")
            return
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['price'])
        plt.title(f"{ticker} Price History")
        plt.ylabel("Price ($)")
        plt.grid(True)
        
        # Add info about first/last dates and values
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        first_price = df['price'].iloc[0]
        last_price = df['price'].iloc[-1]
        
        plt.annotate(f"Start: ${first_price:.2f}", 
                     xy=(df.index[0], first_price),
                     xytext=(10, 10), textcoords='offset points')
        
        plt.annotate(f"End: ${last_price:.2f}", 
                     xy=(df.index[-1], last_price),
                     xytext=(-40, 10), textcoords='offset points')
        
        plt.tight_layout()
        
        # Save to file
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file)
        
        print(f"✅ Plot saved to {output_file}")
        plt.show()
        
    except ImportError:
        print("❌ Matplotlib not installed. Install it with 'pip install matplotlib'")
    except Exception as e:
        print(f"❌ Error plotting data: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Inspect CSV cache data")
    parser.add_argument('--list', '-l', action='store_true', help='List all cached tickers')
    parser.add_argument('--ticker', '-t', type=str, help='Inspect a specific ticker')
    parser.add_argument('--check-all', '-c', action='store_true', help='Check data quality for all tickers')
    parser.add_argument('--plot', '-p', type=str, help='Plot price data for a ticker')
    parser.add_argument('--cache-dir', type=str, default='csv_data_cache', help='Cache directory')
    
    args = parser.parse_args()
    
    # Check if cache exists
    if not os.path.exists(args.cache_dir):
        print(f"❌ Cache directory not found: {args.cache_dir}")
        return
    
    # Execute requested action
    if args.list:
        list_cached_tickers(args.cache_dir)
    elif args.ticker:
        inspect_ticker(args.ticker, args.cache_dir)
    elif args.check_all:
        check_all_tickers_data_quality(args.cache_dir)
    elif args.plot:
        plot_ticker(args.plot, args.cache_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
