"""
NaN Issue Fixer for Portfolio Data

This script diagnoses and fixes the NaN value issue when loading stock data
from CSV cache files.
"""

import os
import pandas as pd
import glob
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import sys

def inspect_csv_file(filepath):
    """Inspect a CSV file to diagnose issues"""
    print(f"\nInspecting file: {filepath}")
    
    try:
        # Try reading without parsing dates first to see raw data
        raw_df = pd.read_csv(filepath)
        print("\nRaw data (first 5 rows):")
        print(raw_df.head())
        
        # Check column names
        print(f"\nColumns: {raw_df.columns.tolist()}")
        
        # Check for date/index column type
        index_col = raw_df.columns[0]  # Usually the first column is the index/date
        print(f"\nIndex column '{index_col}' data type: {raw_df[index_col].dtype}")
        
        # Try to parse with correct settings
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print("\nParsed data with index_col=0, parse_dates=True (first 5 rows):")
        print(df.head())
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        nan_pct = df.isna().mean().mean() * 100
        print(f"\nNaN values in parsed data: {nan_count} ({nan_pct:.2f}%)")
        
        # Check data type of values
        if 'price' in df.columns:
            print(f"\nPrice column data type: {df['price'].dtype}")
            print(f"Price range: {df['price'].min()} to {df['price'].max()}")
        
        return df
    except Exception as e:
        print(f"Error inspecting file: {type(e).__name__}: {str(e)}")
        return None

def fix_csv_file(filepath, output_filepath=None):
    """Fix issues in a CSV file and save a corrected version"""
    if output_filepath is None:
        # Create a new filename with _fixed suffix
        base, ext = os.path.splitext(filepath)
        output_filepath = f"{base}_fixed{ext}"
    
    try:
        print(f"\nFixing file: {filepath}")
        print(f"Output will be saved to: {output_filepath}")
        
        # Read the file with flexible parsing
        df = pd.read_csv(filepath)
        
        # Ensure the first column is interpreted as dates
        date_col = df.columns[0]
        
        # Convert date strings to datetime objects
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            print(f"Error converting dates: {type(e).__name__}: {str(e)}")
            print("Attempting to fix date format...")
            
            # Try multiple date formats
            for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                    if not df[date_col].isna().all():
                        print(f"Successfully parsed dates using format: {date_format}")
                        break
                except:
                    continue
        
        # Set the date column as index
        df.set_index(date_col, inplace=True)
        
        # Ensure numeric columns are properly typed
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
        if 'dividend' in df.columns:
            df['dividend'] = pd.to_numeric(df['dividend'], errors='coerce')
        
        # Sort index to ensure chronological order
        df.sort_index(inplace=True)
        
        # Fill any NaN values
        if 'price' in df.columns:
            nan_before = df['price'].isna().sum()
            if nan_before > 0:
                df['price'] = df['price'].ffill().bfill()
                print(f"Filled {nan_before} NaN values in price column")
        
        # Save fixed file
        df.to_csv(output_filepath)
        
        # Validate the fix
        fixed_df = pd.read_csv(output_filepath, index_col=0, parse_dates=True)
        if 'price' in fixed_df.columns:
            nan_after = fixed_df['price'].isna().sum()
            print(f"NaN values after fixing: {nan_after}")
            
        print(f"✅ File fixed and saved to {output_filepath}")
        return True
    except Exception as e:
        print(f"Error fixing file: {type(e).__name__}: {str(e)}")
        return False

def fix_csv_cache(cache_dir="csv_data_cache"):
    """Fix all CSV files in the cache directory"""
    prices_dir = os.path.join(cache_dir, "prices")
    dividends_dir = os.path.join(cache_dir, "dividends")
    
    if not os.path.exists(prices_dir):
        print(f"❌ Price directory not found: {prices_dir}")
        return
    
    # Create backup directory
    backup_dir = os.path.join(cache_dir, "backup_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory: {backup_dir}")
    
    # Process price files
    price_files = glob.glob(os.path.join(prices_dir, "*.csv"))
    print(f"\nFound {len(price_files)} price CSV files")
    
    fixed_count = 0
    for i, filepath in enumerate(price_files):
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n[{i+1}/{len(price_files)}] Processing {ticker}")
        
        # Create backup
        backup_file = os.path.join(backup_dir, os.path.basename(filepath))
        import shutil
        shutil.copy2(filepath, backup_file)
        
        # Fix and save back to original location
        if fix_csv_file(filepath, filepath):
            fixed_count += 1
    
    # Process dividend files if they exist
    if os.path.exists(dividends_dir):
        div_files = glob.glob(os.path.join(dividends_dir, "*.csv"))
        print(f"\nFound {len(div_files)} dividend CSV files")
        
        for i, filepath in enumerate(div_files):
            ticker = os.path.splitext(os.path.basename(filepath))[0]
            print(f"\n[{i+1}/{len(div_files)}] Processing {ticker} dividends")
            
            # Create backup
            backup_file = os.path.join(backup_dir, os.path.basename(filepath))
            import shutil
            shutil.copy2(filepath, backup_file)
            
            # Fix and save back to original location
            if fix_csv_file(filepath, filepath):
                fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} CSV files")
    print(f"Backups saved to: {backup_dir}")
    print("\nNow you can run ModernPortfolio.py again and it should properly load the data.")

def plot_price_data(filepath):
    """Plot price data from a CSV file to visualize it"""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if 'price' not in df.columns:
            print("No 'price' column found in the CSV file")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['price'])
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        plt.title(f"{ticker} Price History")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting data: {type(e).__name__}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Fix NaN issues in CSV cache files")
    parser.add_argument('--cache-dir', default='csv_data_cache', help='Cache directory path')
    parser.add_argument('--inspect', '-i', type=str, help='Inspect a specific CSV file')
    parser.add_argument('--fix-file', '-f', type=str, help='Fix a specific CSV file')
    parser.add_argument('--fix-all', '-a', action='store_true', help='Fix all CSV files in the cache')
    parser.add_argument('--plot', '-p', type=str, help='Plot price data from a CSV file')
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_csv_file(args.inspect)
    elif args.fix_file:
        fix_csv_file(args.fix_file)
    elif args.plot:
        plot_price_data(args.plot)
    elif args.fix_all:
        fix_csv_cache(args.cache_dir)
    else:
        print("No action specified. Use --inspect, --fix-file, --plot, or --fix-all")
        parser.print_help()

if __name__ == "__main__":
    main()
