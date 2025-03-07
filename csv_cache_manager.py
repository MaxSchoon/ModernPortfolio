"""
CSV Cache Manager for Yahoo Finance Data

This module provides CSV-based caching functionality to store and retrieve financial data,
making it easier to debug and inspect the cached data.
"""

import os
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import glob

class CSVDataCache:
    """Manages caching of financial data in CSV format for easy inspection and debugging"""
    
    def __init__(self, cache_dir: str = "csv_data_cache"):
        """
        Initialize the CSV cache manager
        
        Parameters:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.price_dir = os.path.join(cache_dir, "prices")
        self.div_dir = os.path.join(cache_dir, "dividends")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        
        # Create cache directories if they don't exist
        for directory in [self.cache_dir, self.price_dir, self.div_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
        # Initialize or load metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "last_updated": {},
                "cache_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def get_ticker_filename(self, ticker: str) -> str:
        """Convert ticker to a valid filename"""
        # Replace characters that are invalid in filenames
        return ticker.replace('.', '_').replace('-', '_').replace(':', '_')
    
    def save_price_data(self, ticker: str, data: pd.Series) -> None:
        """Save price data for a ticker to CSV cache"""
        # Validate data before saving
        if data.empty:
            print(f"⚠️ Not caching empty data for {ticker}")
            return
            
        nan_pct = data.isna().mean() * 100
        if nan_pct > 80:  # If more than 80% is NaN, don't cache
            print(f"⚠️ Not caching {ticker} - poor data quality ({nan_pct:.1f}% NaN)")
            return
        
        # Fill NaN values if there are just a few
        if nan_pct > 0:
            print(f"ℹ️ Filling {nan_pct:.1f}% NaN values in {ticker} data")
            data = data.ffill().bfill()
        
        try:
            # Convert to DataFrame for better CSV storage
            df = pd.DataFrame({'price': data})
            
            # Save to CSV file
            filename = self.get_ticker_filename(ticker)
            filepath = os.path.join(self.price_dir, f"{filename}.csv")
            df.to_csv(filepath)
            
            # Update metadata
            self.metadata["last_updated"][ticker] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_metadata()
            
            print(f"✅ Cached price data for {ticker} to {filepath} ({len(data)} points)")
        except Exception as e:
            print(f"❌ Error saving CSV cache for {ticker}: {str(e)}")
    
    def save_div_data(self, ticker: str, data: pd.Series) -> None:
        """Save dividend data for a ticker to CSV cache"""
        try:
            # Convert to DataFrame for better CSV storage
            df = pd.DataFrame({'dividend': data})
            
            # Save to CSV file
            filename = self.get_ticker_filename(ticker)
            filepath = os.path.join(self.div_dir, f"{filename}.csv")
            df.to_csv(filepath)
            
            # Count non-zero dividends
            div_count = (data > 0).sum()
            if div_count > 0:
                print(f"✅ Cached {div_count} dividend payments for {ticker}")
        except Exception as e:
            print(f"❌ Error saving dividend data for {ticker}: {str(e)}")
    
    def get_price_data(self, ticker: str, max_age_days: int = 7) -> Optional[pd.Series]:
        """
        Get price data for ticker from CSV cache if available and not too old
        
        Parameters:
            ticker: The ticker symbol
            max_age_days: Maximum age of cached data in days
            
        Returns:
            Cached price data or None if not available or too old
        """
        # Check if data exists and isn't too old
        if ticker in self.metadata.get("last_updated", {}):
            last_update = datetime.strptime(self.metadata["last_updated"][ticker], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_update).days > max_age_days:
                print(f"⚠️ Cached data for {ticker} is over {max_age_days} days old")
                return None
                
            # Try to load data from CSV
            try:
                filename = self.get_ticker_filename(ticker)
                filepath = os.path.join(self.price_dir, f"{filename}.csv")
                
                if os.path.exists(filepath):
                    # Enhanced CSV loading with more robust error handling
                    try:
                        # First attempt - standard parsing
                        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    except (pd.errors.ParserError, ValueError) as e:
                        print(f"⚠️ Error parsing {ticker} CSV with standard method: {str(e)}")
                        print("Attempting alternative parsing...")
                        
                        # Alternative parsing - load raw then convert
                        raw_df = pd.read_csv(filepath)
                        date_col = raw_df.columns[0]
                        
                        # Convert index to datetime with flexible parsing
                        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
                        
                        # Set index and convert data column
                        raw_df.set_index(date_col, inplace=True)
                        if 'price' in raw_df.columns:
                            raw_df['price'] = pd.to_numeric(raw_df['price'], errors='coerce')
                        
                        df = raw_df
                    
                    # Check if we have price column and sufficient data
                    if 'price' in df.columns and len(df) > 10:
                        # Validate data quality
                        nan_count = df['price'].isna().sum()
                        nan_pct = (nan_count / len(df)) * 100
                        
                        if nan_pct > 80:
                            print(f"⚠️ {ticker} CSV data contains {nan_pct:.1f}% NaN values - unusable")
                            return None
                            
                        if nan_pct > 0:
                            print(f"ℹ️ {ticker} CSV data contains {nan_pct:.1f}% NaN values - will fill")
                            # Fill NaN values
                            df['price'] = df['price'].ffill().bfill()
                        
                        print(f"📊 Using CSV cached data for {ticker} from {last_update.strftime('%Y-%m-%d')}")
                        return df['price']
                    else:
                        print(f"⚠️ Cached data for {ticker} has invalid format or insufficient points")
                else:
                    print(f"⚠️ No CSV cache file found for {ticker}")
            except Exception as e:
                print(f"❌ Error reading CSV cache for {ticker}: {str(e)}")
                
        return None
    
    def get_div_data(self, ticker: str, max_age_days: int = 30) -> Optional[pd.Series]:
        """Get dividend data for ticker from CSV cache if available"""
        # Check if data exists and isn't too old
        if ticker in self.metadata.get("last_updated", {}):
            last_update = datetime.strptime(self.metadata["last_updated"][ticker], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_update).days > max_age_days:
                return None
                
            # Try to load data from CSV
            try:
                filename = self.get_ticker_filename(ticker)
                filepath = os.path.join(self.div_dir, f"{filename}.csv")
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    
                    if 'dividend' in df.columns:
                        return df['dividend']
            except Exception as e:
                print(f"❌ Error reading dividend CSV cache for {ticker}: {str(e)}")
                
        return None
    
    def _save_metadata(self) -> None:
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving metadata: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        # Delete all CSV files in price directory
        for file in glob.glob(os.path.join(self.price_dir, "*.csv")):
            os.remove(file)
            
        # Delete all CSV files in dividend directory
        for file in glob.glob(os.path.join(self.div_dir, "*.csv")):
            os.remove(file)
        
        # Reset metadata
        self.metadata = {
            "last_updated": {},
            "cache_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_metadata()
        
        print("🧹 CSV cache cleared successfully")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get status information about the cache"""
        # Count CSV files in price directory
        price_files = glob.glob(os.path.join(self.price_dir, "*.csv"))
        price_ticker_count = len(price_files)
        
        # Count CSV files in dividend directory
        div_files = glob.glob(os.path.join(self.div_dir, "*.csv"))
        div_ticker_count = len(div_files)
        
        # Calculate total size of all cached files
        price_cache_size = sum(os.path.getsize(f) for f in price_files) / (1024*1024) if price_files else 0
        div_cache_size = sum(os.path.getsize(f) for f in div_files) / (1024*1024) if div_files else 0
        
        # Get tickers from filenames
        price_tickers = [os.path.splitext(os.path.basename(f))[0] for f in price_files]
        
        # Find oldest and newest data
        oldest = None
        newest = None
        if self.metadata.get("last_updated"):
            dates = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") 
                     for d in self.metadata["last_updated"].values()]
            if dates:
                oldest = min(dates).strftime("%Y-%m-%d")
                newest = max(dates).strftime("%Y-%m-%d")
        
        return {
            "price_cache_size_mb": round(price_cache_size, 2),
            "price_ticker_count": price_ticker_count,
            "price_tickers": [self._filename_to_ticker(t) for t in price_tickers[:5]] if price_tickers else [],
            "div_cache_size_mb": round(div_cache_size, 2),
            "div_ticker_count": div_ticker_count,
            "oldest_data": oldest,
            "newest_data": newest,
            "cache_created": self.metadata.get("cache_created")
        }
    
    def _filename_to_ticker(self, filename: str) -> str:
        """Convert a filename back to a ticker symbol (best effort)"""
        # This is a best-effort conversion and may not be perfect for all cases
        return filename.replace('_', '.')
        
    def convert_from_pickle_cache(self, pickle_cache_dir: str = "data_cache") -> int:
        """
        Convert data from pickle cache to CSV cache
        
        Returns:
            Number of tickers successfully converted
        """
        import pickle
        
        print(f"Converting from pickle cache in {pickle_cache_dir} to CSV cache in {self.cache_dir}...")
        
        pickle_price_file = os.path.join(pickle_cache_dir, "price_data.pkl")
        pickle_div_file = os.path.join(pickle_cache_dir, "div_data.pkl")
        pickle_metadata_file = os.path.join(pickle_cache_dir, "cache_metadata.json")
        
        # Check if old cache exists
        if not os.path.exists(pickle_price_file):
            print("❌ No pickle cache found to convert")
            return 0
        
        # Load pickle cache
        try:
            with open(pickle_price_file, 'rb') as f:
                price_data = pickle.load(f)
                
            converted_count = 0
            
            # Save each ticker's data to CSV
            for ticker, data in price_data.items():
                if not data.empty:
                    self.save_price_data(ticker, data)
                    converted_count += 1
            
            # Try to load and convert dividend data
            if os.path.exists(pickle_div_file):
                with open(pickle_div_file, 'rb') as f:
                    div_data = pickle.load(f)
                
                for ticker, data in div_data.items():
                    if ticker in price_data and not data.empty:
                        self.save_div_data(ticker, data)
            
            # Copy metadata
            if os.path.exists(pickle_metadata_file):
                with open(pickle_metadata_file, 'r') as f:
                    old_metadata = json.load(f)
                
                # Copy last_updated info
                for ticker, date in old_metadata.get("last_updated", {}).items():
                    if ticker in price_data:
                        self.metadata["last_updated"][ticker] = date
                
                # Update cache creation date if available
                if "cache_created" in old_metadata:
                    self.metadata["cache_created"] = old_metadata["cache_created"]
                
                self._save_metadata()
            
            print(f"✅ Successfully converted {converted_count} tickers from pickle to CSV cache")
            return converted_count
            
        except Exception as e:
            print(f"❌ Error converting from pickle cache: {str(e)}")
            return 0
    
    def inspect_ticker_data(self, ticker: str) -> None:
        """
        Print detailed information about a cached ticker
        
        This is useful for debugging data issues
        """
        print(f"\n===== Inspecting Cache for {ticker} =====")
        
        # Check if ticker exists in metadata
        if ticker not in self.metadata.get("last_updated", {}):
            print(f"❌ No cache metadata found for {ticker}")
            return
            
        last_update = self.metadata["last_updated"][ticker]
        print(f"Last updated: {last_update}")
        
        # Check price data
        filename = self.get_ticker_filename(ticker)
        price_filepath = os.path.join(self.price_dir, f"{filename}.csv")
        
        if os.path.exists(price_filepath):
            print(f"Price data file: {price_filepath}")
            print(f"File size: {os.path.getsize(price_filepath)/1024:.2f} KB")
            
            try:
                df = pd.read_csv(price_filepath, index_col=0, parse_dates=True)
                print(f"Data points: {len(df)}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"NaN values: {df['price'].isna().sum()} ({df['price'].isna().mean()*100:.1f}%)")
                print(f"Value range: {df['price'].min():.2f} to {df['price'].max():.2f}")
                
                # Print first and last few rows
                print("\nFirst 5 rows:")
                print(df.head().to_string())
                print("\nLast 5 rows:")
                print(df.tail().to_string())
            except Exception as e:
                print(f"❌ Error reading price data: {str(e)}")
        else:
            print(f"❌ No price data file found for {ticker}")
        
        # Check dividend data
        div_filepath = os.path.join(self.div_dir, f"{filename}.csv")
        
        if os.path.exists(div_filepath):
            print(f"\nDividend data file: {div_filepath}")
            try:
                df = pd.read_csv(div_filepath, index_col=0, parse_dates=True)
                nonzero_divs = (df['dividend'] > 0).sum()
                print(f"Dividend payments: {nonzero_divs}")
                
                if nonzero_divs > 0:
                    divs_only = df[df['dividend'] > 0]
                    print("\nMost recent dividends:")
                    print(divs_only.tail().to_string())
            except Exception as e:
                print(f"❌ Error reading dividend data: {str(e)}")
        else:
            print(f"❌ No dividend data file found for {ticker}")
