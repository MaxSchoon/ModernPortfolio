"""
CSV Data Cache Manager

This module provides caching functionality for financial data in CSV format.
It handles both regular ticker data and synthetic assets like cash and T-bills.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import glob
from typing import Any, Optional, Dict

class CSVDataCache:
    """
    A cache manager for financial data stored in CSV files
    """
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize the cache with the specified directory
        
        Parameters:
            cache_dir: Directory for storing CSV files
        """
        self.cache_dir = cache_dir
        self.price_dir = os.path.join(cache_dir, "prices")
        self.div_dir = os.path.join(cache_dir, "dividends")
        self.meta_dir = os.path.join(cache_dir, "metadata")
        self.info_dir = os.path.join(cache_dir, "info")
        
        # Create directories if they don't exist
        for directory in [self.cache_dir, self.price_dir, self.div_dir, self.meta_dir, self.info_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Default risk-free rate (can be overridden)
        self.risk_free_rate = 0.04
                
    def get_price_data(self, ticker: str, max_age_days: int = 7) -> Optional[pd.Series]:
        """
        Get price data for a ticker, either from cache or by generating synthetic data
        
        Parameters:
            ticker: Ticker symbol
            max_age_days: Maximum age of cached data in days
            
        Returns:
            pd.Series with price data or None if not available
        """
        # Special handling for synthetic assets
        if ticker in ['CASH', 'TBILLS']:
            return self._get_synthetic_asset_data(ticker)
            
        # For regular tickers, try to get from cache
        cache_file = os.path.join(self.price_dir, f"{ticker}_price.csv")
        
        if os.path.exists(cache_file):
            # Check file age
            file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
            
            if file_age <= max_age_days:
                try:
                    # Load data from cache
                    df = pd.read_csv(cache_file, index_col=0)
                    df.index = pd.to_datetime(df.index)
                    return df['Close']
                except Exception as e:
                    print(f"Error loading cached price data for {ticker}: {str(e)}")
        
        # If we get here, either the file doesn't exist, is too old, or couldn't be loaded
        return None
        
    def get_div_data(self, ticker: str, max_age_days: int = 7) -> Optional[pd.Series]:
        """
        Get dividend data for a ticker from cache
        
        Parameters:
            ticker: Ticker symbol
            max_age_days: Maximum age of cached data in days
            
        Returns:
            pd.Series with dividend data or None if not available
        """
        # Synthetic assets don't pay dividends
        if ticker in ['CASH', 'TBILLS']:
            price_data = self._get_synthetic_asset_data(ticker)
            if price_data is not None:
                return pd.Series(0.0, index=price_data.index)
            return None
            
        # For regular tickers, try to get from cache
        cache_file = os.path.join(self.div_dir, f"{ticker}_div.csv")
        
        if os.path.exists(cache_file):
            # Check file age
            file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
            
            if file_age <= max_age_days:
                try:
                    # Load data from cache
                    df = pd.read_csv(cache_file, index_col=0)
                    df.index = pd.to_datetime(df.index)
                    return df['Dividend']
                except Exception as e:
                    print(f"Error loading cached dividend data for {ticker}: {str(e)}")
        
        # If we get here, either the file doesn't exist, is too old, or couldn't be loaded
        return None
    
    def save_price_data(self, ticker: str, price_data: pd.Series) -> bool:
        """
        Save price data to cache
        
        Parameters:
            ticker: Ticker symbol
            price_data: Series of price data
            
        Returns:
            Success status
        """
        try:
            cache_file = os.path.join(self.price_dir, f"{ticker}_price.csv")
            
            # Make sure data has the right format
            df = pd.DataFrame({'Close': price_data})
            df.index.name = 'Date'
            
            # Save to file
            df.to_csv(cache_file)
            
            # Update metadata
            self._update_metadata(ticker, 'price', len(price_data))
            
            return True
        except Exception as e:
            print(f"Error saving price data for {ticker}: {str(e)}")
            return False
    
    def save_div_data(self, ticker: str, div_data: pd.Series) -> bool:
        """
        Save dividend data to cache
        
        Parameters:
            ticker: Ticker symbol
            div_data: Series of dividend data
            
        Returns:
            Success status
        """
        try:
            cache_file = os.path.join(self.div_dir, f"{ticker}_div.csv")
            
            # Make sure data has the right format
            df = pd.DataFrame({'Dividend': div_data})
            df.index.name = 'Date'
            
            # Save to file
            df.to_csv(cache_file)
            
            # Update metadata
            self._update_metadata(ticker, 'dividend', len(div_data))
            
            return True
        except Exception as e:
            print(f"Error saving dividend data for {ticker}: {str(e)}")
            return False
    
    def save_info_data(self, ticker: str, info_data: Dict[str, Any]) -> bool:
        """
        Save ticker info data to cache
        
        Parameters:
            ticker: Ticker symbol
            info_data: Dictionary containing ticker info
            
        Returns:
            Success status
        """
        try:
            info_file = os.path.join(self.info_dir, f"{ticker}_info.json")
            
            # Filter out any non-serializable objects
            filtered_info = {}
            for key, value in info_data.items():
                try:
                    # Test JSON serialization
                    json.dumps({key: value})
                    filtered_info[key] = value
                except (TypeError, OverflowError):
                    # Skip non-serializable values
                    pass
            
            # Save to file
            with open(info_file, 'w') as f:
                json.dump(filtered_info, f, indent=2)
            
            # Update metadata
            self._update_metadata(ticker, 'info', len(filtered_info))
            
            return True
        except Exception as e:
            print(f"Error saving info data for {ticker}: {str(e)}")
            return False
    
    def _get_synthetic_asset_data(self, ticker: str, years: int = 5) -> pd.Series:
        """
        Get data for synthetic assets like CASH or TBILLS
        
        Parameters:
            ticker: Ticker symbol ('CASH' or 'TBILLS')
            years: Number of years of data to generate
            
        Returns:
            pd.Series with synthetic price data
        """
        cache_file = os.path.join(self.price_dir, f"{ticker}_price.csv")
        
        if os.path.exists(cache_file):
            try:
                # Load existing synthetic data
                df = pd.read_csv(cache_file, index_col=0)
                df.index = pd.to_datetime(df.index)
                
                # Check if we need to extend the data to the current date
                last_date = df.index.max()
                today = pd.Timestamp.now()
                
                if last_date < today - timedelta(days=2):  # If data is more than 2 days old
                    # Generate and append new data
                    extended_data = self._generate_synthetic_data(
                        ticker, 
                        start_date=last_date + timedelta(days=1), 
                        end_date=today
                    )
                    
                    # Combine existing and new data
                    combined_data = pd.concat([df['Close'], extended_data])
                    
                    # Save the updated data
                    self.save_price_data(ticker, combined_data)
                    
                    return combined_data
                    
                return df['Close']
                
            except Exception as e:
                print(f"Error loading synthetic data for {ticker}: {str(e)}")
                # Fall back to generating new data
        
        # If we get here, either the file doesn't exist, couldn't be loaded,
        # or we need to generate new data
        
        # Generate synthetic data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        synthetic_data = self._generate_synthetic_data(ticker, start_date, end_date)
        
        # Save for future use
        self.save_price_data(ticker, synthetic_data)
        
        # Also create and save empty dividend data
        div_data = pd.Series(0.0, index=synthetic_data.index)
        self.save_div_data(ticker, div_data)
        
        return synthetic_data
    
    def _generate_synthetic_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Generate synthetic price data for CASH or TBILLS
        
        Parameters:
            ticker: 'CASH' or 'TBILLS'
            start_date: Starting date
            end_date: Ending date
            
        Returns:
            Series of synthetic prices
        """
        # Use business days frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Set rate and variance based on asset type
        if ticker == 'CASH':
            rate = self.risk_free_rate
            variance = 1e-8  # Minimal variance
        else:  # TBILLS
            rate = self.risk_free_rate * 1.02  # Slightly higher than cash
            variance = 1e-6  # Slightly more volatile than cash
        
        # Generate daily returns
        daily_return = (1 + rate) ** (1/252) - 1
        
        # Generate prices with compounding returns
        days = len(date_range)
        prices = np.ones(days)
        
        # Apply compounding with tiny random noise
        for i in range(1, days):
            # Add a small amount of noise to simulate minor fluctuations
            daily_noise = np.random.normal(0, np.sqrt(variance))
            prices[i] = prices[i-1] * (1 + daily_return + daily_noise)
        
        # Create a pandas series with the generated prices
        return pd.Series(prices, index=date_range)
    
    def _update_metadata(self, ticker: str, data_type: str, data_length: int) -> None:
        """Update metadata for a ticker"""
        meta_file = os.path.join(self.meta_dir, f"{ticker}_meta.json")
        
        # Read existing metadata if available
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        else:
            metadata = {}
        
        # Update metadata
        metadata['last_updated'] = datetime.now().isoformat()
        metadata[f'{data_type}_length'] = data_length
        
        # Write back to file
        try:
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error updating metadata for {ticker}: {str(e)}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get status information about the cache"""
        # Count price files
        price_files = glob.glob(os.path.join(self.price_dir, '*_price.csv'))
        div_files = glob.glob(os.path.join(self.div_dir, '*_div.csv'))
        
        # Calculate total size
        price_size = sum(os.path.getsize(f) for f in price_files if os.path.exists(f)) / (1024*1024)
        div_size = sum(os.path.getsize(f) for f in div_files if os.path.exists(f)) / (1024*1024)
        
        # Find oldest and newest data
        dates = []
        for file in price_files:
            try:
                mtime = os.path.getmtime(file)
                dates.append(datetime.fromtimestamp(mtime))
            except:
                continue
        
        oldest = min(dates) if dates else None
        newest = max(dates) if dates else None
        
        return {
            'price_ticker_count': len(price_files),
            'div_ticker_count': len(div_files),
            'price_cache_size_mb': price_size,
            'div_cache_size_mb': div_size,
            'total_cache_size_mb': price_size + div_size,
            'oldest_data': oldest.date().isoformat() if oldest else 'N/A',
            'newest_data': newest.date().isoformat() if newest else 'N/A'
        }
    
    def clear_cache(self) -> None:
        """Clear all cache files"""
        price_files = glob.glob(os.path.join(self.price_dir, '*_price.csv'))
        div_files = glob.glob(os.path.join(self.div_dir, '*_div.csv'))
        meta_files = glob.glob(os.path.join(self.meta_dir, '*_meta.json'))
        
        # Delete all files
        for file in price_files + div_files + meta_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {str(e)}")
        
        print(f"Cache cleared: {len(price_files)} price files, {len(div_files)} dividend files")

# Usage example (if run directly)
if __name__ == "__main__":
    cache = CSVDataCache()
    
    # Print cache status
    status = cache.get_cache_status()
    print("Cache Status:")
    print(f"- Price data: {status['price_ticker_count']} tickers ({status['price_cache_size_mb']:.2f} MB)")
    print(f"- Dividend data: {status['div_ticker_count']} tickers ({status['div_cache_size_mb']:.2f} MB)")
    print(f"- Date range: {status['oldest_data']} to {status['newest_data']}")
    
    # Generate synthetic data for CASH and TBILLS
    print("\nGenerating synthetic data:")
    cash_data = cache._get_synthetic_asset_data('CASH', years=5)
    tbills_data = cache._get_synthetic_asset_data('TBILLS', years=5)
    
    print(f"CASH: Generated {len(cash_data)} days of data")
    print(f"TBILLS: Generated {len(tbills_data)} days of data")
    
    # Verify the data
    print("\nSample data:")
    print(f"CASH first price: {cash_data.iloc[0]:.4f}, last price: {cash_data.iloc[-1]:.4f}")
    print(f"TBILLS first price: {tbills_data.iloc[0]:.4f}, last price: {tbills_data.iloc[-1]:.4f}")
