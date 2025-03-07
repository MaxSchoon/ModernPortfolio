"""
Cache Manager for Yahoo Finance Data

This module provides caching functionality to store and retrieve financial data,
reducing the number of API calls to Yahoo Finance and helping avoid rate limits.
"""

import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional, Tuple

class DataCache:
    """Manages caching of financial data to reduce API calls to Yahoo Finance"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize the cache manager
        
        Parameters:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.price_cache_file = os.path.join(cache_dir, "price_data.pkl")
        self.div_cache_file = os.path.join(cache_dir, "div_data.pkl")
        self.info_cache_file = os.path.join(cache_dir, "info_data.json")
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize or load metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "last_updated": {},
                "cache_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def save_price_data(self, ticker: str, data: pd.Series) -> None:
        """Save price data for a ticker to cache"""
        # Validate data before saving
        if data.empty:
            print(f"⚠️ Not caching empty data for {ticker}")
            return
            
        nan_pct = data.isna().mean() * 100
        if nan_pct > 50:  # If more than 50% is NaN, don't cache
            print(f"⚠️ Not caching {ticker} - poor data quality ({nan_pct:.1f}% NaN)")
            return
        
        # Clean up NaN values if there are just a few
        if nan_pct > 0:
            data = data.ffill().bfill()
            
        try:
            if os.path.exists(self.price_cache_file):
                with open(self.price_cache_file, 'rb') as f:
                    cache = pickle.load(f)
            else:
                cache = {}
            
            cache[ticker] = data
            
            with open(self.price_cache_file, 'wb') as f:
                pickle.dump(cache, f)
                
            # Update metadata
            self.metadata["last_updated"][ticker] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_metadata()
            
            print(f"✅ Cached price data for {ticker} ({len(data)} points)")
        except Exception as e:
            print(f"❌ Error saving cache for {ticker}: {str(e)}")
    
    def save_div_data(self, ticker: str, data: pd.Series) -> None:
        """Save dividend data for a ticker to cache"""
        if os.path.exists(self.div_cache_file):
            with open(self.div_cache_file, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = {}
        
        cache[ticker] = data
        
        with open(self.div_cache_file, 'wb') as f:
            pickle.dump(cache, f)
            
        # Metadata already updated in save_price_data
    
    def save_info_data(self, ticker: str, info: Dict[str, Any]) -> None:
        """Save company info data for a ticker to cache"""
        if os.path.exists(self.info_cache_file):
            with open(self.info_cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}
        
        cache[ticker] = {k: v for k, v in info.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
        
        with open(self.info_cache_file, 'w') as f:
            json.dump(cache, f, default=str)  # Use str as fallback for non-serializable objects
    
    def get_price_data(self, ticker: str, max_age_days: int = 7) -> Optional[pd.Series]:
        """
        Get price data for ticker from cache if available and not too old
        
        Parameters:
            ticker: The ticker symbol
            max_age_days: Maximum age of cached data in days
            
        Returns:
            Cached price data or None if not available or too old
        """
        if not os.path.exists(self.price_cache_file):
            print(f"⚠️ No cache file found at {self.price_cache_file}")
            return None
            
        # Check if data exists and isn't too old
        if ticker in self.metadata.get("last_updated", {}):
            last_update = datetime.strptime(self.metadata["last_updated"][ticker], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_update).days > max_age_days:
                print(f"⚠️ Cached data for {ticker} is over {max_age_days} days old")
                return None
                
            try:
                with open(self.price_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    
                if ticker in cache:
                    data = cache[ticker]
                    if len(data) > 10:  # Ensure we have sufficient data
                        print(f"📊 Using cached price data for {ticker} from {last_update.strftime('%Y-%m-%d')}")
                        return data
                    else:
                        print(f"⚠️ Cached data for {ticker} has insufficient points ({len(data)})")
                else:
                    print(f"⚠️ {ticker} not found in cache dictionary (metadata exists but data missing)")
            except Exception as e:
                print(f"❌ Error reading cache: {str(e)}")
                
        else:
            print(f"⚠️ No metadata found for {ticker}")
                
        return None
    
    def get_div_data(self, ticker: str, max_age_days: int = 30) -> Optional[pd.Series]:
        """Get dividend data for ticker from cache if available and not too old"""
        if not os.path.exists(self.div_cache_file):
            return None
            
        # Check if data exists and isn't too old
        if ticker in self.metadata.get("last_updated", {}):
            last_update = datetime.strptime(self.metadata["last_updated"][ticker], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_update).days > max_age_days:
                return None
                
            try:
                with open(self.div_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    
                if ticker in cache:
                    return cache[ticker]
            except Exception as e:
                print(f"Error reading cache: {str(e)}")
                
        return None
    
    def get_info_data(self, ticker: str, max_age_days: int = 30) -> Optional[Dict]:
        """Get company info data for ticker from cache if available and not too old"""
        if not os.path.exists(self.info_cache_file):
            return None
            
        # Check if data exists and isn't too old
        if ticker in self.metadata.get("last_updated", {}):
            last_update = datetime.strptime(self.metadata["last_updated"][ticker], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_update).days > max_age_days:
                return None
                
            try:
                with open(self.info_cache_file, 'r') as f:
                    cache = json.load(f)
                    
                if ticker in cache:
                    return cache[ticker]
            except Exception as e:
                print(f"Error reading cache: {str(e)}")
                
        return None
    
    def _save_metadata(self) -> None:
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        if os.path.exists(self.price_cache_file):
            os.remove(self.price_cache_file)
        if os.path.exists(self.div_cache_file):
            os.remove(self.div_cache_file)
        if os.path.exists(self.info_cache_file):
            os.remove(self.info_cache_file)
        
        self.metadata = {
            "last_updated": {},
            "cache_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_metadata()
        
        print("🧹 Cache cleared successfully")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get status information about the cache"""
        price_cache_size = 0
        price_ticker_count = 0
        cached_tickers = []
        
        if os.path.exists(self.price_cache_file):
            price_cache_size = os.path.getsize(self.price_cache_file) / (1024*1024)  # MB
            try:
                with open(self.price_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    price_ticker_count = len(cache_data)
                    cached_tickers = list(cache_data.keys())
            except Exception as e:
                print(f"⚠️ Error reading cache file for status: {str(e)}")
                
        # Rest of the code remains the same
        div_cache_size = 0
        if os.path.exists(self.div_cache_file):
            div_cache_size = os.path.getsize(self.div_cache_file) / (1024*1024)  # MB
        
        info_cache_size = 0
        info_ticker_count = 0
        if os.path.exists(self.info_cache_file):
            info_cache_size = os.path.getsize(self.info_cache_file) / (1024*1024)  # MB
            with open(self.info_cache_file, 'r') as f:
                info_ticker_count = len(json.load(f))
        
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
            "cached_tickers": cached_tickers[:5] if cached_tickers else [],  # First 5 tickers
            "div_cache_size_mb": round(div_cache_size, 2),
            "info_cache_size_mb": round(info_cache_size, 2),
            "info_ticker_count": info_ticker_count,
            "oldest_data": oldest,
            "newest_data": newest,
            "cache_created": self.metadata.get("cache_created")
        }
