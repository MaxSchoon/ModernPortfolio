"""
Cache Conversion Utility

This script converts from the old pickle-based cache to the new CSV-based cache.
"""

from csv_cache_manager import CSVDataCache
import argparse
import os

def convert_cache(old_cache_dir: str = "data_cache", new_cache_dir: str = "csv_data_cache") -> None:
    """
    Convert data from old pickle cache to new CSV cache
    """
    # Check if old cache exists
    if not os.path.exists(old_cache_dir):
        print(f"❌ Old cache directory not found: {old_cache_dir}")
        return
    
    # Create CSV cache manager
    csv_cache = CSVDataCache(new_cache_dir)
    
    # Do the conversion
    converted = csv_cache.convert_from_pickle_cache(old_cache_dir)
    
    if converted > 0:
        print(f"\n✅ Successfully converted {converted} tickers from pickle to CSV cache")
        
        # Print status of the new cache
        status = csv_cache.get_cache_status()
        print("\nNew CSV Cache Status:")
        print(f"- Price data: {status.get('price_ticker_count', 0)} tickers")
        print(f"- Total size: {status.get('price_cache_size_mb', 0):.2f} MB")
        print(f"- Date range: {status.get('oldest_data', 'N/A')} to {status.get('newest_data', 'N/A')}")
    else:
        print("❌ No data was converted")

def main():
    parser = argparse.ArgumentParser(description="Convert from old pickle cache to new CSV cache")
    parser.add_argument('--old-cache', help='Old pickle cache directory', default='data_cache')
    parser.add_argument('--new-cache', help='New CSV cache directory', default='csv_data_cache')
    
    args = parser.parse_args()
    convert_cache(args.old_cache, args.new_cache)

if __name__ == "__main__":
    main()