"""
Financial Data Cache Manager

DEPRECATED: This is a backward compatibility wrapper.
Please use cache_maintenance.py for all cache management functions.

All functions are directly imported from cache_maintenance.py.
"""

import sys
import warnings

# Show deprecation warning
warnings.warn(
    "cache_manager.py is deprecated and will be removed in a future version. "
    "Please use cache_maintenance.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import all public functions from cache_maintenance.py
from cache_maintenance import (
    # Constants
    DEFAULT_CACHE_DIR,
    
    # Ticker validation
    validate_ticker_list,
    validate_csv_tickers,
    
    # Cache inspection
    list_cached_tickers,
    inspect_ticker_data,
    check_all_tickers_data_quality,
    plot_ticker,
    
    # Data quality fixes
    inspect_csv_file,
    fix_csv_file,
    test_and_repair_cached_data,
    
    # Cache conversion
    convert_pickle_to_csv,
    
    # Batch operations
    batch_validate_and_fix,
    
    # Main entry point
    main
)

# If script is run directly, execute main() from cache_maintenance.py
if __name__ == "__main__":
    main()