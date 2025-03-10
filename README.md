# Modern Portfolio Optimizer

A tool for optimizing investment portfolios based on modern portfolio theory, including Kelly criterion.

## Overview

This tool helps investors create optimized portfolios using historical stock data from Yahoo Finance. It:

- Calculates optimal asset allocations based on modern portfolio theory
- Implements Kelly criterion for position sizing
- Handles data from multiple international exchanges
- Includes caching and batch processing to avoid API rate limits
- Supports various risk-return optimization strategies
- Provides robust data management with CSV cache system

## Installation

```bash
# Clone the repository
git clone https://github.com/MaxSchoon/ModernPortfolio.git
cd modern-portfolio-optimizer

# Install required dependencies
pip install -r requirements.txt
```

## Quick Start

1. Create a `tickers.csv` file with your desired stocks (see Ticker Format Guide below)
2. Run the optimizer with default settings:

```bash
./run_optimizer.sh
```

## Ticker Format Guide

When adding tickers to your `tickers.csv` file, please use the following formats based on the stock exchange:

- **US Stocks**: Use the standard ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`)
- **Amsterdam Exchange**: Add `.AS` suffix (e.g., `ASML.AS`, `BESI.AS`)
- **London Exchange**: Add `.L` suffix (e.g., `BP.L`, `HSBA.L`)
- **Paris Exchange**: Add `.PA` suffix (e.g., `AIR.PA`)
- **Frankfurt Exchange**: Add `.DE` suffix (e.g., `SAP.DE`)
- **Milan Exchange**: Add `.MI` suffix (e.g., `ENI.MI`)

For special assets:
- `CASH`: Represents cash with risk-free rate
- `TBILLS`: Represents Treasury Bills

## Usage Options

### Using the Shell Script

The included shell script provides an easy way to run the optimizer with common options:

```bash
# Normal run (uses both cache and batch fetcher)
./run_optimizer.sh

# Run with 3 years of data instead of default 5
./run_optimizer.sh "" 3

# Clear cache before running
./run_optimizer.sh --clear-cache

# Run without using cache (fetch fresh data)
./run_optimizer.sh --no-cache

# Run without batch fetcher (faster but may hit rate limits)
./run_optimizer.sh --no-batch

# Just fetch and cache the data without running the optimizer
./run_optimizer.sh --fetch-only

# Standardize the cache data
./run_optimizer.sh --standardize-cache
```

### Using the Python Script Directly

```bash
# Basic run
python3 ModernPortfolio.py

# With command line options (including --fast for performance)
python3 ModernPortfolio.py --years 3 --no-cache --risk-free 0.035 --margin-cost 0.06 --fast
```

## Data Management Tools

### Cache Tools

Several tools are provided to manage the CSV data cache:

```bash
# Quick cache inspection
python3 cache_tools.py list

# Inspect a specific ticker's cached data
python3 cache_tools.py inspect --ticker AAPL

# Fix NaN issues in a ticker's data
python3 cache_tools.py fix --ticker MSFT

# Validate cached ticker data
python3 cache_tools.py validate --ticker GOOGL
```

### Cache Maintenance

For more advanced cache maintenance:

```bash
# List all cached tickers
python3 cache_maintenance.py list

# Detailed inspection of a ticker
python3 cache_maintenance.py inspect AAPL

# Plot a ticker's price history
python3 cache_maintenance.py plot AAPL

# Test and repair all data quality issues
python3 cache_maintenance.py repair --batch

# Clear cache (with confirmation prompt)
python3 cache_maintenance.py clear
```

### Cache Standardization

The cache standardization tool ensures consistent date formats and removes future dates:

```bash
# Standardize all cache files
python3 cache_standardize.py

# Verify cache alignment without making changes
python3 cache_standardize.py --verify

# Keep future dates (not recommended)
python3 cache_standardize.py --keep-future
```

### Testing Yahoo Finance Connectivity

```bash
# Test all tickers in the CSV file
python3 test_yahoo.py

# Test specific tickers
python3 test_yahoo.py AAPL MSFT GOOGL
```

## Data Fetching Strategy

This optimizer includes advanced methods to fetch data while avoiding Yahoo Finance rate limits:

1. **Batch Fetcher:** Fetches data in small batches with delays between requests
2. **CSV Caching System:** Stores previously fetched data in standardized CSV format
3. **Data Validation:** Ensures data quality and consistency
4. **Parallel Processing:** Uses concurrent fetching for better performance
5. **Rate Limiting Protection:** Built-in throttling to avoid API blocks

The `data_fetcher.py` module manages all these aspects automatically.

## CSV Cache Format

The tool uses a structured CSV cache system with the following organization:

- `/data_cache/prices/`: Contains price data in `TICKER_price.csv` files
- `/data_cache/dividends/`: Contains dividend data in `TICKER_div.csv` files
- `/data_cache/metadata/`: Contains metadata in `TICKER_meta.json` files
- `/data_cache/info/`: Contains company information in `TICKER_info.json` files

All date formats are standardized to ISO format (YYYY-MM-DD) for consistency.

## Troubleshooting Data Fetching Issues

If you encounter problems fetching data:

1. **Check your internet connection** - Ensure you have internet access
2. **Verify ticker formats** - Make sure your tickers follow the format guide above
3. **Rate limiting** - Yahoo Finance might rate-limit your requests:
   - Use the batch fetcher with `data_fetcher.py`
   - Increase delay between requests with `--delay-min` and `--delay-max` options
   - Reduce batch size with the `--batch-size` option
4. **Fix cache issues** - Cache files might have inconsistencies:
   - Run `python3 cache_standardize.py` to fix date format issues
   - Use `python3 cache_maintenance.py repair` to fix data quality issues
5. **Test connectivity** - Use `test_yahoo.py` to check individual tickers
6. **Pre-fetch data** - Use `./run_optimizer.sh --fetch-only` to fetch data separately
7. **Check cache data quality** - Run `python3 cache_tools.py list` to check cache status

## Project Files

| File | Description |
|------|-------------|
| **ModernPortfolio.py** | Main optimizer script using modern portfolio theory and Kelly criterion |
| **data_fetcher.py** | Handles fetching financial data in batches with rate limiting protection |
| **csv_cache_manager.py** | Manages CSV-based local cache of financial data |
| **utils.py** | Common utility functions used across different scripts |
| **cache_tools.py** | Lightweight tools for cache inspection and basic fixes |
| **cache_maintenance.py** | Advanced cache management and repair tools |
| **cache_standardize.py** | Standardizes date formats in cache files |
| **test_yahoo.py** | Tests connectivity to Yahoo Finance API |
| **run_optimizer.sh** | Helper shell script for running with various options |
| **tickers.csv** | User-provided list of stock tickers to analyze |
| **requirements.txt** | List of Python package dependencies |

## Advanced Configuration

The optimizer supports several advanced parameters:

- Risk-free rate customization
- Margin cost settings
- Time period adjustments
- Risk tolerance controls
- Portfolio constraints
- Cache management options
- Data standardization settings
- Fast: Enables all performance optimizations (also sets --skip-plots)
- Skip-plots: Skips generating plots for faster execution

For advanced usage, see the help documentation:

```bash
python3 ModernPortfolio.py --help
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

