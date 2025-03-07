# Modern Portfolio Optimizer

A tool for optimizing investment portfolios based on modern portfolio theory, including Kelly criterion.

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

## Ensuring Valid Data

To ensure your data is usable, follow these steps:

1. **Validate and correct your tickers first**:
   ```bash
   python fetch_and_validate.py
   ```
   This will check all tickers, correct formats where needed, and pre-fetch data.

2. **Use the corrected tickers file**:
   ```bash
   python ModernPortfolio.py --tickers-file tickers_corrected.csv
   ```

3. **If you're still having issues with specific tickers, test them individually**:
   ```bash
   python ticker_validator.py GOOGL ASML.AS BRK-B
   ```

## Data Fetching Strategy

This optimizer includes two methods to fetch data while avoiding Yahoo Finance rate limits:

1. **Batch Fetcher:** Fetches data in small batches with delays between requests
2. **Caching System:** Stores previously fetched data to reduce API calls

### Running the Optimizer

The simplest way to run the optimizer is using the provided shell script:

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
```

Or you can run the Python script directly:

```bash
# Basic run
python ModernPortfolio.py

# With command line options
python ModernPortfolio.py --years 3 --no-cache --risk-free 0.035 --margin-cost 0.06
```

### Testing Yahoo Finance Connectivity

Before running the optimizer, you can test connectivity to Yahoo Finance and validate your tickers:

```bash
# Test all tickers in the CSV file
python test_yahoo.py

# Test specific tickers
python test_yahoo.py AAPL MSFT GOOGL
```

## Troubleshooting Data Fetching Issues

If you encounter problems fetching data:

1. **Check your internet connection** - Ensure you have internet access
2. **Verify ticker formats** - Make sure your tickers follow the format guide above
3. **Rate limiting** - Yahoo Finance might rate-limit your requests, try using the batch fetcher
4. **Run the test script** - Use `test_yahoo.py` to check individual tickers
5. **Pre-fetch data** - Use `./run_optimizer.sh --fetch-only` to fetch data separately
6. **Check cache** - If using cached data, try clearing the cache with `--clear-cache`

## Files Overview

- `ModernPortfolio.py`: Main optimizer script
- `batch_fetcher.py`: Fetches data in batches with delays to avoid rate limits
- `cache_manager.py`: Manages caching of financial data
- `test_yahoo.py`: Tests connectivity to Yahoo Finance
- `tickers.csv`: List of tickers to analyze
- `run_optimizer.sh`: Helper script to run the optimizer

