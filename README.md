# Modern Portfolio Optimizer

A tool for optimizing investment portfolios based on modern portfolio theory, including Kelly criterion.

## Overview

This tool helps investors create optimized portfolios using historical stock data from Yahoo Finance. It:

- Calculates optimal asset allocations based on modern portfolio theory
- Implements Kelly criterion for position sizing
- Handles data from multiple international exchanges
- Includes caching and batch processing to avoid API rate limits
- Supports various risk-return optimization strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/modern-portfolio-optimizer.git
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
```

### Using the Python Script Directly

```bash
# Basic run
python ModernPortfolio.py

# With command line options
python ModernPortfolio.py --years 3 --no-cache --risk-free 0.035 --margin-cost 0.06
```

## Validating & Testing Data

### Validating Tickers

```bash
# Validate and correct all tickers in your file
python fetch_and_validate.py

# Use the corrected tickers file
python ModernPortfolio.py --tickers-file tickers_corrected.csv

# Test specific tickers individually
python ticker_validator.py GOOGL ASML.AS BRK-B
```

### Testing Yahoo Finance Connectivity

```bash
# Test all tickers in the CSV file
python test_yahoo.py

# Test specific tickers
python test_yahoo.py AAPL MSFT GOOGL
```

## Data Fetching Strategy

This optimizer includes two methods to fetch data while avoiding Yahoo Finance rate limits:

1. **Batch Fetcher:** Fetches data in small batches with delays between requests
2. **Caching System:** Stores previously fetched data to reduce API calls

## Troubleshooting Data Fetching Issues

If you encounter problems fetching data:

1. **Check your internet connection** - Ensure you have internet access
2. **Verify ticker formats** - Make sure your tickers follow the format guide above
3. **Rate limiting** - Yahoo Finance might rate-limit your requests, try using the batch fetcher
4. **Run the test script** - Use `test_yahoo.py` to check individual tickers
5. **Pre-fetch data** - Use `./run_optimizer.sh --fetch-only` to fetch data separately
6. **Check cache** - If using cached data, try clearing the cache with `--clear-cache`

## Project Files

| File | Description |
|------|-------------|
| **ModernPortfolio.py** | Main optimizer script that calculates optimal portfolios using modern portfolio theory and Kelly criterion |
| **batch_fetcher.py** | Handles fetching financial data in batches with delays to avoid rate limits |
| **cache_manager.py** | Manages a local cache of financial data to reduce API calls |
| **fetch_and_validate.py** | Validates tickers and corrects formats where possible |
| **ticker_validator.py** | Tests individual tickers to ensure they can be fetched |
| **test_yahoo.py** | Tests connectivity to Yahoo Finance API |
| **run_optimizer.sh** | Helper shell script that simplifies running the optimizer with various options |
| **tickers.csv** | User-provided list of stock tickers to analyze |
| **requirements.txt** | List of Python package dependencies |

## Advanced Configuration

The optimizer supports several advanced parameters:

- Risk-free rate customization
- Margin cost settings
- Time period adjustments
- Risk tolerance controls
- Portfolio constraints

For advanced usage, see the help documentation:

```bash
python ModernPortfolio.py --help
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

