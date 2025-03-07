# Modern Portfolio Optimizer

A robust tool for optimizing investment portfolios based on modern portfolio theory and Kelly criterion.

## Overview

This tool helps investors create optimized portfolios using historical stock data from Yahoo Finance. It:

- Calculates optimal asset allocations based on modern portfolio theory
- Implements Kelly criterion for position sizing and leveraged returns
- Handles data from multiple international exchanges
- Includes sophisticated caching and batch processing to avoid API rate limits
- Generates visual portfolio analysis with efficient frontier plots
- Provides extensive data quality validation and repair tools

## Installation

```bash
# Clone the repository
git clone https://github.com/MaxSchoon/modern-portfolio-optimizer.git
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

When adding tickers to your `tickers.csv` file, use the following formats based on the stock exchange:

- **US Stocks**: Use the standard ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`)
- **Amsterdam Exchange**: Add `.AS` suffix (e.g., `ASML.AS`, `BESI.AS`)
- **London Exchange**: Add `.L` suffix (e.g., `BP.L`, `HSBA.L`)
- **Paris Exchange**: Add `.PA` suffix (e.g., `AIR.PA`)
- **Frankfurt Exchange**: Add `.DE` suffix (e.g., `SAP.DE`)
- **Milan Exchange**: Add `.MI` suffix (e.g., `ENI.MI`)

Special cases:
- `DANAOS` will be corrected to `DAC`
- `FLOW` will be corrected to `FLOW.AS`
- `HAL` will be corrected to `HAL.AS`

For risk-free assets:
- `CASH`: Represents cash with risk-free rate
- `TBILLS`: Represents Treasury Bills (slightly higher yield than cash)

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
python3 ModernPortfolio.py

# With command line options
python3 ModernPortfolio.py --years 3 --no-cache --risk-free 0.035 --margin-cost 0.06

# Exclude cash/TBILLS (equity-only optimization)
python3 ModernPortfolio.py --exclude-cash

# Specify custom output directory
python3 ModernPortfolio.py --output-dir my_portfolio_results
```

## Data Management

### Testing Yahoo Finance Connectivity

```bash
# Test all tickers in the CSV file
python3 test_yahoo.py

# Test specific tickers
python3 test_yahoo.py AAPL MSFT GOOGL
```

### Using the Data Fetcher

```bash
# Fetch all tickers from CSV file
python3 data_fetcher.py --file tickers.csv

# Fetch specific tickers
python3 data_fetcher.py --tickers AAPL,MSFT,GOOGL

# Customize batch size and delay
python3 data_fetcher.py --file tickers.csv --batch-size 5 --delay 3
```

### Cache Maintenance

```bash
# List all cached tickers
python3 cache_tools.py list

# Inspect data for a specific ticker
python3 cache_tools.py inspect --ticker AAPL

# Fix NaN issues in cached data
python3 cache_tools.py fix --ticker MSFT

# Validate cached data quality
python3 cache_tools.py validate --ticker GOOGL

# Advanced cache maintenance
python3 cache_maintenance.py quality --cache-dir data_cache
python3 cache_maintenance.py repair --batch --cache-dir data_cache
```

## Data Fetching Strategy

This optimizer includes sophisticated methods to fetch data while avoiding Yahoo Finance rate limits:

1. **Smart Batch Fetcher:** 
   - Fetches data in small batches with adaptive delays
   - Uses multiple worker threads for better throughput
   - Implements automatic retries and error handling

2. **CSV Cache System:**
   - Stores data in efficient CSV format
   - Includes validation and repair tools
   - Handles dividend data separately for accurate total return calculations

3. **Synthetic Assets:**
   - Generates synthetic data for cash-like assets
   - Properly models risk-free returns and minimal volatility

## Output and Visualization

The optimizer generates several outputs in the `portfolio_analysis` directory:

- **Efficient Frontier Plot:** Visual representation of risk-return tradeoffs
- **Portfolio Allocation Chart:** Pie chart showing optimal asset allocation
- **Risk-Return Profile:** Scatter plot comparing individual assets
- **Price Charts:** Historical price charts for each asset
- **Results Summary:** 
  - `optimal_weights.json`: Optimized portfolio weights
  - `returns_summary.csv`: Return metrics for each asset
  - `correlations.csv`: Correlation matrix between assets

## Project Structure

| File | Description |
|------|-------------|
| **ModernPortfolio.py** | Main optimizer script with portfolio optimization logic |
| **data_fetcher.py** | Advanced data fetcher with batching and rate limiting |
| **csv_cache_manager.py** | CSV-based cache system for efficient data storage |
| **utils.py** | Utility functions for data processing and validation |
| **cache_tools.py** | Lightweight cache management tools |
| **cache_maintenance.py** | Advanced cache maintenance and repair tools |
| **test_yahoo.py** | Tests connectivity to Yahoo Finance API |
| **run_optimizer.sh** | Helper shell script for running the optimizer |

## Advanced Configuration

The optimizer supports several advanced parameters:

```
usage: ModernPortfolio.py [-h] [--no-cache] [--convert-cache] [--years YEARS]
                         [--risk-free RISK_FREE] [--margin-cost MARGIN_COST]
                         [--tickers-file TICKERS_FILE] [--clear-cache]
                         [--exclude-cash] [--debug]
                         [--output-dir OUTPUT_DIR]
                         [--batch-size BATCH_SIZE] [--workers WORKERS]
                         [--cache-dir CACHE_DIR]

options:
  -h, --help            show this help message and exit
  --no-cache            Disable cache
  --convert-cache       Convert old cache to CSV format
  --years YEARS         Years of historical data
  --risk-free RISK_FREE
                        Risk-free rate
  --margin-cost MARGIN_COST
                        Cost of margin
  --tickers-file TICKERS_FILE
                        Path to tickers file
  --clear-cache         Clear cache before starting
  --exclude-cash        Exclude cash and T-bills from optimization (equity-only)
  --debug               Enable debug mode with more verbose output
  --output-dir OUTPUT_DIR
                        Directory for output files and visualizations
  --batch-size BATCH_SIZE
                        Batch size for data fetching
  --workers WORKERS     Number of worker threads for data fetching
  --cache-dir CACHE_DIR
                        Directory for cached data
```

## Troubleshooting

If you encounter issues:

1. **Check Yahoo Finance connectivity** - Run `test_yahoo.py` to verify API access
2. **Validate ticker formats** - Ensure tickers follow the format guide
3. **Inspect cached data** - Use `cache_tools.py inspect` to check data quality
4. **Run with --debug flag** - Get more detailed error information
5. **Clear cache** - Use `--clear-cache` if cached data might be corrupted
6. **Check data quality** - Run `cache_maintenance.py quality` to identify issues
7. **Repair data** - Use `cache_maintenance.py repair` to fix common data problems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

