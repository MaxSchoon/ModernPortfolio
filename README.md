# Modern Portfolio Optimizer

A fun, experimental project for exploring and testing different configurations of Modern Portfolio Theory (MPT) by Markowitz, including the Kelly criterion. This project is not intended for professional investment advice, but rather as a playground for learning, experimentation, and portfolio optimization techniques.

## Overview

This project features an improved, modular structure for easier experimentation and extension. The codebase is organized into clear modules:

- **src/core/**: Main optimizer logic and portfolio algorithms
- **src/fetching/**: Data fetching and batch processing
- **src/cache/**: Cache management and standardization tools
- **src/utils/**: Utility functions and lightweight cache tools
- **src/tests/**: Test scripts and connectivity checks
- **src/data/**: Ticker lists and data files
- **scripts/**: Shell scripts for running and testing
- **data_cache/**: Local cache of fetched price/dividend/metadata/info data
- **portfolio_analysis/**: Output plots, summaries, and analysis results

This structure makes it simple to test new ideas, swap out components, and keep code organized.

This tool helps investors and enthusiasts create optimized portfolios using historical stock data from Yahoo Finance. It:

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
cd ModernPortfolio

# Install required dependencies
pip install -r requirements.txt
```

## Quick Start

1. Create a `tickers.csv` file in `src/data/` with your desired stocks (see Ticker Format Guide below)
2. Run the optimizer with default settings:

```bash
./scripts/run_optimizer.sh
```

Or run directly:

```bash
python3 src/core/ModernPortfolio.py
```

## Project Structure

```
ModernPortfolio/
├── src/
│   ├── core/
│   │   └── ModernPortfolio.py
│   │   └── portfolio_fix.py
│   ├── fetching/
│   │   └── data_fetcher.py
│   ├── cache/
│   │   ├── cache_maintenance.py
│   │   ├── cache_standardize.py
│   │   └── csv_cache_manager.py
│   ├── utils/
│   │   ├── cache_tools.py
│   │   └── utils.py
│   ├── tests/
│   │   └── test_yahoo.py
│   └── data/
│       ├── tickers.csv
│       └── tickers_corrected.csv
├── scripts/
│   ├── run_optimizer.sh
│   └── test_modern_portfolio.sh
├── data_cache/
│   ├── prices/
│   ├── dividends/
│   ├── metadata/
│   └── info/
├── portfolio_analysis/
│   ├── risk_return_profile.png
│   ├── efficient_frontier.png
│   ├── portfolio_allocation.png
│   └── ...
├── requirements.txt
└── README.md
```

## Project Files

| File/Folder | Description |
|-------------|-------------|
| **src/core/ModernPortfolio.py** | Main optimizer script using modern portfolio theory and Kelly criterion |
| **src/fetching/data_fetcher.py** | Handles fetching financial data in batches with rate limiting protection |
| **src/cache/csv_cache_manager.py** | Manages CSV-based local cache of financial data |
| **src/utils/utils.py** | Common utility functions used across different scripts |
| **src/utils/cache_tools.py** | Lightweight tools for cache inspection and basic fixes |
| **src/cache/cache_maintenance.py** | Advanced cache management and repair tools |
| **src/cache/cache_standardize.py** | Standardizes date formats in cache files |
| **src/tests/test_yahoo.py** | Tests connectivity to Yahoo Finance API |
| **scripts/run_optimizer.sh** | Helper shell script for running with various options |
| **src/data/tickers.csv** | User-provided list of stock tickers to analyze |
| **requirements.txt** | List of Python package dependencies |
| **portfolio_analysis/** | Output plots, summaries, and analysis results |
| **data_cache/** | Local cache of fetched price/dividend/metadata/info data |

## Usage Options

### Using the Shell Script

The included shell script in `scripts/` provides an easy way to run the optimizer with common options:

```bash
./scripts/run_optimizer.sh
```

### Using the Python Script Directly

```bash
python3 src/core/ModernPortfolio.py --years 3 --no-cache --risk-free 0.035 --margin-cost 0.06 --fast
```

## Data Management Tools

- **Cache tools:** `src/utils/cache_tools.py`
- **Cache maintenance:** `src/cache/cache_maintenance.py`
- **Cache standardization:** `src/cache/cache_standardize.py`
- **Testing Yahoo Finance:** `src/tests/test_yahoo.py`

## Output and Analysis

All output plots, summaries, and analysis results are saved in the `portfolio_analysis/` directory.

## Roadmap

**Backtesting System for Efficient Portfolios**

- Develop a robust backtesting engine to simulate how an "efficient" Modern Portfolio (as determined by the optimizer) would have performed over time, with regular rebalancing (e.g., monthly or quarterly).
- To ensure high performance and scalability, consider implementing the core backtesting logic in a faster language such as C++, Go, or Cython (for Python acceleration). This will allow for rapid simulation of large portfolios and long time spans.
- The backtester will:
  - Take historical data and portfolio weights as input
  - Simulate rebalancing at specified intervals
  - Track portfolio returns, volatility, drawdowns, and other metrics
  - Output detailed performance reports and comparison plots
- The goal is to provide deeper insight into the real-world effectiveness of Modern Portfolio Theory strategies under realistic trading conditions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

