from unittest import result
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import os
import argparse
import json
import matplotlib.pyplot as plt
import time  # Add time for performance tracking

# Import the CSV cache manager and cache standardizer
from src.csv_cache_manager import CSVDataCache
try:
    from src.cache_standardize import standardize_cache
    HAS_STANDARDIZER = True
except ImportError:
    print("Warning: cache_standardize module not found. Cache standardization will be skipped.")
    HAS_STANDARDIZER = False

# Only import utils modules if they exist - add a fallback
try:
    from src.utils import load_tickers, format_ticker
except ImportError:
    # Define the functions here if utils.py doesn't exist
    def load_tickers(csv_path: str) -> List[str]:
        """Load tickers from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tickers file not found: {csv_path}")
        
        try:
            print(f"Reading ticker data from: {csv_path}")
            
            # Try to detect delimiter
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    delimiter = ';'
                    print("Detected semicolon delimiter")
                elif ',' in first_line:
                    delimiter = ','
                    print("Detected comma delimiter")
                else:
                    delimiter = ';'  # Default
                    print("Using default semicolon delimiter")
            
            # Skip header comment lines
            df = pd.read_csv(csv_path, sep=delimiter, comment='/')
            
            # Check column names (case insensitive)
            ticker_column = None
            for col in df.columns:
                if (col.lower() == 'ticker'):
                    ticker_column = col
                    break
                    
            if ticker_column is None:
                print(f"Columns in CSV: {', '.join(df.columns)}")
                raise ValueError("CSV file must contain a 'ticker' column")
                
            tickers = df[ticker_column].tolist()
            print(f"Found {len(tickers)} tickers in the CSV file")
            
            # Print first few tickers
            sample = tickers[:min(5, len(tickers))]
            print(f"Sample tickers: {', '.join(sample)}")
            
            return tickers
        except Exception as e:
            print(f"Error reading tickers file: {str(e)}")
            raise

    def format_ticker(ticker: str) -> str:
        """Format ticker symbol for Yahoo Finance""" 
        # For US stocks, nothing changes for most tickers
        if '.' not in ticker and '-' not in ticker:
            return ticker
            
        # Handle stocks with special characters
        if '-' in ticker:  # Stocks like BRK-B need to be formatted as BRK-B
            return ticker
            
        # Handle international exchanges
        if ticker.endswith('.AS'):  # Amsterdam
            return ticker
        elif ticker.endswith('.L'):  # London
            return ticker
        elif ticker.endswith('.PA'):  # Paris 
            return ticker
        elif ticker.endswith('.DE'):  # Germany
            return ticker
        elif ticker.endswith('.MI'):  # Milan
            return ticker
        
        # For other formats, leave as is
        return ticker

# Try to import BatchFetcher, but define a simple version if it fails
try:
    from src.data_fetcher import DataFetcher
except ImportError:
    print("Warning: BatchFetcher module not found. Using simplified version.")
    
    class DataFetcher:
        """Simplified batch fetcher that falls back to individual fetching"""
        def __init__(self, years=5, batch_size=3, delay_min=2.0, delay_max=5.0, retry_count=3):
            self.years = years
            print("Using simplified batch fetcher (missing module)")
            
        def fetch_all(self, tickers, use_cache=True):
            print("Simplified batch fetcher does not implement actual fetching")
            return {ticker: "Not implemented" for ticker in tickers}

# Import the new graphing module
try:
    from src.graphing import (plot_price_data, plot_portfolio_weights,
                           plot_efficient_frontier, plot_returns_comparison)
    HAS_GRAPHING = True
except ImportError:
    print("Warning: graphing module not found. Using built-in plotting functions.")
    HAS_GRAPHING = False

class PortfolioOptimizer:
    def __init__(self, tickers: List[str], risk_free_rate: float = 0.04, margin_cost_rate: float = 0.065, 
                years: int = 5, output_dir: str = "portfolio_analysis", cache_dir: str = "data_cache",
                debug: bool = False, shorts: bool = False, max_long: float = 1.0, max_short: float = 0.3, 
                exclude_cash: bool = False):
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.margin_cost_rate = margin_cost_rate
        self.years = years
        self.price_data = None
        self.div_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.cache = CSVDataCache(cache_dir)  # Pass the cache_dir
        self.output_dir = output_dir
        self.charts_dir = os.path.join(output_dir, "price_charts")
        self.cache_dir = cache_dir
        self.debug = debug
        self.shorts = shorts  # Add shorts flag
        self.max_long = max_long  # Maximum allocation to long positions (as a decimal)
        self.max_short = max_short  # Maximum allocation to short positions (as a decimal)
        self.exclude_cash = exclude_cash  # Add exclude_cash flag
        
        # Create output directories
        for directory in [output_dir, self.charts_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # Add performance tracking
        self.performance_stats = {}

        # If exclude_cash is set, filter out cash tickers at initialization
        if self.exclude_cash:
            self.tickers = [t for t in self.tickers if t not in ['CASH', 'TBILLS']]
            print(f"Exclude cash mode: Removed CASH and TBILLS from tickers list")

    def validate_data(self, ticker: str, prices: pd.Series, dividends: Optional[pd.Series] = None) -> bool:
        """Validate data quality for a given ticker"""
        if prices.empty:
            print(f"Warning: No price data found for {ticker}")
            return False
        
        missing_pct = (prices.isna().sum() / len(prices)) * 100
        if missing_pct > 10:
            print(f"Warning: {ticker} has {missing_pct:.1f}% missing price data")
            return False
            
        if dividends is not None and not dividends.empty:
            print(f"Info: {ticker} has {len(dividends)} dividend payments")
            print(f"Latest dividend: {dividends.iloc[-1]:.4f}")
            
        return True

    def calculate_weighted_dividend_yield(self, ticker: str) -> Tuple[float, float]:
        """Calculate exponentially weighted dividend yield for a ticker"""
        if ticker in ['CASH', 'TBILLS'] or ticker not in self.div_data:
            return 0.0, 0.0
            
        # Get annual dividend data
        prices = self.price_data[ticker]
        dividends = self.div_data[ticker]
        
        # Calculate annual dividends for each year
        annual_div = []
        annual_yield = []
        
        for year in range(self.years):
            start_idx = -(year + 1) * 252
            end_idx = -year * 252 if year > 0 else None
            
            year_dividends = dividends.iloc[start_idx:end_idx].sum()
            year_avg_price = prices.iloc[start_idx:end_idx].mean()
            
            if year_avg_price > 0:
                annual_div.append(year_dividends)
                annual_yield.append(year_dividends / year_avg_price)
        
        # Calculate exponentially weighted averages
        if not annual_yield:
            return 0.0, 0.0
            
        weights = np.exp(-np.arange(len(annual_yield)) * 0.5)  # Higher weight for recent years
        weights = weights / weights.sum()
        
        weighted_yield = np.sum(weights * annual_yield)
        total_div = np.sum(annual_div)
        
        return weighted_yield, total_div

    def print_data_summary(self):
        """Print summary statistics for all assets"""
        print("\nData Summary:")
        print("-" * 70)
        
        for ticker in self.tickers:
            prices = self.price_data[ticker]
            
            # Check for NaN values
            nan_count = prices.isna().sum()
            if (nan_count > 0):
                print(f"\n{ticker} - WARNING: Contains {nan_count} NaN values!")
                # Try to get valid price range using non-NaN values
                valid_prices = prices.dropna()
                if valid_prices.empty:
                    print("No valid price data available!")
                    continue
                print(f"Using {len(valid_prices)} valid data points for statistics")
                first_date = valid_prices.index[0].date()
                last_date = valid_prices.index[-1].date()
                price_min = valid_prices.min()
                price_max = valid_prices.max()
                first_price = valid_prices.iloc[0]
                last_price = valid_prices.iloc[-1]
            else:
                first_date = prices.index[0].date()
                last_date = prices.index[-1].date()
                price_min = prices.min()
                price_max = prices.max()
                first_price = prices.iloc[0]
                last_price = prices.iloc[-1]
            
            print(f"\n{ticker}:")
            print(f"Period: {first_date} to {last_date}")
            print(f"Trading days: {len(prices) - nan_count}")
            print(f"Price range: {price_min:.2f} to {price_max:.2f}")
            
            # Calculate total and annualized returns only if we have valid data
            if not np.isnan(first_price) and not np.isnan(last_price) and first_price > 0:
                total_return = (last_price/first_price) - 1
                ann_return = (1 + total_return)**(252/(len(prices) - nan_count)) - 1
                print(f"Total return: {total_return*100:.2f}%")
                print(f"Annualized return: {ann_return*100:.2f}%")
            else:
                print("Total return: N/A (invalid price data)")
                print("Annualized return: N/A (invalid price data)")
            
            if ticker not in ['CASH', 'TBILLS']:
                weighted_yield, total_div = self.calculate_weighted_dividend_yield(ticker)
                print(f"Total dividends (5Y): {total_div:.2f}")
                print(f"Weighted dividend yield: {weighted_yield*100:.2f}%")
                
                # Calculate yearly dividend growth if possible
                dividends = self.div_data[ticker]
                try:
                    annual_divs = [dividends.iloc[-(i+1)*252:-i*252 if i > 0 else None].sum() 
                                 for i in range(min(5, len(dividends)//252))]
                    
                    # Only calculate growth rate if we have valid dividend history
                    if len(annual_divs) > 1 and all(div > 0 for div in annual_divs):
                        try:
                            div_growth = (annual_divs[0]/annual_divs[-1])**(1/len(annual_divs)) - 1
                            print(f"Dividend growth rate: {div_growth*100:.2f}%")
                        except (ZeroDivisionError, RuntimeWarning):
                            print("Dividend growth rate: N/A (insufficient data)")
                    else:
                        print("Dividend growth rate: N/A (insufficient data)")
                except Exception as e:
                    print(f"Error calculating dividend growth: {str(e)}")

    def test_yahoo_connectivity(self):
        """Test connectivity to Yahoo Finance API"""
        print("Testing Yahoo Finance connectivity...")
        
        try:
            # Try to fetch a well-known ticker
            test_ticker = "AAPL"
            stock = yf.Ticker(test_ticker)
            data = stock.history(period="1d")
            
            if (data.empty):
                print("❌ Error: Connected to Yahoo Finance but received empty data")
                return False
                
            print(f"✅ Successfully connected to Yahoo Finance API")
            print(f"   Sample data for {test_ticker}: {data['Close'].iloc[-1]:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Yahoo Finance: {str(e)}")
            return False

    def fetch_data(self, years: Optional[int] = None, use_cache: bool = True, 
                   batch_size: int = 5, max_workers: int = 3) -> None:
        """
        Fetch price and dividend data using DataFetcher
        
        Parameters:
            years: Number of years of data to fetch
            use_cache: Whether to use cached data
            batch_size: Number of tickers to fetch in each batch
            max_workers: Maximum number of concurrent workers
        """
        if years is None:
            years = self.years
            
        print(f"\nFetching data for {len(self.tickers)} tickers over {years} years...")
        
        # Initialize DataFetcher with appropriate settings
        fetcher = DataFetcher(
            cache_dir=self.cache_dir,
            batch_size=batch_size,
            years=years,
            max_workers=max_workers
        )
        
        # Fetch data for all tickers
        results = fetcher.fetch_all(self.tickers, use_cache)
        
        # Initialize price and div dataframes
        self.price_data = pd.DataFrame()
        self.div_data = pd.DataFrame()
        
        # Process fetched data
        price_data_dict = {}
        div_data_dict = {}
        
        for ticker in self.tickers:
            # Check if data fetching was successful
            if ticker in results and results[ticker].startswith("✅"):
                # Get data from cache (since DataFetcher will have saved it there)
                cached_price = self.cache.get_price_data(ticker)
                cached_div = self.cache.get_div_data(ticker)
                
                if cached_price is not None:
                    price_data_dict[ticker] = cached_price
                    
                    # Plot price data
                    self._plot_price_data(ticker, cached_price)
                    
                if cached_div is not None:
                    div_data_dict[ticker] = cached_div
            else:
                print(f"❌ {ticker}: Failed to fetch data")
        
        # Create DataFrame from collected series
        if price_data_dict:
            self.price_data = pd.concat(price_data_dict, axis=1)
        if div_data_dict:
            self.div_data = pd.concat(div_data_dict, axis=1)
        
        # Remove tickers that failed to load
        failed_tickers = [ticker for ticker in self.tickers if ticker not in self.price_data.columns]
        if failed_tickers:
            print(f"\n❌ The following tickers failed to load: {', '.join(failed_tickers)}")
            self.tickers = [t for t in self.tickers if t in self.price_data.columns]
        
        # Only create synthetic assets if not excluding cash
        if not self.exclude_cash:
            for synthetic in ['CASH', 'TBILLS']:
                if synthetic not in self.tickers:
                    self.tickers.append(synthetic)
                    self._create_synthetic_asset(synthetic, self.price_data.index if not self.price_data.empty else None)
        
        print(f"\n✅ Successfully loaded data for {len(self.tickers)} tickers")

    def _create_synthetic_asset(self, ticker: str, common_index=None):
        """
        Create synthetic data for cash-like assets using the common date index
        from real ticker data to ensure alignment
        
        Parameters:
            ticker: 'CASH' or 'TBILLS'
            common_index: Date index to use (from real tickers)
        """
        # If no common index is provided, create a generic one
        if (common_index is None):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*self.years)
            common_index = pd.date_range(start=start_date, end=end_date, freq='B')
            print(f"⚠️ Creating generic date range for {ticker} with {len(common_index)} days")
        else:
            print(f"🔄 Creating {ticker} with {len(common_index)} days aligned to other tickers")
        
        # Set return based on asset type
        if ticker == 'CASH':
            rate = self.risk_free_rate
            variance = 1e-10  # Very small variance for cash
        else:  # TBILLS
            rate = self.risk_free_rate * 1.02  # Slightly higher than cash
            variance = 1e-6   # More volatile than cash (increased from 1e-8)
        
        # Calculate daily return
        daily_return = (1 + rate) ** (1/252) - 1
        
        # Generate smooth price series with minimal noise
        days = len(common_index)
        prices = np.ones(days)
        
        # Apply compounding with tiny random noise
        for i in range(1, days):
            daily_noise = np.random.normal(0, np.sqrt(variance))
            prices[i] = prices[i-1] * (1 + daily_return + daily_noise)
        
        # Create price series with the common index
        price_series = pd.Series(prices, index=common_index)
        self.price_data[ticker] = price_series
        
        # Create zero-dividend series
        self.div_data[ticker] = pd.Series(0.0, index=common_index)
        
        print(f"💰 {ticker}: Synthetic data generated successfully ({len(price_series)} days)")

    def _plot_price_data(self, ticker: str, price_data: pd.Series) -> None:
        """Plot price data and save to file"""
        if HAS_GRAPHING:
            try:
                plot_price_data(ticker, price_data, output_dir=self.charts_dir)
            except Exception as e:
                print(f"⚠️ Error creating plot for {ticker}: {str(e)}")
        else:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(price_data.index, price_data.values)
                plt.title(f"{ticker} Price History")
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                plt.grid(True)
                
                # Add annotations for start/end prices
                start_price = price_data.iloc[0]
                end_price = price_data.iloc[-1]
                plt.annotate(f"${start_price:.2f}", xy=(price_data.index[0], start_price),
                            xytext=(10, 10), textcoords="offset points")
                plt.annotate(f"${end_price:.2f}", xy=(price_data.index[-1], end_price),
                            xytext=(-40, 10), textcoords="offset points")
                
                # Save plot to the charts directory instead of main output directory
                plot_file = os.path.join(self.charts_dir, f"{ticker}_price.png")
                plt.savefig(plot_file)
                plt.close()
            except Exception as e:
                print(f"⚠️ Error creating plot for {ticker}: {str(e)}")

    def calculate_returns(self) -> None:
        """Calculate return metrics from price and dividend data"""
        print("\nCalculating returns...")
        
        # Validate that we have data to work with
        if self.price_data is None or self.price_data.empty:
            raise ValueError("No price data available. Cannot calculate returns.")
            
        # Check for tickers without data
        empty_tickers = [ticker for ticker in self.tickers 
                         if ticker not in self.price_data.columns]
        if empty_tickers:
            print(f"⚠️ The following tickers have no price data: {', '.join(empty_tickers)}")
            self.tickers = [t for t in self.tickers if t in self.price_data.columns]
            
        if not self.tickers:
            raise ValueError("No tickers with price data. Cannot calculate returns.")
            
        # Find common date range with sufficient data for all tickers
        aligned_data = self._align_data()
        
        if aligned_data is None:
            raise ValueError("Failed to align price data across tickers.")
        
        # Use the aligned data
        prices_df = aligned_data['prices']
        div_df = aligned_data['dividends']
        
        # CRITICAL FIX: Validate data shapes before calculations
        if prices_df.shape != div_df.shape:
            print(f"⚠️ Shape mismatch detected: prices={prices_df.shape}, dividends={div_df.shape}")
            # Ensure both DataFrames have the same index
            common_index = prices_df.index.intersection(div_df.index)
            if len(common_index) < min(len(prices_df), len(div_df)) * 0.9:  # If we lose >10% of data
                print(f"❌ Major index mismatch: common dates={len(common_index)}, prices={len(prices_df)}, dividends={len(div_df)}")
                print("Attempting to rebuild dividend data with price index...")
            
            # Re-index both DataFrames to the common dates
            prices_df = prices_df.loc[common_index]
            div_df = div_df.loc[common_index]
            
            # Final verification
            if prices_df.shape != div_df.shape:
                print(f"❌ Data alignment failed. Creating zero dividends with matching shape.")
                # Create a new dividends DataFrame with zeros that exactly matches prices
                div_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        
        # Verify tickers match in both DataFrames
        price_cols = set(prices_df.columns)
        div_cols = set(div_df.columns)
        if price_cols != div_cols:
            print(f"⚠️ Column mismatch: prices has {len(price_cols)} columns, dividends has {len(div_cols)} columns")
            # Use only columns that exist in both
            common_cols = list(price_cols.intersection(div_cols))
            prices_df = prices_df[common_cols]
            div_df = div_df[common_cols]
            # Update tickers list to reflect only the common columns
            self.tickers = [t for t in self.tickers if t in common_cols]
            print(f"Using {len(common_cols)} common tickers for calculations")
        
        # Identify synthetic assets
        synthetic_tickers = [t for t in self.tickers if t in ['CASH', 'TBILLS']]
        
        # Calculate returns for regular tickers
        price_returns = prices_df.pct_change().fillna(0)
        div_returns = (div_df / prices_df.shift(1)).fillna(0)
        total_returns = price_returns.add(div_returns)
        
        # Override returns for synthetic assets with their theoretical returns
        for ticker in synthetic_tickers:
            if ticker in total_returns.columns:
                # Set constant daily return for synthetic assets
                if ticker == 'CASH':
                    daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
                    total_returns[ticker] = daily_rf
                elif ticker == 'TBILLS':
                    daily_rf = (1 + self.risk_free_rate * 1.15) ** (1/252) - 1
                    total_returns[ticker] = daily_rf
        
        # Calculate annualized returns
        self.mean_returns = total_returns.mean() * 252
        
        # Calculate covariance matrix
        self.cov_matrix = total_returns.cov() * 252
        
        # Ensure covariance matrix is well-behaved for cash-like assets
        for ticker in synthetic_tickers:
            if ticker in self.tickers:
                # Get the index for this ticker
                idx = list(self.tickers).index(ticker)
                
                # Override covariance values with more distinction between CASH and TBILLS
                if ticker == 'CASH':
                    variance = 1e-8  # Very low volatility for cash
                else:  # TBILLS
                    variance = 0.0025  # Increased variance for T-bills to reflect a standard deviation of approximately 5%
                
                # Set near-zero correlation with all other assets
                for i in range(len(self.tickers)):
                    self.cov_matrix.iloc[idx, i] = 0
                    self.cov_matrix.iloc[i, idx] = 0
                    
                # Set tiny variance for the asset itself
                self.cov_matrix.iloc[idx, idx] = variance
        
        # Print return metrics for debugging
        print("\nAnnualized Return Metrics:")
        print("-" * 50)
        
        # Create a returns summary dataframe
        returns_summary = pd.DataFrame(index=self.tickers)
        returns_summary['AnnReturn'] = self.mean_returns * 100
        returns_summary['AnnVolatility'] = np.sqrt(np.diag(self.cov_matrix)) * 100
        returns_summary['Sharpe'] = (self.mean_returns - self.risk_free_rate) / np.sqrt(np.diag(self.cov_matrix))
        
        # Store the returns summary for later use
        self.returns_summary = returns_summary
        
        # Save returns summary
        summary_file = os.path.join(self.output_dir, "returns_summary.csv")
        returns_summary.to_csv(summary_file)
        print(f"Returns summary saved to {summary_file}")
        
        # Save correlation matrix
        if total_returns is not None:
            corr_matrix = total_returns.corr()
            corr_file = os.path.join(self.output_dir, "correlations.csv")
            corr_matrix.to_csv(corr_file)
            print(f"Correlation matrix saved to {corr_file}")
        
        # Plot returns comparison
        self._plot_returns_comparison(returns_summary)
        
        # Only print individual ticker returns in debug mode
        if self.debug:
            for ticker in self.tickers:
                print(f"{ticker}: Return = {self.mean_returns[ticker]*100:.2f}%, " +
                     f"Volatility = {np.sqrt(self.cov_matrix.loc[ticker, ticker])*100:.2f}%")

    def _align_data(self):
        """Align price and dividend data to ensure all tickers have data for the same dates"""
        start_time = time.time()
        print("Aligning data across tickers...")
        
        # First identify synthetic assets
        synthetic_tickers = [t for t in self.tickers if t in ['CASH', 'TBILLS']]
        real_tickers = [t for t in self.tickers if t not in ['CASH', 'TBILLS']]
        
        # Handle case where there are no real tickers (only synthetic)
        if not real_tickers:
            print("⚠️ No real tickers found, only synthetic assets")
            
            # In this case, just use the synthetic data as is
            price_df = self.price_data[self.tickers].copy()
            div_df = self.div_data[self.tickers].copy()
            
            return {
                'prices': price_df,
                'dividends': div_df
            }
        
        # Process real ticker data - use vectorized operations where possible
        price_df = self.price_data[real_tickers].copy()
        
        # CRITICAL FIX: Handle future dates issue
        current_date = pd.Timestamp.now().normalize()  # Normalize to remove time component
        
        print(f"Full date range before filtering: {price_df.index.min()} to {price_df.index.max()}")
        
        # Filter out future dates - vectorized operation
        if price_df.index.max() > current_date:
            price_df = price_df.loc[price_df.index <= current_date]
            print(f"Date range after filtering future dates: {price_df.index.min()} to {price_df.index.max()}")
        
        # Ensure date format is consistent
        price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d')
        
        # Calculate missing data percentages vectorized
        missing_pcts = price_df.isna().mean() * 100
        
        # Only print detailed missing data report in debug mode
        if self.debug:
            print("\nMissing data report BEFORE filtering:")
            for ticker, pct in missing_pcts.items():
                print(f"{ticker}: {price_df[ticker].isna().sum()} missing values ({pct:.1f}%)")
        else:
            print("\nAnalyzing data quality (use --debug for details)...")
            print(f"Average missing data: {missing_pcts.mean():.1f}%")
        
        # Check if ALL tickers have high missing data
        high_missing_threshold = 45.0
        all_high_missing = (missing_pcts > high_missing_threshold).all()
        
        if all_high_missing:
            print("\n⚠️ All tickers have high missing data percentages. This suggests a systematic issue.")
            
            # Find common valid date range more efficiently
            valid_data = ~price_df.isna()
            if valid_data.any().any():
                # For each ticker, get first and last valid index
                first_valid = pd.Series({col: valid_data[col][valid_data[col]].index.min() 
                                        for col in valid_data.columns if valid_data[col].any()})
                last_valid = pd.Series({col: valid_data[col][valid_data[col]].index.max() 
                                      for col in valid_data.columns if valid_data[col].any()})
                
                if not first_valid.empty and not last_valid.empty:
                    latest_start = first_valid.max()
                    earliest_end = last_valid.min()
                    
                    if latest_start <= earliest_end:
                        print(f"\n🔄 Attempting to salvage data by using common date range:")
                        print(f"   {latest_start.date()} to {earliest_end.date()}")
                        
                        # Filter to this date range
                        price_df = price_df.loc[latest_start:earliest_end]
                        
                        # Check quality of this subset
                        missing_pcts = price_df.isna().mean() * 100
                        print("\nMissing data after date range restriction:")
                        for ticker, pct in missing_pcts.items():
                            print(f"{ticker}: {price_df[ticker].isna().sum()} missing values ({pct:.1f}%)")
        
        # If we still have a systematic issue, use more efficient ticker selection
        all_high_missing = (missing_pcts > high_missing_threshold).all()
        if all_high_missing:
            # Sort tickers by missing percentage and take top N
            sorted_tickers = missing_pcts.sort_values().index.tolist()
            keep_count = max(10, int(len(sorted_tickers) * 0.2))
            keep_count = min(keep_count, len(sorted_tickers))
            
            keep_tickers = sorted_tickers[:keep_count]
            print(f"Keeping top {keep_count} tickers with least missing data:")
            for ticker in keep_tickers:
                print(f"- {ticker}: {missing_pcts[ticker]:.1f}% missing")
            
            # Adjust the dataframe to only include these tickers
            price_df = price_df[keep_tickers]
            real_tickers = keep_tickers
        else:
            # More efficient filtering
            valid_mask = missing_pcts <= 45
            valid_tickers = missing_pcts[valid_mask].index.tolist()
            
            if len(valid_tickers) < len(real_tickers):
                print(f"Proceeding with {len(valid_tickers)} out of {len(real_tickers)} real tickers after quality check")
                if valid_tickers:
                    price_df = price_df[valid_tickers]
                    real_tickers = valid_tickers
        
        if len(real_tickers) == 0:
            print("❌ No valid real tickers with sufficient data.")
            return None
        
        # Create dividends DataFrame more efficiently
        # Pre-create a DataFrame with zeros and then fill it
        div_df = pd.DataFrame(0.0, index=price_df.index, columns=real_tickers)
        
        # Update with actual dividend data where available
        for ticker in real_tickers:
            if ticker in self.div_data:
                common_idx = price_df.index.intersection(self.div_data[ticker].index)
                if not common_idx.empty:
                    div_df.loc[common_idx, ticker] = self.div_data[ticker].loc[common_idx]
        
        # Apply forward and backward fill in one step
        price_df = price_df.ffill().bfill()
        
        # CRITICAL FIX: Ensure synthetic assets are created with EXACT same index as real data
        for ticker in synthetic_tickers:
            # Remove any existing synthetic data to avoid shape issues
            if ticker in self.price_data.columns:
                self.price_data = self.price_data.drop(ticker, axis=1)
            if ticker in self.div_data.columns:
                self.div_data = self.div_data.drop(ticker, axis=1)
            
            # Create new synthetic data with exact same index as price_df
            self._create_synthetic_asset(ticker, price_df.index)
            
            # Add the new synthetic data to our DataFrames
            price_df[ticker] = self.price_data[ticker].loc[price_df.index]
            div_df[ticker] = self.div_data[ticker].loc[price_df.index]
            print(f"✅ {ticker}: Successfully added synthetic data to aligned dataset")
        
        # Verify final shapes match
        if price_df.shape != div_df.shape:
            print(f"❌ CRITICAL ERROR: Final shapes don't match: price={price_df.shape}, div={div_df.shape}")
            # Force alignment as last resort
            common_index = price_df.index.intersection(div_df.index)
            price_df = price_df.loc[common_index]
            div_df = div_df.loc[common_index]
        
        # Update ticker list 
        self.tickers = price_df.columns.tolist()
            
        # Print data quality summary
        nan_counts = price_df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️ WARNING: Still have {nan_counts.sum()} NaN values after alignment!")
            for ticker, count in nan_counts[nan_counts > 0].items():
                print(f"  {ticker}: {count} NaN values ({count/len(price_df)*100:.1f}%)")
        else:
            print(f"✅ No NaN values in aligned data - good for optimization")
            
        print(f"✅ Successfully aligned data for {len(self.tickers)} tickers, covering {len(price_df)} dates")
        print(f"   Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
        
        # Track performance
        self.performance_stats['align_data_time'] = time.time() - start_time
            
        return {
            'prices': price_df,
            'dividends': div_df
        }

    def portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        ret = np.sum(self.mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        vol = max(vol, 1e-8)  # Prevent division by zero
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe

    def optimize_portfolio(self, exclude_cash: bool = False, skip_plots: bool = False) -> Dict:
        """
        Optimize portfolio weights for maximum Sharpe ratio
        
        Parameters:
            exclude_cash: If True, exclude CASH and TBILLS from optimization
            skip_plots: If True, skip generating plots (faster)
        """
        optimization_start = time.time()
        start_time = time.time()
        print("\nOptimizing portfolio allocation...")
        
        # CRITICAL FIX 6: Add more robust pre-checks before optimization
        if self.mean_returns is None or self.cov_matrix is None:
            print("❌ No return data available for optimization.")
            return None
        
        # CRITICAL FIX 7: Verify data dimensions match before optimization
        if len(self.tickers) != len(self.mean_returns) or len(self.tickers) != self.cov_matrix.shape[0]:
            print(f"❌ Dimension mismatch: tickers={len(self.tickers)}, returns={len(self.mean_returns)}, cov={self.cov_matrix.shape}")
            return None
        
        # Check for problematic values in returns and covariance
        if np.isnan(self.mean_returns).any():
            nan_tickers = [t for t, val in zip(self.tickers, np.isnan(self.mean_returns)) if val]
            print(f"❌ The following tickers have NaN returns: {', '.join(nan_tickers)}")
            # Filter out problematic tickers
            good_indices = ~np.isnan(self.mean_returns)
            self.tickers = [t for i, t in enumerate(self.tickers) if good_indices[i]]
            self.mean_returns = self.mean_returns[good_indices]
            self.cov_matrix = self.cov_matrix.loc[self.tickers, self.tickers]
            
        # CRITICAL FIX 8: Better handling of covariance matrix issues
        if np.isnan(self.cov_matrix).any().any():
            print("❌ The covariance matrix contains NaN values. Attempting to fix...")
            
            # Replace NaN values with zeros in correlation matrix then rebuild covariance
            # This is more robust than just removing tickers
            try:
                # Get volatilities (standard deviations)
                vols = np.sqrt(np.diagonal(self.cov_matrix))
                
                # Calculate correlation matrix
                corr_matrix = self.cov_matrix.copy()
                for i in range(len(self.tickers)):
                    for j in range(len(self.tickers)):
                        if vols[i] > 0 and vols[j] > 0:
                            corr_matrix.iloc[i, j] = self.cov_matrix.iloc[i, j] / (vols[i] * vols[j])
                
                # Replace NaN correlations with zeros (no correlation)
                corr_matrix = corr_matrix.fillna(0)
                
                # Rebuild covariance matrix
                for i in range(len(self.tickers)):
                    for j in range(len(self.tickers)):
                        self.cov_matrix.iloc[i, j] = corr_matrix.iloc[i, j] * vols[i] * vols[j]
                        
                print("✅ Successfully repaired covariance matrix")
                
            except Exception as e:
                print(f"❌ Error repairing covariance matrix: {str(e)}")
                # Fall back to removing problematic tickers
                problem_rows = np.isnan(self.cov_matrix).any(axis=1)
                nan_tickers = [t for i, t in enumerate(self.tickers) if problem_rows[i]]
                print(f"Removing problematic tickers: {', '.join(nan_tickers)}")
                
                # Remove problematic tickers
                good_tickers = [t for t in self.tickers if t not in nan_tickers]
                if not good_tickers:
                    print("❌ No valid tickers left after removing problematic ones.")
                    return None
                    
                self.tickers = good_tickers
                self.mean_returns = self.mean_returns[self.tickers]
                self.cov_matrix = self.cov_matrix.loc[self.tickers, self.tickers]
        
        # CRITICAL FIX 9: Now check for more extreme values and filter if needed
        extreme_return_threshold = 5.0  # 500% annualized return is extreme
        extreme_vols = np.sqrt(np.diag(self.cov_matrix))
        extreme_vol_threshold = 2.0  # 200% volatility is extreme
        
        extreme_return_mask = np.abs(self.mean_returns) > extreme_return_threshold
        extreme_vol_mask = extreme_vols > extreme_vol_threshold
        extreme_mask = extreme_return_mask | extreme_vol_mask
        
        if extreme_mask.any():
            extreme_tickers = [t for i, t in enumerate(self.tickers) if extreme_mask[i]]
            print(f"⚠️ Found {len(extreme_tickers)} tickers with extreme statistics:")
            for ticker in extreme_tickers:
                idx = self.tickers.index(ticker)
                print(f"  {ticker}: Return={self.mean_returns[idx]*100:.1f}%, Volatility={extreme_vols[idx]*100:.1f}%")
                
            print(f"Proceeding with optimization, but consider removing these tickers if optimization fails")
        
        # The rest of the optimization continues as before
        if exclude_cash:
            opt_tickers = [t for t in self.tickers if t not in ['CASH', 'TBILLS']]
            if not opt_tickers:
                print("❌ No equity tickers available for optimization after excluding cash.")
                return None
                
            print(f"Optimizing {len(opt_tickers)} tickers (excluding cash assets)")
                
            # Extract relevant returns and covariance
            opt_returns = self.mean_returns[opt_tickers].values  # Extract values for faster operations
            opt_cov = self.cov_matrix.loc[opt_tickers, opt_tickers].values
        else:
            opt_tickers = self.tickers
            opt_returns = self.mean_returns.values  # Extract values for faster operations  
            opt_cov = self.cov_matrix.values
            print(f"Optimizing {len(opt_tickers)} tickers")
        
        n_assets = len(opt_tickers)
        
        # Set up constraints based on whether shorts are allowed
        if self.shorts:
            print("Enabling short selling (allowing negative weights)")
            print(f"Maximum long exposure: {self.max_long*100:.0f}%, Maximum short exposure: {self.max_short*100:.0f}%")
            
            # Reparameterize weights for long-short optimization
            # Each asset gets two variables: one for long position, one for short position
            # Instead of bounds from -1 to 1, we'll have bounds from 0 to max_long for long positions
            # and 0 to max_short for short positions
            bounds = tuple([(0, self.max_long) for _ in range(n_assets)] + 
                           [(0, self.max_short) for _ in range(n_assets)])
            
            # Define constraint for gross exposure = 1
            # w_1^+ + w_2^+ + ... + w_n^+ + w_1^- + w_2^- + ... + w_n^- = 1
            sum_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            # Define objective function that works with the 2n variables
            def negative_sharpe_reparameterized(x):
                # Extract long and short components
                long_weights = x[:n_assets]
                short_weights = x[n_assets:]
                
                # Calculate net weights: w_i = w_i^+ - w_i^-
                net_weights = long_weights - short_weights
                
                # Calculate return and volatility using net weights
                ret = np.sum(opt_returns * net_weights)
                vol = np.sqrt(np.dot(net_weights.T, np.dot(opt_cov, net_weights)))
                vol = max(vol, 1e-8)  # Prevent division by zero
                sharpe = (ret - self.risk_free_rate) / vol
                return -sharpe
                
            # Create a balanced initial guess that respects constraints
            # Allocate 80% of the weights to the long part and 20% to the short part
            init_guess = np.zeros(2 * n_assets)
            
            # For long weights: use 80% of available weight space
            long_weights_raw = np.random.dirichlet(np.ones(n_assets)) * self.max_long * 0.95
            
            # For short weights: use 20% of available weight space
            short_weights_raw = np.random.dirichlet(np.ones(n_assets)) * self.max_short * 0.95
            
            # Combine them ensuring the total sums to 1
            total_weight = np.sum(long_weights_raw) + np.sum(short_weights_raw)
            scaling_factor = 1.0 / total_weight
            
            # Set initial guess
            init_guess[:n_assets] = long_weights_raw * scaling_factor
            init_guess[n_assets:] = short_weights_raw * scaling_factor
            
            # Validate the initial guess
            total_init = np.sum(init_guess)
            print(f"Initial guess: Long={np.sum(init_guess[:n_assets]):.3f}, Short={np.sum(init_guess[n_assets:]):.3f}, Total={total_init:.3f}")
            
            # Use the reparameterized negative_sharpe function for optimization
            constraints = (sum_constraint,)
            
            # Use more aggressive optimization settings
            optimization_start = time.time()
            for attempt in range(5):
                try:
                    if attempt > 0:
                        # Try different initialization strategies on subsequent attempts
                        if attempt == 1:
                            # Try focusing on high Sharpe assets for long, low Sharpe for short
                            sharpes = (opt_returns - self.risk_free_rate) / np.sqrt(np.diag(opt_cov))
                            
                            # Normalize Sharpe ratios to [0,1] range
                            normalized_sharpes = (sharpes - np.min(sharpes)) / (np.max(sharpes) - np.min(sharpes) + 1e-8)
                            
                            # Use normalized Sharpe ratios for weighting
                            long_weights = normalized_sharpes * self.max_long * 0.95
                            short_weights = (1 - normalized_sharpes) * self.max_short * 0.95
                            
                            # Normalize to ensure sum equals 1
                            total = np.sum(long_weights) + np.sum(short_weights)
                            long_weights = long_weights / total
                            short_weights = short_weights / total
                            
                            init_guess = np.concatenate([long_weights, short_weights])
                        
                        elif attempt == 2:
                            # Try completely random weights, still respecting constraints
                            init_guess = np.random.random(2 * n_assets)
                            # Scale long and short parts separately
                            long_sum = np.sum(init_guess[:n_assets])
                            short_sum = np.sum(init_guess[n_assets:])
                            
                            if long_sum > 0:
                                init_guess[:n_assets] = init_guess[:n_assets] / long_sum * self.max_long * 0.95
                            if short_sum > 0:
                                init_guess[n_assets:] = init_guess[n_assets:] / short_sum * self.max_short * 0.95
                            
                            # Normalize to sum to 1
                            init_guess = init_guess / np.sum(init_guess)
                        
                        print(f"Optimization attempt {attempt+1}: Using alternative initial weights")
                        print(f"Initial guess: Long={np.sum(init_guess[:n_assets]):.3f}, Short={np.sum(init_guess[n_assets:]):.3f}, Total={np.sum(init_guess):.3f}")
                    
                    # Use more aggressive optimization settings
                    result = minimize(negative_sharpe_reparameterized, init_guess, method='SLSQP',
                                 bounds=bounds, constraints=constraints,
                                 options={'maxiter': 10000, 'ftol': 1e-10, 'disp': True})
                    
                    if result['success']:
                        print(f"✅ Optimization successful after {attempt+1} attempt(s)")
                        break
                    else:
                        print(f"⚠️ Attempt {attempt+1} failed: {result['message']}")
                        # Continue to next attempt with different initialization
                
                except Exception as e:
                    print(f"❌ Error during optimization attempt {attempt+1}: {str(e)}")
                    if attempt == 4:  # Last attempt
                        print("\nOptimization diagnostics:")
                        print(f"Shape of returns vector: {np.shape(opt_returns)}")
                        print(f"Shape of covariance matrix: {np.shape(opt_cov)}")
                        return None
            
            # Extract the optimized weights
            if result['success']:
                long_weights = result.x[:n_assets]
                short_weights = result.x[n_assets:]
                
                # Calculate net weights
                optimized_weights = long_weights - short_weights
                
                # Perform post-optimization rescaling to ensure constraints are exactly met
                # Calculate actual long and short exposures
                actual_long = np.sum(np.maximum(optimized_weights, 0))
                actual_short = np.sum(np.abs(np.minimum(optimized_weights, 0)))
                
                print(f"Optimized weights before rescaling:")
                print(f"Long exposure: {actual_long:.4f}, Short exposure: {actual_short:.4f}")
                print(f"Net exposure: {np.sum(optimized_weights):.4f}")
                
                # If we exceed constraints, scale back proportionally
                if actual_long > self.max_long or actual_short > self.max_short:
                    print("Rescaling weights to strictly enforce exposure limits...")
                    long_scale = min(1.0, self.max_long / max(actual_long, 1e-8))
                    short_scale = min(1.0, self.max_short / max(actual_short, 1e-8))
                    
                    # Apply scaling
                    positive_weights = np.maximum(optimized_weights, 0) * long_scale
                    negative_weights = np.minimum(optimized_weights, 0) * short_scale
                    
                    # Combine
                    optimized_weights = positive_weights + negative_weights
                    
                    # Ensure the net sum is still 1.0 (can be slightly off due to scaling)
                    optimized_weights = optimized_weights / np.sum(optimized_weights) if np.sum(optimized_weights) != 0 else optimized_weights
                    
                    # Calculate final exposures after scaling
                    final_long = np.sum(np.maximum(optimized_weights, 0))
                    final_short = np.sum(np.abs(np.minimum(optimized_weights, 0)))
                    print(f"Exposures after rescaling: Long={final_long:.4f}, Short={final_short:.4f}, Net={np.sum(optimized_weights):.4f}")
                
                # Create the proper results dictionary instead of returning raw result
                # Create dictionary of tickers to weights
                weight_dict = dict(zip(opt_tickers, optimized_weights))
                
                # Calculate portfolio metrics with these weights
                weights_array = np.array(optimized_weights)
                port_ret = np.sum(opt_returns * weights_array)
                port_vol = np.sqrt(np.dot(weights_array.T, np.dot(opt_cov, weights_array)))
                port_sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0
                
                # Create results dictionary
                results = {
                    'weights': weight_dict,
                    'return': port_ret,
                    'volatility': port_vol,
                    'sharpe': port_sharpe,
                    'kelly_metrics': self.calculate_kelly(port_ret, port_vol)
                }
                
                # If we excluded cash, reincorporate cash assets with 0% allocation
                if exclude_cash:
                    full_weights = {}
                    for ticker in self.tickers:
                        if ticker in weight_dict:
                            full_weights[ticker] = weight_dict[ticker]
                        else:
                            full_weights[ticker] = 0.0
                    
                    # Recalculate metrics with the full set of tickers
                    weights_array = np.array([full_weights[t] for t in self.tickers])
                    port_ret, port_vol, port_sharpe = self.portfolio_metrics(weights_array)
                    
                    results = {
                        'weights': full_weights,
                        'return': port_ret,
                        'volatility': port_vol,
                        'sharpe': port_sharpe,
                        'kelly_metrics': self.calculate_kelly(port_ret, port_vol)
                    }
            else:
                print(f"❌ All optimization attempts failed")
                return None
        else:
            # Original long-only implementation
            bounds = tuple((0, 1) for _ in range(n_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            initial_weights = np.array([1/n_assets] * n_assets)

            # Standard optimization function
            def negative_sharpe(weights):
                ret = np.sum(opt_returns * weights)
                vol = np.sqrt(np.dot(weights.T, np.dot(opt_cov, weights)))
                vol = max(vol, 1e-8)
                sharpe = (ret - self.risk_free_rate) / vol
                return -sharpe
                
            result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result['success']:
                optimized_weights = result.x
                results = self.get_optimization_results(optimized_weights)
            else:
                print(f"❌ Optimization failed: {result['message']}")
                return None

        self.performance_stats['optimization_time'] = time.time() - optimization_start
            
        # Save optimization results
        if results:
            # Save weights as JSON
            weights_file = os.path.join(self.output_dir, "optimal_weights.json")
            with open(weights_file, 'w') as f:
                weights_pct = {ticker: weight*100 for ticker, weight in results['weights'].items()}
                json.dump(weights_pct, f, indent=2)
            print(f"Optimal weights saved to {weights_file}")
            
            # Print long and short allocations when shorts are enabled
            if self.shorts:
                # Separate long and short positions
                long_positions = {k: v for k, v in results['weights'].items() if v > 0}
                short_positions = {k: v for k, v in results['weights'].items() if v < 0}
                
                # Calculate sums
                long_sum = sum(long_positions.values()) * 100
                short_sum = abs(sum(short_positions.values())) * 100
                
                print(f"\nPortfolio Allocations with Short Selling:")
                print(f"Total Long: {long_sum:.2f}% | Total Short: {short_sum:.2f}% | Net: 100.00%")
                
                if long_sum > self.max_long * 100 + 0.01 or short_sum > self.max_short * 100 + 0.01:
                    print("⚠️ Warning: Exposure limits have been exceeded slightly due to optimization precision.")
                    print("   Consider re-running the optimization with stricter constraints.")
                
                print("\nLong Positions:")
                for ticker, weight in sorted(long_positions.items(), key=lambda x: -x[1]):
                    if weight > 0.001:  # Only show positions with at least 0.1% allocation
                        print(f"{ticker}: {weight*100:.2f}%")
                
                print("\nShort Positions:")
                for ticker, weight in sorted(short_positions.items(), key=lambda x: x[1]):  # Sort by most negative
                    if abs(weight) > 0.001:  # Only show positions with at least 0.1% allocation
                        print(f"{ticker}: {weight*100:.2f}%")
        
            # Skip plotting for faster performance if requested
            if not skip_plots:
                # Plot portfolio allocation
                self._plot_portfolio_weights(results['weights'])
                
                # Plot efficient frontier
                self._plot_efficient_frontier(results)
        
        self.performance_stats['total_optimization_time'] = time.time() - start_time
        print(f"Portfolio optimization completed in {self.performance_stats['total_optimization_time']:.2f} seconds")
            
        return results

    def calculate_kelly(self, portfolio_ret: float, portfolio_vol: float) -> Dict:
        """Calculate Kelly Criterion and leveraged metrics"""
        portfolio_vol = max(portfolio_vol, 1e-8)
        kelly_fraction = (portfolio_ret - self.margin_cost_rate) / portfolio_vol**2
        safe_kelly = min(kelly_fraction, 2.0)  # Limit leverage to 2x for safety

        leveraged_return = safe_kelly * portfolio_ret - (safe_kelly - 1) * self.margin_cost_rate
        leveraged_volatility = safe_kelly * portfolio_vol
        leveraged_sharpe = (leveraged_return - self.risk_free_rate) / leveraged_volatility if leveraged_volatility > 0 else 0.0

        return {
            'kelly_fraction': kelly_fraction,
            'safe_kelly': safe_kelly,
            'leveraged_return': leveraged_return,
            'leveraged_volatility': leveraged_volatility,
            'leveraged_sharpe': leveraged_sharpe
        }

    def get_optimization_results(self, optimal_weights: np.ndarray) -> Dict:
        """Compile all optimization results"""
        port_ret, port_vol, port_sharpe = self.portfolio_metrics(optimal_weights)
        kelly_metrics = self.calculate_kelly(port_ret, port_vol)
        
        return {
            'weights': dict(zip(self.tickers, optimal_weights)),
            'return': port_ret,
            'volatility': port_vol,
            'sharpe': port_sharpe,
            'kelly_metrics': kelly_metrics
        }

    def _plot_portfolio_weights(self, weights: Dict[str, float]) -> None:
        """Plot portfolio weights as a pie chart - handling both long and short positions"""
        if HAS_GRAPHING:
            try:
                plot_portfolio_weights(weights, output_dir=self.output_dir, allow_shorts=self.shorts)
            except Exception as e:
                print(f"⚠️ Error creating portfolio weights plot: {str(e)}")
        else:
            try:
                # Convert weights to percentages for display
                weights_pct = {k: v*100 for k, v in weights.items()}
                
                # Separate long and short positions
                long_weights = {k: v for k, v in weights_pct.items() if v > 0}
                short_weights = {k: abs(v) for k, v in weights_pct.items() if v < 0}
                
                # Count number of plots needed
                has_long = len(long_weights) > 0
                has_short = len(short_weights) > 0
                
                # Determine filename suffix based on shorts being allowed and used
                filename_suffix = "_shorts" if self.shorts and has_short else ""
                
                if has_long and has_short:
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                    fig.suptitle('Optimal Portfolio Allocation (Long-Short Strategy)', fontsize=16)
                    
                    # Plot long positions
                    # Filter weights for readability (only include weights > 1%)
                    significant_long = {k: v for k, v in long_weights.items() if v > 1}
                    other_long = sum(v for k, v in long_weights.items() if v <= 1)
                    
                    if other_long > 0:
                        significant_long['Other (Long)'] = other_long
                    
                    ax1.pie(significant_long.values(), labels=significant_long.keys(), 
                           autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Long Positions')
                    ax1.axis('equal')
                    
                    # Plot short positions
                    significant_short = {k: v for k, v in short_weights.items() if v > 1}
                    other_short = sum(v for k, v in short_weights.items() if v <= 1)
                    
                    if other_short > 0:
                        significant_short['Other (Short)'] = other_short
                    
                    ax2.pie(significant_short.values(), labels=significant_short.keys(), 
                           autopct='%1.1f%%', startangle=90)
                    ax2.set_title('Short Positions')
                    ax2.axis('equal')
                    
                elif has_long:
                    # Only long positions, create single plot
                    plt.figure(figsize=(10, 8))
                    
                    # Filter weights for readability (only include weights > 1%)
                    significant_weights = {k: v for k, v in long_weights.items() if v > 1}
                    other_weight = sum(v for k, v in long_weights.items() if v <= 1)
                    
                    if other_weight > 0:
                        significant_weights['Other'] = other_weight
                    
                    plt.pie(significant_weights.values(), labels=significant_weights.keys(), 
                           autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title('Optimal Portfolio Allocation (Long Only)')
                    
                elif has_short:
                    # Only short positions (unusual but possible)
                    plt.figure(figsize=(10, 8))
                    
                    # Filter weights for readability (only include weights > 1%)
                    significant_weights = {k: v for k, v in short_weights.items() if v > 1}
                    other_weight = sum(v for k, v in short_weights.items() if v <= 1)
                    
                    if other_weight > 0:
                        significant_weights['Other'] = other_weight
                    
                    plt.pie(significant_weights.values(), labels=significant_weights.keys(), 
                           autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title('Optimal Portfolio Allocation (Short Only)')
                
                # Save plot with appropriate suffix
                plot_file = os.path.join(self.output_dir, f"portfolio_allocation{filename_suffix}.png")
                plt.savefig(plot_file)
                plt.close()
                print(f"Portfolio allocation chart saved to {plot_file}")
                
            except Exception as e:
                print(f"⚠️ Error creating portfolio weights plot: {str(e)}")

    def _plot_efficient_frontier(self, optimal_result: Dict) -> None:
        """Plot the efficient frontier with the optimal portfolio"""
        start_time = time.time()
        if HAS_GRAPHING:
            try:
                plot_efficient_frontier(
                    self.mean_returns, 
                    self.cov_matrix, 
                    self.tickers, 
                    optimal_result, 
                    self.risk_free_rate,
                    self.shorts, 
                    self.max_long, 
                    self.max_short,
                    output_dir=self.output_dir
                )
            except Exception as e:
                print(f"⚠️ Error creating efficient frontier plot: {str(e)}")
        else:
            try:
                # Reduce the number of points for faster computation
                num_points = 30
                target_returns = np.linspace(min(self.mean_returns), max(self.mean_returns), num_points)
                efficient_volatilities = []
                
                # Convert to numpy arrays for faster computation
                mean_returns_array = self.mean_returns.values
                cov_matrix_array = self.cov_matrix.values
                
                # Set bounds based on whether shorts are allowed
                n_assets = len(self.mean_returns)
                if self.shorts:
                    bounds = tuple((-1, 1) for _ in range(n_assets))
                    
                    # Define a constraint for the sum of weights being 1 (net exposure)
                    sum_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                    
                    # Define constraints for maximum long and short exposure
                    def long_exposure_constraint(weights):
                        # Sum of all positive weights should be <= max_long
                        return self.max_long - np.sum(np.maximum(weights, 0))
                        
                    def short_exposure_constraint(weights):
                        # Sum of absolute values of all negative weights should be <= max_short
                        return self.max_short - np.sum(np.abs(np.minimum(weights, 0)))
                        
                    constraints = [
                        sum_constraint,
                        {'type': 'ineq', 'fun': long_exposure_constraint},
                        {'type': 'ineq', 'fun': short_exposure_constraint},
                        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns_array * x) - target}
                    ]
                else:
                    bounds = tuple((0, 1) for _ in range(n_assets))
                    # Constraint (sum of weights = 1 and return = target)
                    constraints = [
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns_array * x) - target}
                    ]
                
                # Calculate volatility for each target return
                for target in target_returns:
                    # Minimize volatility subject to target return
                    def portfolio_volatility(weights):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_array, weights)))
                    
                    init_guess = np.array([1.0/n_assets] * n_assets)
                    
                    try:
                        result = minimize(portfolio_volatility, init_guess, 
                                        method='SLSQP', bounds=bounds, constraints=constraints)
                        efficient_volatilities.append(result['fun'])
                    except:
                        efficient_volatilities.append(np.nan)
                
                # Filter out any failed optimizations
                valid = ~np.isnan(efficient_volatilities)
                efficient_volatilities = np.array(efficient_volatilities)[valid]
                efficient_returns = target_returns[valid]
                
                # Plot efficient frontier
                plt.figure(figsize=(12, 8))
                plt.plot(efficient_volatilities * 100, efficient_returns * 100, 'b-', label='Efficient Frontier')
                
                # Plot optimal portfolio
                opt_return = optimal_result['return'] * 100
                opt_vol = optimal_result['volatility'] * 100
                plt.scatter(opt_vol, opt_return, s=100, color='r', marker='*', label='Optimal Portfolio')
                
                # Plot individual assets
                vols = np.sqrt(np.diag(self.cov_matrix)) * 100
                plt.scatter(vols, self.mean_returns.values * 100, s=50, alpha=0.7, label='Individual Assets')
                
                # Add labels for assets - FIX: Use iloc instead of integer indexing
                for i, ticker in enumerate(self.tickers):
                    plt.annotate(ticker, xy=(vols[i], self.mean_returns.iloc[i] * 100), xytext=(5, 5), 
                                textcoords='offset points', fontsize=9)
                
                # Add labels and title
                plt.xlabel('Annualized Volatility (%)')
                plt.ylabel('Annualized Return (%)')
                plt.title('Efficient Frontier and Optimal Portfolio')
                plt.grid(True)
                plt.legend()
                
                # Save plot
                plot_file = os.path.join(self.output_dir, "efficient_frontier.png")
                plt.savefig(plot_file)
                plt.close()
                print(f"Efficient frontier plot saved to {plot_file}")
                
            except Exception as e:
                print(f"⚠️ Error creating efficient frontier plot: {str(e)}")
            
        self.performance_stats['efficient_frontier_time'] = time.time() - start_time

    def _plot_returns_comparison(self, returns_summary):
        """Plot comparison of returns and volatility"""
        if HAS_GRAPHING:
            try:
                plot_returns_comparison(returns_summary, self.risk_free_rate, 
                                       output_dir=self.output_dir, allow_shorts=self.shorts)
            except Exception as e:
                print(f"⚠️ Error creating returns comparison plot: {str(e)}")
        else:
            try:
                # Determine filename suffix based on shorts being allowed
                filename_suffix = "_shorts" if self.shorts else ""
                
                plt.figure(figsize=(12, 8))
                
                # Extract metrics
                returns = returns_summary['AnnReturn']
                vols = returns_summary['AnnVolatility']
                sharpes = returns_summary['Sharpe']
                
                # Create scatter plot
                plt.scatter(vols, returns, s=50, alpha=0.7)
                
                # Add labels for each point
                for ticker, ret, vol in zip(returns.index, returns, vols):
                    plt.annotate(ticker, xy=(vol, ret), xytext=(5, 5), 
                                textcoords='offset points', fontsize=9)
                
                # Add labels and title
                plt.xlabel('Annualized Volatility (%)')
                plt.ylabel('Annualized Return (%)')
                title = 'Risk-Return Profile of Assets'
                if self.shorts:
                    title += ' (Short Selling Enabled)'
                plt.title(title)
                plt.grid(True)
                
                # Add risk-free rate as horizontal line
                plt.axhline(y=self.risk_free_rate*100, color='r', linestyle='--', 
                           label=f'Risk-Free Rate ({self.risk_free_rate*100:.1f}%)')
                
                plt.legend()
                
                # Save plot with appropriate suffix
                plot_file = os.path.join(self.output_dir, f"risk_return_profile{filename_suffix}.png")
                plt.savefig(plot_file)
                plt.close()
                print(f"Risk-return profile saved to {plot_file}")
                
            except Exception as e:
                print(f"⚠️ Error creating returns comparison plot: {str(e)}")

    def standardize_cache_data(self, remove_future: bool = True) -> bool:
        """
        Standardize cached data to ensure consistent date formats and remove future dates
        
        Parameters:
            remove_future: Whether to remove future dates from cache files
            
        Returns:
            True if standardization was performed, False otherwise
        """
        if not HAS_STANDARDIZER:
            print("⚠️ Cache standardization module not available. Skipping standardization.")
            return False
            
        print("\n🔄 Standardizing cache data...")
        try:
            stats = standardize_cache(self.cache_dir, remove_future)
            if stats:
                print(f"✅ Standardized {stats.get('price_processed', 0)} price files and {stats.get('div_processed', 0)} dividend files")
                if stats.get('future_dates_removed', 0) > 0:
                    print(f"⚠️ Removed {stats.get('future_dates_removed', 0)} future dates from cache files")
                return True
        except Exception as e:
            print(f"⚠️ Error during cache standardization: {str(e)}")
        
        return False

# Import ML optimizer if available
try:
    from src.ml_optimizer import MLPortfolioOptimizer
    HAS_ML_OPTIMIZER = True
except ImportError:
    print("Warning: ML optimizer module not found. Using traditional optimization.")
    HAS_ML_OPTIMIZER = False

def main():
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Modern Portfolio Optimizer')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--convert-cache', action='store_true', help='Convert old cache to CSV format')  
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--risk-free', type=float, default=0.04, help='Risk-free rate')
    parser.add_argument('--margin-cost', type=float, default=0.065, help='Cost of margin')
    parser.add_argument('--tickers-file', type=str, default='tickers.csv', help='Path to tickers file')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before starting')
    parser.add_argument('--exclude-cash', action='store_true', 
                        help='Exclude cash and T-bills from optimization (equity-only)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
    parser.add_argument('--output-dir', type=str, default='portfolio_analysis', 
                        help='Directory for output files and visualizations')
    parser.add_argument('--batch-size', type=int, default=50, 
                        help='Batch size for data fetching')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of worker threads for data fetching')
    parser.add_argument('--cache-dir', type=str, default='data_cache',
                        help='Directory for cached data')
    # Add standardization control option
    parser.add_argument('--no-standardize', action='store_true', 
                        help='Skip cache standardization step')
    parser.add_argument('--keep-future-dates', action='store_true', 
                        help='Do not remove future dates during standardization')
    # Add new performance-related arguments
    parser.add_argument('--skip-plots', action='store_true', 
                        help='Skip generating plots for faster performance')
    parser.add_argument('--fast', action='store_true',
                        help='Enable all performance optimizations')
    # Add short selling options
    parser.add_argument('--shorts', action='store_true',
                        help='Allow short positions in the portfolio')
    parser.add_argument('--max-long', type=float, default=80.0,
                        help='Maximum percentage allocated to long positions (default: 80%%)')
    parser.add_argument('--max-short', type=float, default=20.0,
                        help='Maximum percentage allocated to short positions (default: 20%%)')
    # Add ML-related arguments
    parser.add_argument('--use-ml', action='store_true',
                        help='Use machine learning for portfolio optimization')
    
    args = parser.parse_args()
    
    # If fast mode is enabled, set all performance flags
    if args.fast:
        args.skip_plots = True
    
    # Clear cache if requested
    if args.clear_cache:
        cache = CSVDataCache(args.cache_dir)
        cache.clear_cache()
        print("✅ Cache cleared successfully")
    
    # Load tickers from CSV
    try:
        csv_path = os.path.join(os.path.dirname(__file__), args.tickers_file)
        tickers = load_tickers(csv_path)
        print(f"📋 Loaded {len(tickers)} tickers from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to load tickers: {str(e)}")
        return

    # Initialize and run optimization with timing information
    total_start_time = time.time()
    
    optimizer = PortfolioOptimizer(
        tickers, 
        risk_free_rate=args.risk_free,
        margin_cost_rate=args.margin_cost,
        years=args.years,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        debug=args.debug,
        shorts=args.shorts,  # Pass shorts argument
        max_long=args.max_long/100.0,  # Convert percentage to decimal
        max_short=args.max_short/100.0  # Convert percentage to decimal
    )
    
    print("🔄 Fetching data...")
    fetch_start_time = time.time()
    optimizer.fetch_data(
        years=args.years,
        use_cache=not args.no_cache,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    fetch_time = time.time() - fetch_start_time
    print(f"Data fetching completed in {fetch_time:.2f} seconds")
    
    try:
        # Standardize cache IMMEDIATELY after fetching data
        if not args.no_standardize and HAS_STANDARDIZER:
            print("\n🧹 Standardizing cache data BEFORE processing...")
            standardized = optimizer.standardize_cache_data(remove_future=not args.keep_future_dates)
            
            if standardized:
                # CRITICAL: Reload data from standardized cache
                print("🔄 Reloading data from standardized cache...")
                price_data_dict = {}
                div_data_dict = {}
                
                for ticker in optimizer.tickers:
                    # Reload each ticker's data from cache
                    cached_price = optimizer.cache.get_price_data(ticker)
                    cached_div = optimizer.cache.get_div_data(ticker)
                    
                    if cached_price is not None:
                        price_data_dict[ticker] = cached_price
                        
                    if cached_div is not None:
                        div_data_dict[ticker] = cached_div
                
                # Recreate DataFrames with standardized data
                if price_data_dict:
                    optimizer.price_data = pd.concat(price_data_dict, axis=1)
                if div_data_dict:
                    optimizer.div_data = pd.concat(div_data_dict, axis=1)
                
                print(f"✅ Successfully reloaded data for {len(price_data_dict)} tickers after standardization")
    except Exception as e:
        print(f"❌ Cache standardization error: {str(e)}")
    
    # Add debug information before calculation
    if args.debug:
        print("\nData diagnostics before calculating returns:")
        if optimizer.price_data is not None:
            print(f"- Number of tickers with price data: {len(optimizer.price_data.columns)}")
            print(f"- First 5 tickers: {', '.join(optimizer.price_data.columns[:5])}")
            print(f"- Date range: {optimizer.price_data.index.min()} to {optimizer.price_data.index.max()}")
            print(f"- Total dates: {len(optimizer.price_data.index)}")
            print(f"- Current date: {pd.Timestamp.now().normalize()}")
            print(f"- Future dates in data: {sum(optimizer.price_data.index > pd.Timestamp.now().normalize())}")
    
    print("🧮 Calculating returns...")
    returns_start_time = time.time()
    try:
        optimizer.calculate_returns()
        returns_time = time.time() - returns_start_time
        print(f"Returns calculation completed in {returns_time:.2f} seconds")
        
        # Display top 5 tickers by return if debug mode is enabled
        if args.debug and hasattr(optimizer, 'returns_summary'):
            print("\nAnnualized Return Metrics:")
            print("Top 5 tickers by return:")
            print("-" * 50)
            # Sort by AnnReturn in descending order and take top 5
            top_returns = optimizer.returns_summary.sort_values('AnnReturn', ascending=False).head(5)
            for ticker, row in top_returns.iterrows():
                print(f"{ticker}: Return = {row['AnnReturn']:.2f}%, " +
                     f"Volatility = {row['AnnVolatility']:.2f}%, " +
                     f"Sharpe = {row['Sharpe']:.2f}")
            print("-" * 50)
        
        print("🎯 Optimizing portfolio...")
        
        # Use ML optimizer if requested and available
        if args.use_ml and HAS_ML_OPTIMIZER:
            print("Using machine learning approach for portfolio optimization")
            
            # Create ML optimizer
            ml_optimizer = MLPortfolioOptimizer(
                optimizer.mean_returns,
                optimizer.cov_matrix,
                optimizer.tickers,
                risk_free_rate=args.risk_free,
                margin_cost_rate=args.margin_cost,
                shorts=args.shorts,
                max_long=args.max_long/100.0,
                max_short=args.max_short/100.0,
                debug=args.debug
            )
            
            # Run ML optimization
            result = ml_optimizer.optimize()
            
            # Plot results if needed
            if not args.skip_plots:
                optimizer._plot_portfolio_weights(result['weights'])
                optimizer._plot_efficient_frontier(result)
        else:
            # Use traditional optimization
            result = optimizer.optimize_portfolio(exclude_cash=args.exclude_cash, skip_plots=args.skip_plots)
        
        if result:
            # Print results
            print("\nOptimal Portfolio Allocation:")
            for ticker, weight in sorted(result['weights'].items(), key=lambda x: -x[1]):
                if weight > 0.001:  # Only show positions with at least 0.1% allocation
                    print(f"{ticker}: {weight*100:.2f}%")
            
            print(f"\nUnleveraged Portfolio Metrics:")
            print(f"Expected Return: {result['return']*100:.2f}%")
            print(f"Volatility: {result['volatility']*100:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe']:.2f}")
            
            k = result['kelly_metrics']
            print(f"\nKelly Criterion Results:")
            print(f"Kelly Fraction: {k['kelly_fraction']:.2f}x")
            print(f"Recommended Leverage: {k['safe_kelly']:.2f}x")
            print(f"Leveraged Expected Return: {k['leveraged_return']*100:.2f}%")
            print(f"Leveraged Volatility: {k['leveraged_volatility']*100:.2f}%")
            print(f"Leveraged Sharpe Ratio: {k['leveraged_sharpe']:.2f}")
            
            if k['kelly_fraction'] > 2:
                print("\nWarning: Kelly leverage > 2x. Consider using lower leverage for safety.")
        else:
            print("❌ Portfolio optimization did not return valid results.")
    except Exception as e:
        print(f"\n❌ Error during portfolio analysis: {str(e)}")
    
    # Add performance summary at the end
    total_time = time.time() - total_start_time
    print(f"\nPerformance Summary:")
    print(f"- Total execution time: {total_time:.2f} seconds")
    print(f"- Data fetching: {fetch_time:.2f} seconds")
    if 'returns_time' in locals():
        print(f"- Returns calculation: {returns_time:.2f} seconds")
    
    if hasattr(optimizer, 'performance_stats'):
        for key, value in optimizer.performance_stats.items():
            print(f"- {key}: {value:.2f} seconds")

    print(f"\n✅ Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()