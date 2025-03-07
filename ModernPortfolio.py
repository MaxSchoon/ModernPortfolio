import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import os
import time
import argparse
import sys
import random
import pickle
import json
import requests.exceptions
from urllib3.exceptions import HTTPError
import ssl
import matplotlib.pyplot as plt

# Import the CSV cache manager
from csv_cache_manager import CSVDataCache

# Only import utils modules if they exist - add a fallback
try:
    from utils import load_tickers, format_ticker
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
            
            df = pd.read_csv(csv_path, sep=delimiter)
            
            # Check column names (case insensitive)
            ticker_column = None
            for col in df.columns:
                if col.lower() == 'ticker':
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
        # Handle special case for Danaos (DAC)
        if ticker.lower() == 'danaos':
            return 'DAC'
            
        # Handle Flow Traders which should be FLOW.AS
        if ticker.lower() == 'flow':
            return 'FLOW.AS'
            
        # Handle HAL Trust which should be HAL.AS
        if ticker.lower() == 'hal':
            return 'HAL.AS'
            
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
    from batch_fetcher import BatchFetcher
except ImportError:
    print("Warning: BatchFetcher module not found. Using simplified version.")
    
    class BatchFetcher:
        """Simplified batch fetcher that falls back to individual fetching"""
        def __init__(self, years=5, batch_size=3, delay_min=2.0, delay_max=5.0, retry_count=3):
            self.years = years
            print("Using simplified batch fetcher (missing module)")
            
        def fetch_all(self, tickers, use_cache=True):
            print("Simplified batch fetcher does not implement actual fetching")
            return {ticker: "Not implemented" for ticker in tickers}

class PortfolioOptimizer:
    def __init__(self, tickers: List[str], risk_free_rate: float = 0.04, margin_cost_rate: float = 0.065, 
                years: int = 5, output_dir: str = "portfolio_analysis"):
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.margin_cost_rate = margin_cost_rate
        self.years = years
        self.price_data = None
        self.div_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.cache = CSVDataCache()  # Use CSV cache instead of old cache
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

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
            if nan_count > 0:
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
            
            if data.empty:
                print("❌ Error: Connected to Yahoo Finance but received empty data")
                return False
                
            print(f"✅ Successfully connected to Yahoo Finance API")
            print(f"   Sample data for {test_ticker}: {data['Close'].iloc[-1]:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Yahoo Finance: {str(e)}")
            return False

    def fetch_data(self, years: Optional[int] = None, use_cache: bool = True, use_batch_fetcher: bool = False) -> None:
        """
        Fetch price and dividend data for the specified tickers
        
        Parameters:
            years: Number of years of data to fetch
            use_cache: Whether to use cached data
            use_batch_fetcher: Whether to use batch fetcher (recommended for multiple tickers)
        """
        if years is None:
            years = self.years
            
        # First test connectivity
        if not self.test_yahoo_connectivity():
            print("⚠️ Warning: Yahoo Finance connectivity test failed.")
            print("Proceeding anyway, but expect potential issues with data fetching.")
            
        # Make sure end date is current date, not future date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        print(f"\nFetching data from {start_date.date()} to {end_date.date()}")
        
        self.price_data = pd.DataFrame()
        self.div_data = pd.DataFrame()

        print("\nFetching Data Status:")
        print("-" * 50)
        
        # Check cache status if using cache
        if use_cache:
            cache_status = self.cache.get_cache_status()
            print("\nCache Status:")
            print(f"- Price data: {cache_status.get('price_ticker_count', 0)} tickers ({cache_status.get('price_cache_size_mb', 0):.2f} MB)")
            print(f"- Date range: {cache_status.get('oldest_data', 'N/A')} to {cache_status.get('newest_data', 'N/A')}")
        else:
            print("\nCache disabled. All data will be fetched from Yahoo Finance.")
        
        # Handle synthetic assets (CASH/TBILLS)
        for ticker in self.tickers:
            if ticker in ['CASH', 'TBILLS']:
                self._create_synthetic_asset(ticker, start_date, end_date)
                continue
                
            # For real tickers, try to load from cache or fetch them
            if use_cache:
                cached_data = self.cache.get_price_data(ticker, max_age_days=30)
                if cached_data is not None:
                    if len(cached_data) > 100:  # Ensure we have enough data
                        # Check for NaN values
                        nan_pct = cached_data.isna().mean() * 100
                        if nan_pct > 50:
                            print(f"⚠️ {ticker}: Cached data has too many NaN values ({nan_pct:.1f}%), fetching fresh data")
                        else:
                            self.price_data[ticker] = cached_data
                            div_data = self.cache.get_div_data(ticker, max_age_days=30)
                            if div_data is not None:
                                self.div_data[ticker] = div_data
                            else:
                                # Create empty dividend data
                                self.div_data[ticker] = pd.Series(0.0, index=cached_data.index)
                                
                            print(f"✅ {ticker}: Using {len(cached_data)} days of cached data")
                            
                            # Plot price data
                            self._plot_price_data(ticker, cached_data)
                            continue
            
            # If we get here, need to fetch new data
            formatted_ticker = format_ticker(ticker)
            if formatted_ticker != ticker:
                print(f"🔄 {ticker} formatted to {formatted_ticker}")
                
            try:
                print(f"🔍 Fetching data for {formatted_ticker}...")
                stock = yf.Ticker(formatted_ticker)
                
                # Fetch prices for date range
                prices = stock.history(start=start_date, end=end_date)['Close']
                
                # Check data quality
                if len(prices) < 100:
                    print(f"❌ {ticker}: Insufficient data ({len(prices)} days)")
                    continue
                    
                # Clean up NaN values if needed
                nan_count = prices.isna().sum()
                if (nan_count > 0):
                    print(f"⚠️ {ticker}: Found {nan_count} NaN values, filling them")
                    prices = prices.ffill().bfill()
                    
                # Format dates and save price data
                prices.index = prices.index.tz_localize(None)
                self.price_data[ticker] = prices
                
                # Get dividends
                try:
                    dividends = stock.dividends
                    if not dividends.empty:
                        dividends.index = dividends.index.tz_localize(None)
                        div_series = pd.Series(0.0, index=prices.index)
                        common_dates = dividends.index.intersection(div_series.index)
                        
                        if not common_dates.empty:
                            div_series.loc[common_dates] = dividends[dividends.index.isin(common_dates)]
                        
                        self.div_data[ticker] = div_series
                        print(f"   Found {len(dividends)} dividend payments")
                    else:
                        self.div_data[ticker] = pd.Series(0.0, index=prices.index)
                        print(f"   No dividend data available")
                except Exception as e:
                    print(f"⚠️ Error fetching dividends for {ticker}: {str(e)}")
                    self.div_data[ticker] = pd.Series(0.0, index=prices.index)
                    
                # Save to cache if enabled
                if use_cache:
                    self.cache.save_price_data(ticker, prices)
                    self.cache.save_div_data(ticker, self.div_data[ticker])
                    
                print(f"✅ {ticker}: Successfully fetched {len(prices)} days of price data")
                
                # Plot price data for newly fetched ticker
                if ticker in self.price_data:
                    self._plot_price_data(ticker, self.price_data[ticker])
                
            except Exception as e:
                print(f"❌ Error fetching {ticker}: {str(e)}")
                continue
                
        # Remove tickers that failed to load
        failed_tickers = [ticker for ticker in self.tickers if ticker not in self.price_data.columns and ticker not in ['CASH', 'TBILLS']]
        if failed_tickers:
            print(f"\n❌ The following tickers failed to load: {', '.join(failed_tickers)}")
            self.tickers = [t for t in self.tickers if t in self.price_data.columns or t in ['CASH', 'TBILLS']]
            
        # Add synthetic assets if needed
        for ticker in self.tickers:
            if ticker in ['CASH', 'TBILLS'] and ticker not in self.price_data.columns:
                self._create_synthetic_asset(ticker, start_date, end_date)
                
        print(f"\n✅ Successfully loaded data for {len(self.tickers)} tickers")

    def _create_synthetic_asset(self, ticker: str, start_date, end_date):
        """Create synthetic data for cash-like assets"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Set return based on asset type
        if ticker == 'CASH':
            rate = self.risk_free_rate
            variance = 1e-8
        else:  # TBILLS
            rate = self.risk_free_rate * 1.02  # Slightly higher than cash
            variance = 1e-6  # Slightly more volatile than cash
        
        daily_return = (1 + rate) ** (1/252) - 1
        prices = (1 + daily_return) ** np.arange(len(date_range))
        self.price_data[ticker] = pd.Series(prices * (1 + np.random.normal(0, np.sqrt(variance), len(date_range))), 
                                          index=date_range)
        self.div_data[ticker] = pd.Series(0.0, index=date_range, dtype='float64')
        print(f"💰 {ticker}: Synthetic data generated successfully")

    def _plot_price_data(self, ticker: str, price_data: pd.Series) -> None:
        """Plot price data and save to file"""
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
            
            # Save plot
            plot_file = os.path.join(self.output_dir, f"{ticker}_price.png")
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
        price_df = aligned_data['prices']
        div_df = aligned_data['dividends']
        
        # Calculate returns
        price_returns = price_df.pct_change().fillna(0)
        div_returns = (div_df / price_df.shift(1)).fillna(0)
        total_returns = price_returns.add(div_returns)
    
        # Set cash returns explicitly
        if 'CASH' in self.tickers:
            daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            total_returns['CASH'] = daily_rf
            
        if 'TBILLS' in self.tickers:
            daily_rf = (1 + self.risk_free_rate * 1.02) ** (1/252) - 1
            total_returns['TBILLS'] = daily_rf
    
        # Calculate annualized returns
        self.mean_returns = total_returns.mean() * 252
    
        # Calculate covariance matrix
        self.cov_matrix = total_returns.cov() * 252
    
        # Ensure covariance matrix is well-behaved for cash-like assets
        for ticker in ['CASH', 'TBILLS']:
            if ticker in self.tickers:
                idx = self.tickers.index(ticker)
                variance = 1e-8 if ticker == 'CASH' else 1e-6
                self.cov_matrix.iloc[idx, :] = 0
                self.cov_matrix.iloc[:, idx] = 0
                self.cov_matrix.iloc[idx, idx] = variance
        
        # Print return metrics for debugging
        print("\nAnnualized Return Metrics:")
        print("-" * 50)
        
        # Create a returns summary dataframe
        returns_summary = pd.DataFrame(index=self.tickers)
        returns_summary['AnnReturn'] = self.mean_returns * 100
        returns_summary['AnnVolatility'] = np.sqrt(np.diag(self.cov_matrix)) * 100
        returns_summary['Sharpe'] = (self.mean_returns - self.risk_free_rate) / np.sqrt(np.diag(self.cov_matrix))
        
        # Save returns summary
        summary_file = os.path.join(self.output_dir, "returns_summary.csv")
        returns_summary.to_csv(summary_file)
        print(f"Returns summary saved to {summary_file}")
        
        # Save correlation matrix
        if total_returns is not None:  # Add this check
            corr_matrix = total_returns.corr()
            corr_file = os.path.join(self.output_dir, "correlations.csv")
            corr_matrix.to_csv(corr_file)
            print(f"Correlation matrix saved to {corr_file}")
        
        # Plot returns comparison
        self._plot_returns_comparison(returns_summary)
        
        for ticker in self.tickers:
            print(f"{ticker}: Return = {self.mean_returns[ticker]*100:.2f}%, " +
                 f"Volatility = {np.sqrt(self.cov_matrix.loc[ticker, ticker])*100:.2f}%")

    def _align_data(self):
        """Align price and dividend data to ensure all tickers have data for the same dates"""
        print("Aligning data across tickers...")
        
        # Extract price and dividend DataFrames
        price_df = self.price_data[self.tickers].copy()
        div_df = self.div_data[self.tickers].copy() if all(t in self.div_data for t in self.tickers) else None
        
        # If dividend data is missing for some tickers, create it with zeros
        if div_df is None:
            div_df = pd.DataFrame(0.0, index=price_df.index, columns=self.tickers)
            for ticker in self.tickers:
                if ticker in self.div_data:
                    common_idx = price_df.index.intersection(self.div_data[ticker].index)
                    div_df.loc[common_idx, ticker] = self.div_data[ticker].loc[common_idx]
        
        # Fill missing prices with forward/backward fill
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill missing dividends with zeros
        div_df = div_df.fillna(0)
        
        # Check if we have valid data for all tickers
        for ticker in self.tickers:
            nan_pct = price_df[ticker].isna().mean() * 100
            if nan_pct > 0:
                print(f"⚠️ {ticker}: {nan_pct:.1f}% missing data after alignment")
                
                if nan_pct > 50:
                    print(f"❌ {ticker}: Too much missing data, removing from analysis")
                    price_df = price_df.drop(ticker, axis=1)
                    div_df = div_df.drop(ticker, axis=1)
        
        # Remove tickers with invalid data
        valid_tickers = price_df.columns.tolist()
        if len(valid_tickers) < len(self.tickers):
            dropped = [t for t in self.tickers if t not in valid_tickers]
            print(f"❌ Removed {len(dropped)} tickers with insufficient data: {', '.join(dropped)}")
            self.tickers = valid_tickers
            
        if not self.tickers:
            print("❌ No valid tickers left after alignment")
            return None
            
        print(f"✅ Successfully aligned data for {len(self.tickers)} tickers, covering {len(price_df)} dates")
            
        return {
            'prices': price_df[self.tickers],
            'dividends': div_df[self.tickers]
        }

    def _plot_returns_comparison(self, returns_summary: pd.DataFrame) -> None:
        """Plot comparison of returns and volatility"""
        try:
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
            plt.title('Risk-Return Profile of Assets')
            plt.grid(True)
            
            # Add risk-free rate as horizontal line
            plt.axhline(y=self.risk_free_rate*100, color='r', linestyle='--', 
                       label=f'Risk-Free Rate ({self.risk_free_rate*100:.1f}%)')
            
            plt.legend()
            
            # Save plot
            plot_file = os.path.join(self.output_dir, "risk_return_profile.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Risk-return profile saved to {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating returns comparison plot: {str(e)}")

    def portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        ret = np.sum(self.mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        vol = max(vol, 1e-8)  # Prevent division by zero
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe

    def optimize_portfolio(self, exclude_cash: bool = False) -> Dict:
        """
        Optimize portfolio weights for maximum Sharpe ratio
        
        Parameters:
            exclude_cash: If True, exclude CASH and TBILLS from optimization
        """
        print("\nOptimizing portfolio allocation...")
        
        # Filter tickers if excluding cash
        if exclude_cash:
            opt_tickers = [t for t in self.tickers if t not in ['CASH', 'TBILLS']]
            if not opt_tickers:
                print("❌ No equity tickers available for optimization after excluding cash.")
                return None
                
            print(f"Optimizing {len(opt_tickers)} tickers (excluding cash assets)")
                
            # Extract relevant returns and covariance
            opt_returns = self.mean_returns[opt_tickers]
            opt_cov = self.cov_matrix.loc[opt_tickers, opt_tickers]
        else:
            opt_tickers = self.tickers
            opt_returns = self.mean_returns
            opt_cov = self.cov_matrix
            print(f"Optimizing all {len(opt_tickers)} tickers")
        
        n_assets = len(opt_tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)

        def negative_sharpe(weights):
            ret = np.sum(opt_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(opt_cov, weights)))
            vol = max(vol, 1e-8)  # Prevent division by zero
            sharpe = (ret - self.risk_free_rate) / vol
            return -sharpe
            
        try:
            result = minimize(negative_sharpe, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
                            
            if not result['success']:
                print(f"❌ Optimization failed: {result['message']}")
                return None
        except Exception as e:
            print(f"❌ Error during optimization: {str(e)}")
            return None
        
        # If we excluded cash, reincorporate cash assets with 0% allocation
        if exclude_cash:
            full_weights = {}
            result_weights = dict(zip(opt_tickers, result.x))
            
            for ticker in self.tickers:
                if ticker in result_weights:
                    full_weights[ticker] = result_weights[ticker]
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
            # Use normal results
            results = self.get_optimization_results(result.x)
        
        # Save optimization results
        if results:
            # Save weights as JSON
            weights_file = os.path.join(self.output_dir, "optimal_weights.json")
            with open(weights_file, 'w') as f:
                weights_pct = {ticker: weight*100 for ticker, weight in results['weights'].items()}
                json.dump(weights_pct, f, indent=2)
            print(f"Optimal weights saved to {weights_file}")
            
            # Plot portfolio allocation
            self._plot_portfolio_weights(results['weights'])
            
            # Plot efficient frontier
            self._plot_efficient_frontier(results)
            
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
        """Plot portfolio weights as a pie chart"""
        try:
            # Convert weights to percentages for display
            weights_pct = {k: v*100 for k, v in weights.items()}
            
            # Filter weights for readability (only include weights > 1%)
            significant_weights = {k: v for k, v in weights_pct.items() if v > 1}
            other_weight = 100 - sum(significant_weights.values())
            
            if other_weight > 0:
                significant_weights['Other'] = other_weight
            
            plt.figure(figsize=(10, 8))
            plt.pie(significant_weights.values(), labels=significant_weights.keys(), 
                   autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Optimal Portfolio Allocation')
            
            # Save plot
            plot_file = os.path.join(self.output_dir, "portfolio_allocation.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Portfolio allocation chart saved to {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating portfolio weights plot: {str(e)}")

    def _plot_efficient_frontier(self, optimal_result: Dict) -> None:
        """Plot the efficient frontier with the optimal portfolio"""
        try:
            # Generate efficient frontier points
            target_returns = np.linspace(min(self.mean_returns), max(self.mean_returns), 100)
            efficient_volatilities = []
            
            # Calculate volatility for each target return
            for target in target_returns:
                # Minimize volatility subject to target return
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                
                # Constraint (sum of weights = 1 and return = target)
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target}
                ]
                
                n_assets = len(self.mean_returns)
                bounds = tuple((0, 1) for _ in range(n_assets))
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
            plt.scatter(vols, self.mean_returns * 100, s=50, alpha=0.7, label='Individual Assets')
            
            # Add labels for assets
            for i, ticker in enumerate(self.tickers):
                plt.annotate(ticker, xy=(vols[i], self.mean_returns[i] * 100), xytext=(5, 5), 
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

def main():
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Modern Portfolio Optimizer')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--convert-cache', action='store_true', help='Convert old cache to CSV format')  # Fixed typo here
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
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        cache = CSVDataCache()
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

    # Initialize and run optimization
    optimizer = PortfolioOptimizer(
        tickers, 
        risk_free_rate=args.risk_free,
        margin_cost_rate=args.margin_cost,
        years=args.years,
        output_dir=args.output_dir
    )
    
    print("🔄 Fetching data...")
    optimizer.fetch_data(
        years=args.years,
        use_cache=not args.no_cache
    )
    
    try:
        print("🧮 Calculating returns...")
        optimizer.calculate_returns()
        
        print("🎯 Optimizing portfolio...")
        results = optimizer.optimize_portfolio(exclude_cash=args.exclude_cash)
        
        if not results:
            print("❌ Portfolio optimization failed.")
            return
        
        # Print results
        print("\nOptimal Portfolio Allocation:")
        for ticker, weight in sorted(results['weights'].items(), key=lambda x: -x[1]):
            if weight > 0.001:  # Only show positions with at least 0.1% allocation
                print(f"{ticker}: {weight*100:.2f}%")
        
        print(f"\nUnleveraged Portfolio Metrics:")
        print(f"Expected Return: {results['return']*100:.2f}%")
        print(f"Volatility: {results['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe']:.2f}")
        
        k = results['kelly_metrics']
        print(f"\nKelly Criterion Results:")
        print(f"Kelly Fraction: {k['kelly_fraction']:.2f}x")
        print(f"Recommended Leverage: {k['safe_kelly']:.2f}x")
        print(f"Leveraged Expected Return: {k['leveraged_return']*100:.2f}%")
        print(f"Leveraged Volatility: {k['leveraged_volatility']*100:.2f}%")
        print(f"Leveraged Sharpe Ratio: {k['leveraged_sharpe']:.2f}")
        
        if k['kelly_fraction'] > 2:
            print("\nWarning: Kelly leverage > 2x. Consider using lower leverage for safety.")
            
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        
    print(f"\n✅ Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()


