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
import time

from src.cache.csv_cache_manager import CSVDataCache
try:
    from src.cache.cache_standardize import standardize_cache
    HAS_STANDARDIZER = True
except ImportError:
    print("Warning: cache_standardize module not found. Cache standardization will be skipped.")
    HAS_STANDARDIZER = False

try:
    from src.utils.utils import load_tickers, format_ticker
except ImportError:
    def load_tickers(csv_path: str) -> List[str]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tickers file not found: {csv_path}")
        try:
            print(f"Reading ticker data from: {csv_path}")
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    delimiter = ';'
                    print("Detected semicolon delimiter")
                elif ',' in first_line:
                    delimiter = ','
                    print("Detected comma delimiter")
                else:
                    delimiter = ';'
                    print("Using default semicolon delimiter")
            df = pd.read_csv(csv_path, sep=delimiter, comment='/')
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
            sample = tickers[:min(5, len(tickers))]
            print(f"Sample tickers: {', '.join(sample)}")
            return tickers
        except Exception as e:
            print(f"Error reading tickers file: {str(e)}")
            raise
    def format_ticker(ticker: str) -> str:
        if ticker.lower() == 'danaos':
            return 'DAC'
        if ticker.lower() == 'flow':
            return 'FLOW.AS'
        if ticker.lower() == 'hal':
            return 'HAL.AS'
        if '.' not in ticker and '-' not in ticker:
            return ticker
        if '-' in ticker:
            return ticker
        if ticker.endswith('.AS'):
            return ticker
        elif ticker.endswith('.L'):
            return ticker
        elif ticker.endswith('.PA'):
            return ticker
        elif ticker.endswith('.DE'):
            return ticker
        elif ticker.endswith('.MI'):
            return ticker
        return ticker

try:
    from src.fetching.data_fetcher import DataFetcher
except ImportError:
    print("Warning: BatchFetcher module not found. Using simplified version.")
    class DataFetcher:
        def __init__(self, years=5, batch_size=3, delay_min=2.0, delay_max=5.0, retry_count=3):
            self.years = years
            print("Using simplified batch fetcher (missing module)")
        def fetch_all(self, tickers, use_cache=True):
            print("Simplified batch fetcher does not implement actual fetching")
            return {ticker: "Not implemented" for ticker in tickers}

class PortfolioOptimizer:
    def __init__(self, tickers: List[str], risk_free_rate: float = 0.04, margin_cost_rate: float = 0.065, 
                years: int = 5, output_dir: str = "portfolio_analysis", cache_dir: str = "data_cache",
                debug: bool = False):
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.margin_cost_rate = margin_cost_rate
        self.years = years
        self.price_data = None
        self.div_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.cache = CSVDataCache(cache_dir)
        self.output_dir = output_dir
        self.charts_dir = os.path.join(output_dir, "price_charts")
        self.cache_dir = cache_dir
        self.debug = debug
        for directory in [output_dir, self.charts_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        self.performance_stats = {}

    def validate_data(self, ticker: str, prices: pd.Series, dividends: Optional[pd.Series] = None) -> bool:
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
        if ticker in ['CASH', 'TBILLS'] or ticker not in self.div_data:
            return 0.0, 0.0
        prices = self.price_data[ticker]
        dividends = self.div_data[ticker]
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
        if not annual_yield:
            return 0.0, 0.0
        weights = np.exp(-np.arange(len(annual_yield)) * 0.5)
        weights = weights / weights.sum()
        weighted_yield = np.sum(weights * annual_yield)
        total_div = np.sum(annual_div)
        return weighted_yield, total_div

    def print_data_summary(self):
        print("\nData Summary:")
        print("-" * 70)
        for ticker in self.tickers:
            prices = self.price_data[ticker]
            nan_count = prices.isna().sum()
            if (nan_count > 0):
                print(f"\n{ticker} - WARNING: Contains {nan_count} NaN values!")
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
                dividends = self.div_data[ticker]
                try:
                    annual_divs = [dividends.iloc[-(i+1)*252:-i*252 if i > 0 else None].sum() 
                                 for i in range(min(5, len(dividends)//252))]
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
        print("Testing Yahoo Finance connectivity...")
        try:
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
        if years is None:
            years = self.years
        print(f"\nFetching data for {len(self.tickers)} tickers over {years} years...")
        fetcher = DataFetcher(
            cache_dir=self.cache_dir,
            batch_size=batch_size,
            years=years,
            max_workers=max_workers
        )
        results = fetcher.fetch_all(self.tickers, use_cache)
        self.price_data = pd.DataFrame()
        self.div_data = pd.DataFrame()
        price_data_dict = {}
        div_data_dict = {}
        for ticker in self.tickers:
            if ticker in results and results[ticker].startswith("✅"):
                cached_price = self.cache.get_price_data(ticker)
                cached_div = self.cache.get_div_data(ticker)
                if cached_price is not None:
                    price_data_dict[ticker] = cached_price
                    self._plot_price_data(ticker, cached_price)
                if cached_div is not None:
                    div_data_dict[ticker] = cached_div
            else:
                print(f"❌ {ticker}: Failed to fetch data")
        if price_data_dict:
            self.price_data = pd.concat(price_data_dict, axis=1)
        if div_data_dict:
            self.div_data = pd.concat(div_data_dict, axis=1)
        failed_tickers = [ticker for ticker in self.tickers if ticker not in self.price_data.columns]
        if failed_tickers:
            print(f"\n❌ The following tickers failed to load: {', '.join(failed_tickers)}")
            self.tickers = [t for t in self.tickers if t in self.price_data.columns]
        print(f"\n✅ Successfully loaded data for {len(self.tickers)} tickers")

    def _create_synthetic_asset(self, ticker: str, common_index=None):
        if (common_index is None):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*self.years)
            common_index = pd.date_range(start=start_date, end=end_date, freq='B')
            print(f"⚠️ Creating generic date range for {ticker} with {len(common_index)} days")
        else:
            print(f"🔄 Creating {ticker} with {len(common_index)} days aligned to other tickers")
        if ticker == 'CASH':
            rate = self.risk_free_rate
            variance = 1e-10
        else:
            rate = self.risk_free_rate * 1.02
            variance = 1e-6
        daily_return = (1 + rate) ** (1/252) - 1
        days = len(common_index)
        prices = np.ones(days)
        for i in range(1, days):
            daily_noise = np.random.normal(0, np.sqrt(variance))
            prices[i] = prices[i-1] * (1 + daily_return + daily_noise)
        price_series = pd.Series(prices, index=common_index)
        self.price_data[ticker] = price_series
        self.div_data[ticker] = pd.Series(0.0, index=common_index)
        print(f"💰 {ticker}: Synthetic data generated successfully ({len(price_series)} days)")

    def _plot_price_data(self, ticker: str, price_data: pd.Series) -> None:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(price_data.index, price_data.values)
            plt.title(f"{ticker} Price History")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.grid(True)
            start_price = price_data.iloc[0]
            end_price = price_data.iloc[-1]
            plt.annotate(f"${start_price:.2f}", xy=(price_data.index[0], start_price),
                        xytext=(10, 10), textcoords="offset points")
            plt.annotate(f"${end_price:.2f}", xy=(price_data.index[-1], end_price),
                        xytext=(-40, 10), textcoords="offset points")
            plot_file = os.path.join(self.charts_dir, f"{ticker}_price.png")
            plt.savefig(plot_file)
            plt.close()
        except Exception as e:
            print(f"⚠️ Error creating plot for {ticker}: {str(e)}")

    def calculate_returns(self) -> None:
        print("\nCalculating returns...")
        if self.price_data is None or self.price_data.empty:
            raise ValueError("No price data available. Cannot calculate returns.")
        empty_tickers = [ticker for ticker in self.tickers 
                         if ticker not in self.price_data.columns]
        if empty_tickers:
            print(f"⚠️ The following tickers have no price data: {', '.join(empty_tickers)}")
            self.tickers = [t for t in self.tickers if t in self.price_data.columns]
        if not self.tickers:
            raise ValueError("No tickers with price data. Cannot calculate returns.")
        aligned_data = self._align_data()
        if aligned_data is None:
            raise ValueError("Failed to align price data across tickers.")
        prices_df = aligned_data['prices']
        div_df = aligned_data['dividends']
        synthetic_tickers = [t for t in self.tickers if t in ['CASH', 'TBILLS']]
        price_returns = prices_df.pct_change().fillna(0)
        div_returns = (div_df / prices_df.shift(1)).fillna(0)
        total_returns = price_returns.add(div_returns)
        for ticker in synthetic_tickers:
            if ticker in total_returns.columns:
                if ticker == 'CASH':
                    daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
                    total_returns[ticker] = daily_rf
                elif ticker == 'TBILLS':
                    daily_rf = (1 + self.risk_free_rate * 1.02) ** (1/252) - 1
                    total_returns[ticker] = daily_rf
        self.mean_returns = total_returns.mean() * 252
        self.cov_matrix = total_returns.cov() * 252
        for ticker in synthetic_tickers:
            if ticker in self.tickers:
                idx = list(self.tickers).index(ticker)
                if ticker == 'CASH':
                    variance = 1e-8
                else:
                    variance = 1e-5
                for i in range(len(self.tickers)):
                    self.cov_matrix.iloc[idx, i] = 0
                    self.cov_matrix.iloc[i, idx] = 0
                self.cov_matrix.iloc[idx, idx] = variance
        print("\nAnnualized Return Metrics:")
        print("-" * 50)
        returns_summary = pd.DataFrame(index=self.tickers)
        returns_summary['AnnReturn'] = self.mean_returns * 100
        returns_summary['AnnVolatility'] = np.sqrt(np.diag(self.cov_matrix)) * 100
        returns_summary['Sharpe'] = (self.mean_returns - self.risk_free_rate) / np.sqrt(np.diag(self.cov_matrix))
        self.returns_summary = returns_summary
        summary_file = os.path.join(self.output_dir, "returns_summary.csv")
        returns_summary.to_csv(summary_file)
        print(f"Returns summary saved to {summary_file}")
        if total_returns is not None:
            corr_matrix = total_returns.corr()
            corr_file = os.path.join(self.output_dir, "correlations.csv")
            corr_matrix.to_csv(corr_file)
            print(f"Correlation matrix saved to {corr_file}")
        self._plot_returns_comparison(returns_summary)
        if self.debug:
            for ticker in self.tickers:
                print(f"{ticker}: Return = {self.mean_returns[ticker]*100:.2f}%, " +
                     f"Volatility = {np.sqrt(self.cov_matrix.loc[ticker, ticker])*100:.2f}%")

    def _align_data(self):
        start_time = time.time()
        print("Aligning data across tickers...")
        synthetic_tickers = [t for t in self.tickers if t in ['CASH', 'TBILLS']]
        real_tickers = [t for t in self.tickers if t not in ['CASH', 'TBILLS']]
        if not real_tickers:
            print("⚠️ No real tickers found, only synthetic assets")
            price_df = self.price_data[self.tickers].copy()
            div_df = self.div_data[self.tickers].copy()
            return {
                'prices': price_df,
                'dividends': div_df
            }
        price_df = self.price_data[real_tickers].copy()
        current_date = pd.Timestamp.now().normalize()
        print(f"Full date range before filtering: {price_df.index.min()} to {price_df.index.max()}")
        if price_df.index.max() > current_date:
            price_df = price_df.loc[price_df.index <= current_date]
            print(f"Date range after filtering future dates: {price_df.index.min()} to {price_df.index.max()}")
        price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d')
        missing_pcts = price_df.isna().mean() * 100
        if self.debug:
            print("\nMissing data report BEFORE filtering:")
            for ticker, pct in missing_pcts.items():
                print(f"{ticker}: {price_df[ticker].isna().sum()} missing values ({pct:.1f}%)")
        else:
            print("\nAnalyzing data quality (use --debug for details)...")
            print(f"Average missing data: {missing_pcts.mean():.1f}%")
        high_missing_threshold = 45.0
        all_high_missing = (missing_pcts > high_missing_threshold).all()
        if all_high_missing:
            print("\n⚠️ All tickers have high missing data percentages. This suggests a systematic issue.")
            valid_data = ~price_df.isna()
            if valid_data.any().any():
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
                        price_df = price_df.loc[latest_start:earliest_end]
                        missing_pcts = price_df.isna().mean() * 100
                        print("\nMissing data after date range restriction:")
                        for ticker, pct in missing_pcts.items():
                            print(f"{ticker}: {price_df[ticker].isna().sum()} missing values ({pct:.1f}%)")
        all_high_missing = (missing_pcts > high_missing_threshold).all()
        if all_high_missing:
            sorted_tickers = missing_pcts.sort_values().index.tolist()
            keep_count = max(10, int(len(sorted_tickers) * 0.2))
            keep_count = min(keep_count, len(sorted_tickers))
            keep_tickers = sorted_tickers[:keep_count]
            print(f"Keeping top {keep_count} tickers with least missing data:")
            for ticker in keep_tickers:
                print(f"- {ticker}: {missing_pcts[ticker]:.1f}% missing")
            price_df = price_df[keep_tickers]
            real_tickers = keep_tickers
        else:
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
        div_df = pd.DataFrame(0.0, index=price_df.index, columns=real_tickers)
        for ticker in real_tickers:
            if ticker in self.div_data:
                common_idx = price_df.index.intersection(self.div_data[ticker].index)
                if not common_idx.empty:
                    div_df.loc[common_idx, ticker] = self.div_data[ticker].loc[common_idx]
        price_df = price_df.ffill().bfill()
        for ticker in synthetic_tickers:
            if ticker in self.price_data.columns:
                self._create_synthetic_asset(ticker, price_df.index)
                price_df[ticker] = self.price_data[ticker]
                div_df[ticker] = self.div_data[ticker]
                print(f"✅ {ticker}: Successfully added synthetic data to aligned dataset")
        self.tickers = price_df.columns.tolist()
        nan_counts = price_df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️ WARNING: Still have {nan_counts.sum()} NaN values after alignment!")
            for ticker, count in nan_counts[nan_counts > 0].items():
                print(f"  {ticker}: {count} NaN values ({count/len(price_df)*100:.1f}%)")
        else:
            print(f"✅ No NaN values in aligned data - good for optimization")
        print(f"✅ Successfully aligned data for {len(self.tickers)} tickers, covering {len(price_df)} dates")
        print(f"   Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
        self.performance_stats['align_data_time'] = time.time() - start_time
        return {
            'prices': price_df,
            'dividends': div_df
        }

    def portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        ret = np.sum(self.mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        vol = max(vol, 1e-8)
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe

    def optimize_portfolio(self, exclude_cash: bool = False, skip_plots: bool = False) -> Dict:
        start_time = time.time()
        print("\nOptimizing portfolio allocation...")
        if self.mean_returns is None or self.cov_matrix is None:
            print("❌ No return data available for optimization.")
            return None
        if len(self.tickers) != len(self.mean_returns) or len(self.tickers) != self.cov_matrix.shape[0]:
            print(f"❌ Dimension mismatch: tickers={len(self.tickers)}, returns={len(self.mean_returns)}, cov={self.cov_matrix.shape}")
            return None
        if np.isnan(self.mean_returns).any():
            nan_tickers = [t for t, val in zip(self.tickers, np.isnan(self.mean_returns)) if val]
            print(f"❌ The following tickers have NaN returns: {', '.join(nan_tickers)}")
            good_indices = ~np.isnan(self.mean_returns)
            self.tickers = [t for i, t in enumerate(self.tickers) if good_indices[i]]
            self.mean_returns = self.mean_returns[good_indices]
            self.cov_matrix = self.cov_matrix.loc[self.tickers, self.tickers]
        if np.isnan(self.cov_matrix).any().any():
            print("❌ The covariance matrix contains NaN values. Attempting to fix...")
            try:
                vols = np.sqrt(np.diagonal(self.cov_matrix))
                corr_matrix = self.cov_matrix.copy()
                for i in range(len(self.tickers)):
                    for j in range(len(self.tickers)):
                        if vols[i] > 0 and vols[j] > 0:
                            corr_matrix.iloc[i, j] = self.cov_matrix.iloc[i, j] / (vols[i] * vols[j])
                corr_matrix = corr_matrix.fillna(0)
                for i in range(len(self.tickers)):
                    for j in range(len(self.tickers)):
                        self.cov_matrix.iloc[i, j] = corr_matrix.iloc[i, j] * vols[i] * vols[j]
                print("✅ Successfully repaired covariance matrix")
            except Exception as e:
                print(f"❌ Error repairing covariance matrix: {str(e)}")
                problem_rows = np.isnan(self.cov_matrix).any(axis=1)
                nan_tickers = [t for i, t in enumerate(self.tickers) if problem_rows[i]]
                print(f"Removing problematic tickers: {', '.join(nan_tickers)}")
                good_tickers = [t for t in self.tickers if t not in nan_tickers]
                if not good_tickers:
                    print("❌ No valid tickers left after removing problematic ones.")
                    return None
                self.tickers = good_tickers
                self.mean_returns = self.mean_returns[self.tickers]
                self.cov_matrix = self.cov_matrix.loc[self.tickers, self.tickers]
        extreme_return_threshold = 5.0
        extreme_vols = np.sqrt(np.diag(self.cov_matrix))
        extreme_vol_threshold = 2.0
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
        if exclude_cash:
            opt_tickers = [t for t in self.tickers if t not in ['CASH', 'TBILLS']]
            if not opt_tickers:
                print("❌ No equity tickers available for optimization after excluding cash.")
                return None
            print(f"Optimizing {len(opt_tickers)} tickers (excluding cash assets)")
            opt_returns = self.mean_returns[opt_tickers].values
            opt_cov = self.cov_matrix.loc[opt_tickers, opt_tickers].values
        else:
            opt_tickers = self.tickers
            opt_returns = self.mean_returns.values
            opt_cov = self.cov_matrix.values
            print(f"Optimizing {len(opt_tickers)} tickers")
        n_assets = len(opt_tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        def negative_sharpe(weights):
            ret = np.sum(opt_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(opt_cov, weights)))
            vol = max(vol, 1e-8)
            sharpe = (ret - self.risk_free_rate) / vol
            return -sharpe
        optimization_start = time.time()
        for attempt in range(3):
            try:
                if attempt > 0:
                    if isinstance(opt_returns, np.ndarray):
                        opt_returns_array = opt_returns
                        opt_cov_array = opt_cov
                    else:
                        opt_returns_array = opt_returns.values
                        opt_cov_array = opt_cov.values
                    sharpes = (opt_returns_array - self.risk_free_rate) / np.sqrt(np.diag(opt_cov_array))
                    sharpes = np.clip(sharpes, 0, None)
                    if sharpes.sum() > 0:
                        initial_weights = sharpes / sharpes.sum()
                    else:
                        initial_weights = np.random.random(n_assets)
                        initial_weights = initial_weights / initial_weights.sum()
                    print(f"Optimization attempt {attempt+1}: Using alternative initial weights")
                result = minimize(negative_sharpe, initial_weights, method='SLSQP',
                                bounds=bounds, constraints=constraints, 
                                options={'maxiter': 1000, 'ftol': 1e-9})
                if result['success']:
                    print(f"✅ Optimization successful after {attempt+1} attempt(s)")
                    break
                else:
                    print(f"⚠️ Attempt {attempt+1} failed: {result['message']}")
                    if attempt == 2:
                        print(f"❌ All optimization attempts failed")
                        return None
            except Exception as e:
                print(f"❌ Error during optimization attempt {attempt+1}: {str(e)}")
                if attempt == 2:
                    print("\nData diagnostics:")
                    print(f"Shape of returns vector: {np.shape(opt_returns)}")
                    print(f"Shape of covariance matrix: {np.shape(opt_cov)}")
                    print("First 5 returns:")
                    for i, ticker in enumerate(opt_tickers[:5]):
                        print(f"{ticker}: {opt_returns[i]:.4f}")
                    return None
        self.performance_stats['optimization_time'] = time.time() - optimization_start
        if exclude_cash:
            full_weights = {}
            result_weights = dict(zip(opt_tickers, result.x))
            for ticker in self.tickers:
                if ticker in result_weights:
                    full_weights[ticker] = result_weights[ticker]
                else:
                    full_weights[ticker] = 0.0
            weights_array = np.array([full_weights[t] for t in self.tickers])
            port_ret, port_vol, port_sharpe = self.portfolio_metrics(weights_array)
            results = {
                'weights': full_weights,
                'return': port_ret,
                'volatility': port_vol,
                'sharpe': port_sharpe,
                'kelly_metrics': self.calculate_kelly(port_ret, port_vol),
                'opt_tickers': opt_tickers
            }
        else:
            results = self.get_optimization_results(result.x)
            results['opt_tickers'] = opt_tickers
        if results:
            weights_file = os.path.join(self.output_dir, "optimal_weights.json")
            with open(weights_file, 'w') as f:
                weights_pct = {ticker: weight*100 for ticker, weight in results['weights'].items()}
                json.dump(weights_pct, f, indent=2)
            print(f"Optimal weights saved to {weights_file}")
            if not skip_plots:
                self._plot_portfolio_weights(results['weights'])
                self._plot_efficient_frontier(results, results['opt_tickers'])
                self._plot_returns_comparison(self.returns_summary, results['opt_tickers'])
        self.performance_stats['total_optimization_time'] = time.time() - start_time
        print(f"Portfolio optimization completed in {self.performance_stats['total_optimization_time']:.2f} seconds")
        return results

    def calculate_kelly(self, portfolio_ret: float, portfolio_vol: float) -> Dict:
        portfolio_vol = max(portfolio_vol, 1e-8)
        kelly_fraction = (portfolio_ret - self.margin_cost_rate) / portfolio_vol**2
        safe_kelly = min(kelly_fraction, 2.0)
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
        try:
            weights_pct = {k: v*100 for k, v in weights.items()}
            significant_weights = {k: v for k, v in weights_pct.items() if v > 1}
            other_weight = 100 - sum(significant_weights.values())
            if other_weight > 0:
                significant_weights['Other'] = other_weight
            plt.figure(figsize=(10, 8))
            plt.pie(significant_weights.values(), labels=significant_weights.keys(), 
                   autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Optimal Portfolio Allocation')
            plot_file = os.path.join(self.output_dir, "portfolio_allocation.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Portfolio allocation chart saved to {plot_file}")
        except Exception as e:
            print(f"⚠️ Error creating portfolio weights plot: {str(e)}")

    def _plot_efficient_frontier(self, optimal_result: Dict, tickers_to_plot: List[str] = None) -> None:
        start_time = time.time()
        try:
            if tickers_to_plot is None:
                tickers_to_plot = self.tickers
            mean_returns = self.mean_returns[tickers_to_plot]
            cov_matrix = self.cov_matrix.loc[tickers_to_plot, tickers_to_plot]
            num_points = 30
            target_returns = np.linspace(min(mean_returns), max(mean_returns), num_points)
            efficient_volatilities = []
            mean_returns_array = mean_returns.values
            cov_matrix_array = cov_matrix.values
            for target in target_returns:
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_array, weights)))
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(mean_returns_array * x) - target}
                ]
                n_assets = len(mean_returns)
                bounds = tuple((0, 1) for _ in range(n_assets))
                init_guess = np.array([1.0/n_assets] * n_assets)
                try:
                    result = minimize(portfolio_volatility, init_guess, 
                                     method='SLSQP', bounds=bounds, constraints=constraints)
                    efficient_volatilities.append(result['fun'])
                except:
                    efficient_volatilities.append(np.nan)
            valid = ~np.isnan(efficient_volatilities)
            efficient_volatilities = np.array(efficient_volatilities)[valid]
            efficient_returns = target_returns[valid]
            plt.figure(figsize=(12, 8))
            plt.plot(efficient_volatilities * 100, efficient_returns * 100, 'b-', label='Efficient Frontier')
            opt_return = optimal_result['return'] * 100
            opt_vol = optimal_result['volatility'] * 100
            plt.scatter(opt_vol, opt_return, s=100, color='r', marker='*', label='Optimal Portfolio')
            vols = np.sqrt(np.diag(cov_matrix)) * 100
            plt.scatter(vols, mean_returns.values * 100, s=50, alpha=0.7, label='Individual Assets')
            for i, ticker in enumerate(tickers_to_plot):
                plt.annotate(ticker, xy=(vols[i], mean_returns.iloc[i] * 100), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9)
            plt.xlabel('Annualized Volatility (%)')
            plt.ylabel('Annualized Return (%)')
            plt.title('Efficient Frontier and Optimal Portfolio')
            plt.grid(True)
            plt.legend()
            plot_file = os.path.join(self.output_dir, "efficient_frontier.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Efficient frontier plot saved to {plot_file}")
        except Exception as e:
            print(f"⚠️ Error creating efficient frontier plot: {str(e)}")
        self.performance_stats['efficient_frontier_time'] = time.time() - start_time

    def _plot_returns_comparison(self, returns_summary, tickers_to_plot: List[str] = None):
        try:
            plt.figure(figsize=(12, 8))
            if tickers_to_plot is not None:
                returns = returns_summary.loc[tickers_to_plot, 'AnnReturn']
                vols = returns_summary.loc[tickers_to_plot, 'AnnVolatility']
                sharpes = returns_summary.loc[tickers_to_plot, 'Sharpe']
            else:
                returns = returns_summary['AnnReturn']
                vols = returns_summary['AnnVolatility']
                sharpes = returns_summary['Sharpe']
            plt.scatter(vols, returns, s=50, alpha=0.7)
            for ticker, ret, vol in zip(returns.index, returns, vols):
                plt.annotate(ticker, xy=(vol, ret), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9)
            plt.xlabel('Annualized Volatility (%)')
            plt.ylabel('Annualized Return (%)')
            plt.title('Risk-Return Profile of Assets')
            plt.grid(True)
            plt.axhline(y=self.risk_free_rate*100, color='r', linestyle='--', 
                       label=f'Risk-Free Rate ({self.risk_free_rate*100:.1f}%)')
            plt.legend()
            plot_file = os.path.join(self.output_dir, "risk_return_profile.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Risk-return profile saved to {plot_file}")
        except Exception as e:
            print(f"⚠️ Error creating returns comparison plot: {str(e)}")

    def standardize_cache_data(self, remove_future: bool = True) -> bool:
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

def main():
    parser = argparse.ArgumentParser(description='Modern Portfolio Optimizer')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--convert-cache', action='store_true', help='Convert old cache to CSV format')  
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--risk-free', type=float, default=0.04, help='Risk-free rate')
    parser.add_argument('--margin-cost', type=float, default=0.065, help='Cost of margin')
    parser.add_argument('--tickers-file', type=str, default='src/data/tickers.csv', help='Path to tickers file')
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
    parser.add_argument('--no-standardize', action='store_true', 
                        help='Skip cache standardization step')
    parser.add_argument('--keep-future-dates', action='store_true', 
                        help='Do not remove future dates during standardization')
    parser.add_argument('--skip-plots', action='store_true', 
                        help='Skip generating plots for faster performance')
    parser.add_argument('--fast', action='store_true',
                        help='Enable all performance optimizations')
    args = parser.parse_args()
    if args.fast:
        args.skip_plots = True
    if args.clear_cache:
        cache = CSVDataCache(args.cache_dir)
        cache.clear_cache()
        print("✅ Cache cleared successfully")
    try:
        csv_path = args.tickers_file
        tickers = load_tickers(csv_path)
        print(f"📋 Loaded {len(tickers)} tickers from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to load tickers: {str(e)}")
        return
    total_start_time = time.time()
    optimizer = PortfolioOptimizer(
        tickers, 
        risk_free_rate=args.risk_free,
        margin_cost_rate=args.margin_cost,
        years=args.years,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        debug=args.debug
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
        if not args.no_standardize and HAS_STANDARDIZER:
            print("\n🧹 Standardizing cache data BEFORE processing...")
            standardized = optimizer.standardize_cache_data(remove_future=not args.keep_future_dates)
            if standardized:
                print("🔄 Reloading data from standardized cache...")
                price_data_dict = {}
                div_data_dict = {}
                for ticker in optimizer.tickers:
                    cached_price = optimizer.cache.get_price_data(ticker)
                    cached_div = optimizer.cache.get_div_data(ticker)
                    if cached_price is not None:
                        price_data_dict[ticker] = cached_price
                    if cached_div is not None:
                        div_data_dict[ticker] = cached_div
                if price_data_dict:
                    optimizer.price_data = pd.concat(price_data_dict, axis=1)
                if div_data_dict:
                    optimizer.div_data = pd.concat(div_data_dict, axis=1)
                print(f"✅ Successfully reloaded data for {len(price_data_dict)} tickers after standardization")
    except Exception as e:
        print(f"❌ Cache standardization error: {str(e)}")
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
        if args.debug and hasattr(optimizer, 'returns_summary'):
            print("\nAnnualized Return Metrics:")
            print("Top 5 tickers by return:")
            print("-" * 50)
            top_returns = optimizer.returns_summary.sort_values('AnnReturn', ascending=False).head(5)
            for ticker, row in top_returns.iterrows():
                print(f"{ticker}: Return = {row['AnnReturn']:.2f}%, " +
                     f"Volatility = {row['AnnVolatility']:.2f}%, " +
                     f"Sharpe = {row['Sharpe']:.2f}")
            print("-" * 50)
        print("🎯 Optimizing portfolio...")
        result = optimizer.optimize_portfolio(exclude_cash=args.exclude_cash, skip_plots=args.skip_plots)
        if result:
            print("\nOptimal Portfolio Allocation:")
            for ticker, weight in sorted(result['weights'].items(), key=lambda x: -x[1]):
                if weight > 0.001:
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