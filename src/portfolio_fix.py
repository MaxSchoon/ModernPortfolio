"""
Portfolio Optimization Troubleshooter

This script diagnoses and fixes issues with the portfolio optimization process.
It focuses on data quality, alignment, and calculation problems.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
from src.csv_cache_manager import CSVDataCache
from scipy.optimize import minimize
import argparse
import json

class PortfolioTroubleshooter:
    def __init__(self, cache_dir="csv_data_cache", output_dir="debug_output"):
        self.cache = CSVDataCache(cache_dir)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Parameters
        self.risk_free_rate = 0.04
        self.years = 5
        
    def load_ticker_data(self, ticker, save_plots=True):
        """
        Load and validate data for a specific ticker
        """
        print(f"\n=== Loading data for {ticker} ===")
        
        try:
            # Load price data
            price_data = self.cache.get_price_data(ticker, max_age_days=365)  # Accept data up to a year old
            
            if price_data is None:
                print(f"❌ No price data found for {ticker}")
                return None
                
            # Basic data validation
            print(f"- Data points: {len(price_data)}")
            print(f"- Date range: {price_data.index.min().date()} to {price_data.index.max().date()}")
            print(f"- Price range: ${price_data.min():.2f} to ${price_data.max():.2f}")
            
            # Check for NaN values
            nan_count = price_data.isna().sum()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count} NaN values ({nan_count/len(price_data)*100:.1f}%)")
                # Fill NaN values
                price_data = price_data.fillna(method='ffill').fillna(method='bfill')
                print("  NaN values filled with forward/backward fill")
            
            # Visualize the data
            if save_plots:
                self._plot_price_data(ticker, price_data)
            
            # Get dividend data
            div_data = self.cache.get_div_data(ticker, max_age_days=365)
            
            if div_data is not None:
                # Count dividend payments
                div_count = (div_data > 0).sum()
                print(f"- Dividend payments: {div_count}")
                
                # Calculate dividend yield
                if div_count > 0:
                    annual_div = div_data.sum()
                    avg_price = price_data.mean()
                    div_yield = annual_div / avg_price * 100
                    print(f"- Estimated dividend yield: {div_yield:.2f}%")
            else:
                print("- No dividend data available")
                div_data = pd.Series(0.0, index=price_data.index)
            
            return {
                'prices': price_data,
                'dividends': div_data
            }
            
        except Exception as e:
            print(f"❌ Error loading data for {ticker}: {type(e).__name__}: {str(e)}")
            return None
    
    def _plot_price_data(self, ticker, price_data):
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
            print(f"- Price plot saved to {plot_file}")
        except Exception as e:
            print(f"⚠️ Error creating plot for {ticker}: {str(e)}")
    
    def analyze_returns(self, tickers):
        """Analyze returns for a list of tickers"""
        print("\n=== Analyzing Returns ===")
        
        # Load data for all tickers
        ticker_data = {}
        for ticker in tickers:
            data = self.load_ticker_data(ticker)
            if data:
                ticker_data[ticker] = data
        
        if not ticker_data:
            print("❌ No valid ticker data loaded")
            return None
            
        # Align all data to common dates
        print("\nAligning data to common dates...")
        aligned_data = self._align_ticker_data(ticker_data)
        
        if aligned_data is None:
            return None
            
        prices_df = aligned_data['prices']
        div_df = aligned_data['dividends']
        
        # Print alignment info
        print(f"- Aligned data has {len(prices_df)} dates")
        print(f"- Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
        
        # Calculate returns
        print("\nCalculating returns...")
        price_returns = prices_df.pct_change().fillna(0)
        div_returns = (div_df / prices_df.shift(1)).fillna(0)
        total_returns = price_returns.add(div_returns)
        
        # Create synthetic cash asset if needed
        if 'CASH' in tickers and 'CASH' not in total_returns.columns:
            daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            total_returns['CASH'] = daily_rf
            prices_df['CASH'] = (1 + daily_rf) ** np.arange(len(prices_df))
        
        # Calculate annualized metrics
        ann_return = total_returns.mean() * 252
        ann_vol = total_returns.std() * np.sqrt(252)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol
        
        # Create a returns summary
        returns_summary = pd.DataFrame({
            'AnnReturn': ann_return * 100,
            'AnnVolatility': ann_vol * 100,
            'Sharpe': sharpe
        })
        
        # Sort by Sharpe ratio
        returns_summary = returns_summary.sort_values('Sharpe', ascending=False)
        
        print("\nReturn Metrics:")
        print(returns_summary)
        
        # Save returns summary
        summary_file = os.path.join(self.output_dir, "returns_summary.csv")
        returns_summary.to_csv(summary_file)
        print(f"\nSummary saved to {summary_file}")
        
        # Visualize the returns
        self._plot_returns_comparison(returns_summary)
        
        # Correlation analysis
        print("\nCalculating correlations...")
        corr = total_returns.corr()
        
        # Save correlations
        corr_file = os.path.join(self.output_dir, "correlations.csv")
        corr.to_csv(corr_file)
        print(f"Correlations saved to {corr_file}")
        
        # Run portfolio optimization
        print("\nRunning portfolio optimization...")
        try:
            result = self._optimize_portfolio(total_returns)
            
            # Format and save results
            if result:
                weights = result['weights']
                
                # Format weights as percentages
                weights_pct = {ticker: weight * 100 for ticker, weight in weights.items()}
                
                # Sort weights by allocation
                sorted_weights = sorted(weights_pct.items(), key=lambda x: x[1], reverse=True)
                
                print("\nOptimal Portfolio Weights:")
                for ticker, weight in sorted_weights:
                    if weight > 0.1:  # Only show weights above 0.1%
                        print(f"- {ticker}: {weight:.2f}%")
                
                # Save weights
                weights_file = os.path.join(self.output_dir, "optimal_weights.json")
                with open(weights_file, 'w') as f:
                    json.dump(weights_pct, f, indent=2)
                print(f"\nOptimal weights saved to {weights_file}")
                
                # Visualization of weights
                self._plot_portfolio_weights(weights_pct)
                
                # Plot efficient frontier
                self._plot_efficient_frontier(total_returns, result)
                
                return {
                    'returns': returns_summary,
                    'correlations': corr,
                    'optimization': result
                }
            else:
                print("❌ Portfolio optimization failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in portfolio optimization: {type(e).__name__}: {str(e)}")
            return None
    
    def _align_ticker_data(self, ticker_data):
        """Align price and dividend data for all tickers"""
        if not ticker_data:
            return None
            
        # Extract price and dividend series
        prices = {}
        dividends = {}
        for ticker, data in ticker_data.items():
            prices[ticker] = data['prices']
            dividends[ticker] = data['dividends']
        
        # Create DataFrames
        prices_df = pd.DataFrame(prices)
        div_df = pd.DataFrame(dividends)
        
        # Find common date range with sufficient data
        start_dates = []
        end_dates = []
        
        for ticker in prices_df.columns:
            ticker_data = prices_df[ticker].dropna()
            if len(ticker_data) > 0:
                start_dates.append(ticker_data.index[0])
                end_dates.append(ticker_data.index[-1])
        
        if not start_dates or not end_dates:
            print("❌ No valid date ranges found")
            return None
            
        # Use the latest start and earliest end to ensure all tickers have data
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        print(f"Common date range: {common_start.date()} to {common_end.date()}")
        
        # Trim to common date range
        prices_df = prices_df.loc[common_start:common_end]
        div_df = div_df.loc[common_start:common_end]
        
        # Fill missing values
        for ticker in prices_df.columns:
            # Calculate percentage of missing data
            missing_pct = prices_df[ticker].isna().mean() * 100
            if missing_pct > 0:
                print(f"⚠️ {ticker}: {missing_pct:.1f}% missing data in aligned range")
                
                # Only keep tickers with less than 20% missing data
                if missing_pct > 20:
                    print(f"❌ Removing {ticker} due to excessive missing data")
                    prices_df = prices_df.drop(ticker, axis=1)
                    if ticker in div_df.columns:
                        div_df = div_df.drop(ticker, axis=1)
                else:
                    # Fill missing values
                    prices_df[ticker] = prices_df[ticker].fillna(method='ffill').fillna(method='bfill')
                    print(f"  Missing values filled for {ticker}")
        
        if prices_df.empty:
            print("❌ No valid data remains after alignment")
            return None
            
        # Fill dividend data where needed
        for ticker in div_df.columns:
            if ticker in prices_df.columns:
                div_df[ticker] = div_df[ticker].fillna(0)
        
        # Ensure all price columns are also in div_df (with zeros if not present)
        for ticker in prices_df.columns:
            if ticker not in div_df.columns:
                div_df[ticker] = pd.Series(0.0, index=prices_df.index)
        
        # Final check for any remaining NaN values
        price_nan = prices_df.isna().sum().sum()
        div_nan = div_df.isna().sum().sum()
        
        if price_nan > 0 or div_nan > 0:
            print(f"⚠️ {price_nan} NaN values remain in prices, {div_nan} in dividends")
            
            # Final attempt to fill NaNs
            prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
            div_df = div_df.fillna(0)
        
        return {
            'prices': prices_df,
            'dividends': div_df
        }
    
    def _optimize_portfolio(self, returns_df):
        """Perform portfolio optimization"""
        # Calculate annualized returns and covariance
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Last check for NaN values
        if mean_returns.isna().any() or cov_matrix.isna().any().any():
            print("❌ NaN values detected in returns or covariance matrix")
            print(f"NaN in mean returns: {mean_returns[mean_returns.isna()].index.tolist()}")
            
            # Try to fix by removing problematic tickers
            valid_tickers = mean_returns[~mean_returns.isna()].index
            print(f"Attempting to continue with {len(valid_tickers)} valid tickers")
            mean_returns = mean_returns[valid_tickers]
            cov_matrix = cov_matrix.loc[valid_tickers, valid_tickers]
        
        n_assets = len(mean_returns)
        
        # Initial guess (equal weight)
        init_guess = np.array([1.0/n_assets] * n_assets)
        
        # Constraint (sum of weights = 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective function (negative Sharpe ratio)
        def neg_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Handle potential zero volatility
            if portfolio_volatility < 1e-8:
                portfolio_volatility = 1e-8
                
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        try:
            # Run optimization
            result = minimize(neg_sharpe, init_guess, 
                             method='SLSQP', 
                             bounds=bounds, 
                             constraints=constraints)
            
            if result['success']:
                # Extract optimal weights
                optimal_weights = result['x']
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(mean_returns * optimal_weights)
                portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                print(f"\nPortfolio metrics:")
                print(f"- Expected Return: {portfolio_return*100:.2f}%")
                print(f"- Volatility: {portfolio_volatility*100:.2f}%")
                print(f"- Sharpe Ratio: {portfolio_sharpe:.2f}")
                
                # Create weights dictionary
                weights_dict = dict(zip(mean_returns.index, optimal_weights))
                
                return {
                    'weights': weights_dict,
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe': portfolio_sharpe
                }
            else:
                print(f"❌ Optimization failed: {result['message']}")
                return None
                
        except Exception as e:
            print(f"❌ Error in optimization: {type(e).__name__}: {str(e)}")
            if isinstance(e, ValueError):
                print(f"Error details: {str(e)}")
            return None
    
    def _plot_returns_comparison(self, returns_summary):
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
    
    def _plot_portfolio_weights(self, weights_pct):
        """Plot portfolio weights as a pie chart"""
        try:
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
    
    def _plot_efficient_frontier(self, returns_df, optimal_result):
        """Plot the efficient frontier with the optimal portfolio"""
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        try:
            # Generate efficient frontier points
            target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
            efficient_volatilities = []
            
            # Calculate volatility for each target return
            for target in target_returns:
                # Minimize volatility subject to target return
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Constraint (sum of weights = 1 and return = target)
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}
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
            vols = np.sqrt(np.diag(cov_matrix)) * 100
            plt.scatter(vols, mean_returns * 100, s=50, alpha=0.7, label='Individual Assets')
            
            # Add labels for assets
            for i, ticker in enumerate(mean_returns.index):
                plt.annotate(ticker, xy=(vols[i], mean_returns[i] * 100), xytext=(5, 5), 
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
    parser = argparse.ArgumentParser(description="Portfolio Optimization Troubleshooter")
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers to analyze')
    parser.add_argument('--csv-file', type=str, default='tickers.csv', help='CSV file containing tickers')
    parser.add_argument('--cache-dir', type=str, default='csv_data_cache', help='Cache directory')
    parser.add_argument('--output-dir', type=str, default='portfolio_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize troubleshooter
    troubleshooter = PortfolioTroubleshooter(args.cache_dir, args.output_dir)
    
    # Get tickers to analyze
    tickers = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        # Read from CSV file
        try:
            csv_path = args.csv_file
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(os.path.dirname(__file__), csv_path)
                
            if not os.path.exists(csv_path):
                print(f"❌ CSV file not found: {csv_path}")
                return
                
            # Try to detect delimiter
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    delimiter = ';'
                elif ',' in first_line:
                    delimiter = ','
                else:
                    delimiter = ';'  # Default
            
            df = pd.read_csv(csv_path, sep=delimiter)
            
            # Find ticker column
            ticker_column = None
            for col in df.columns:
                if col.lower() == 'ticker':
                    ticker_column = col
                    break
                    
            if ticker_column is None:
                print(f"❌ No 'ticker' column found in CSV. Columns: {', '.join(df.columns)}")
                return
                
            tickers = df[ticker_column].tolist()
            print(f"Loaded {len(tickers)} tickers from {csv_path}")
                
        except Exception as e:
            print(f"❌ Error loading tickers from CSV: {str(e)}")
            return
    
    if not tickers:
        print("❌ No tickers specified")
        return
    
    # Run return analysis
    print(f"Analyzing {len(tickers)} tickers...")
    troubleshooter.analyze_returns(tickers)
    
    print(f"\n✅ Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
