"""
Graphing utilities for the Modern Portfolio Optimizer.
This module contains functions for plotting portfolio data and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize

def plot_price_data(ticker: str, price_data: pd.Series, output_dir: str) -> None:
    """
    Plot price data for a single ticker and save to file
    
    Args:
        ticker: The ticker symbol
        price_data: The price data as a pandas Series
        output_dir: Directory to save the plot in
    """
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
    plot_file = os.path.join(output_dir, f"{ticker}_price.png")
    plt.savefig(plot_file)
    plt.close()

def plot_portfolio_weights(weights: Dict[str, float], output_dir: str, allow_shorts: bool = False) -> None:
    """
    Plot portfolio weights as a pie chart, handling both long and short positions
    
    Args:
        weights: Dictionary of ticker to weight
        output_dir: Directory to save the plot in
        allow_shorts: Whether short selling is allowed
    """
    # Convert weights to percentages for display
    weights_pct = {k: v*100 for k, v in weights.items()}
    
    # Separate long and short positions
    long_weights = {k: v for k, v in weights_pct.items() if v > 0}
    short_weights = {k: abs(v) for k, v in weights_pct.items() if v < 0}
    
    # Count number of plots needed
    has_long = len(long_weights) > 0
    has_short = len(short_weights) > 0
    
    # Determine filename suffix based on shorts being allowed and used
    filename_suffix = "_shorts" if allow_shorts and has_short else ""
    
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
    plot_file = os.path.join(output_dir, f"portfolio_allocation{filename_suffix}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Portfolio allocation chart saved to {plot_file}")

def plot_efficient_frontier(
    mean_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    tickers: List[str], 
    optimal_result: Dict, 
    risk_free_rate: float, 
    allow_shorts: bool = False, 
    max_long: float = 1.0, 
    max_short: float = 0.3, 
    actual_long: float = None,
    actual_short: float = None,
    output_dir: str = "."
) -> None:
    """
    Plot the efficient frontier with the optimal portfolio
    
    Args:
        mean_returns: Series of mean returns for each ticker
        cov_matrix: Covariance matrix
        tickers: List of ticker symbols
        optimal_result: Optimal portfolio allocation and metrics
        risk_free_rate: Risk-free rate
        allow_shorts: Whether short selling is allowed
        max_long: Maximum allocation to long positions
        max_short: Maximum allocation to short positions
        actual_long: Actual long exposure in the optimized portfolio
        actual_short: Actual short exposure in the optimized portfolio
        output_dir: Directory to save the plot in
    """
    # Reduce the number of points for faster computation
    num_points = 30
    target_returns = np.linspace(min(mean_returns), max(mean_returns), num_points)
    efficient_volatilities = []
    
    # Convert to numpy arrays for faster computation
    mean_returns_array = mean_returns.values
    cov_matrix_array = cov_matrix.values
    
    # Set bounds based on whether shorts are allowed
    n_assets = len(mean_returns)
    
    # Determine filename suffix based on shorts being allowed
    filename_suffix = "_shorts" if allow_shorts else ""
    
    if allow_shorts:
        # Implementation of the reparameterized approach (2n variables approach)
        # Each asset has a long component and a short component
        
        # For 2n-approach, create bounds for long and short parts separately
        # Long parts are bounded between 0 and max_long
        # Short parts are bounded between 0 and max_short
        bounds = tuple([(0, max_long) for _ in range(n_assets)] + [(0, max_short) for _ in range(n_assets)])
        
        # Define constraints for the 2n-approach
        # 1. Sum of all components (long and short) equals 1 (gross exposure = 100%)
        sum_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Function to convert 2n-weights back to n-weights for evaluation
        def get_net_weights(x):
            return x[:n_assets] - x[n_assets:]
            
        # For existing optimization points calculation
        for target in target_returns:
            # Portfolio volatility using net weights
            def portfolio_volatility(x):
                weights = get_net_weights(x)
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_array, weights)))
            
            # Target return constraint using net weights
            def target_return_constraint(x):
                weights = get_net_weights(x)
                return np.sum(mean_returns_array * weights) - target
            
            # Define all constraints
            constraints = [
                sum_constraint,
                {'type': 'eq', 'fun': target_return_constraint}
            ]
            
            # Create initial guess with proper distribution
            # Start with a balanced allocation
            init_guess = np.zeros(2 * n_assets)
            # Set long components
            init_guess[:n_assets] = np.random.random(n_assets) * max_long * 0.9
            # Set short components
            init_guess[n_assets:] = np.random.random(n_assets) * max_short * 0.9
            # Normalize to sum to 1
            init_guess = init_guess / np.sum(init_guess)
            
            try:
                # Try optimization with multiple attempts
                for attempt in range(3):
                    try:
                        result = minimize(portfolio_volatility, init_guess, 
                                       method='SLSQP', bounds=bounds, constraints=constraints,
                                       options={'maxiter': 1000, 'ftol': 1e-8})
                        if result['success']:
                            efficient_volatilities.append(result['fun'])
                            break
                        else:
                            # Try a different initial guess
                            init_guess = np.random.random(2 * n_assets)
                            # Ensure positive components
                            init_guess = np.abs(init_guess)
                            # Normalize
                            init_guess = init_guess / np.sum(init_guess)
                    except:
                        pass
                
                # If all attempts failed, mark this point as NaN
                if attempt == 2 and not result['success']:
                    efficient_volatilities.append(np.nan)
            except:
                efficient_volatilities.append(np.nan)
    else:
        # Original approach for long-only
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Calculate volatility for each target return
        for target in target_returns:
            # Minimize volatility subject to target return
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_array, weights)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(mean_returns_array * x) - target}
            ]
            
            init_guess = np.array([1.0/n_assets] * n_assets)
            
            try:
                # Try multiple times with different initial guesses if needed
                for attempt in range(3):
                    try:
                        result = minimize(portfolio_volatility, init_guess, 
                                       method='SLSQP', bounds=bounds, constraints=constraints,
                                       options={'maxiter': 1000, 'ftol': 1e-8})
                        if result['success']:
                            efficient_volatilities.append(result['fun'])
                            break
                        else:
                            # Try a different initial guess
                            init_guess = np.random.random(n_assets)
                            init_guess = init_guess / np.sum(init_guess)
                    except:
                        pass
                
                # If all attempts failed, mark this point as NaN
                if attempt == 2 and not result['success']:
                    efficient_volatilities.append(np.nan)
            except:
                efficient_volatilities.append(np.nan)
    
    # Filter out any failed optimizations
    valid = ~np.isnan(efficient_volatilities)
    efficient_volatilities = np.array(efficient_volatilities)[valid]
    efficient_returns = target_returns[valid]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot efficient frontier if we have valid points
    if len(efficient_volatilities) > 1:
        plt.plot(efficient_volatilities * 100, efficient_returns * 100, 'b-', label='Efficient Frontier')
    
    # Plot optimal portfolio
    opt_return = optimal_result['return'] * 100
    opt_vol = optimal_result['volatility'] * 100
    plt.scatter(opt_vol, opt_return, s=100, color='r', marker='*', label='Optimal Portfolio')
    
    # Plot individual assets
    vols = np.sqrt(np.diag(cov_matrix)) * 100
    plt.scatter(vols, mean_returns.values * 100, s=50, alpha=0.7, label='Individual Assets')
    
    # Add labels for assets
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, xy=(vols[i], mean_returns.iloc[i] * 100), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    title = 'Efficient Frontier and Optimal Portfolio'
    if allow_shorts:
        if actual_long is not None and actual_short is not None:
            title += f' (with Short Selling: {actual_long*100:.2f}% Long, {actual_short*100:.2f}% Short)'
        else:
            title += ' (with Short Selling)'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Save plot with appropriate suffix
    plot_file = os.path.join(output_dir, f"efficient_frontier{filename_suffix}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Efficient frontier plot saved to {plot_file}")

def plot_returns_comparison(returns_summary: pd.DataFrame, risk_free_rate: float, output_dir: str, allow_shorts: bool = False) -> None:
    """
    Plot comparison of returns and volatility
    
    Args:
        returns_summary: DataFrame with return metrics for each asset
        risk_free_rate: Risk-free rate
        output_dir: Directory to save the plot in
        allow_shorts: Whether short selling is allowed
    """
    # Determine filename suffix based on shorts being allowed
    filename_suffix = "_shorts" if allow_shorts else ""
    
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
    if allow_shorts:
        title += ' (Short Selling Enabled)'
    plt.title(title)
    plt.grid(True)
    
    # Add risk-free rate as horizontal line
    plt.axhline(y=risk_free_rate*100, color='r', linestyle='--', 
               label=f'Risk-Free Rate ({risk_free_rate*100:.1f}%)')
    
    plt.legend()
    
    # Save plot with appropriate suffix
    plot_file = os.path.join(output_dir, f"risk_return_profile{filename_suffix}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Risk-return profile saved to {plot_file}")
