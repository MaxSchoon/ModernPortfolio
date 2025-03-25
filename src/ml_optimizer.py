"""
Machine Learning based Portfolio Optimization.
This module provides ML-based alternatives to traditional optimization methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

class MLPortfolioOptimizer:
    """
    Machine Learning based Portfolio Optimizer that uses clustering and
    dimensionality reduction techniques to optimize portfolio allocation.
    """
    
    def __init__(self, 
                mean_returns: pd.Series, 
                cov_matrix: pd.DataFrame, 
                tickers: List[str],
                risk_free_rate: float = 0.04,
                margin_cost_rate: float = 0.065,
                shorts: bool = False,
                max_long: float = 0.8,
                max_short: float = 0.2,
                debug: bool = False):
        """
        Initialize the ML Portfolio Optimizer.
        
        Args:
            mean_returns: Series of mean returns for each ticker
            cov_matrix: Covariance matrix of returns
            tickers: List of ticker symbols
            risk_free_rate: Risk-free rate
            margin_cost_rate: Cost of margin
            shorts: Whether short selling is allowed
            max_long: Maximum allocation to long positions
            max_short: Maximum allocation to short positions
            debug: Enable debug output
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.margin_cost_rate = margin_cost_rate
        self.shorts = shorts
        self.max_long = max_long
        self.max_short = max_short
        self.debug = debug
    
    def optimize(self) -> Dict:
        """
        Perform ML-based portfolio optimization.
        
        Returns:
            Dictionary with optimization results including weights and metrics
        """
        start_time = time.time()
        
        if self.debug:
            print("Starting ML-based portfolio optimization...")
        
        # Method selection based on problem size
        n_assets = len(self.tickers)
        
        if n_assets > 50:
            # For large portfolios, use clustering
            result = self._optimize_with_clustering()
        else:
            # For smaller portfolios, use PCA-based approach
            result = self._optimize_with_pca()
        
        optimization_time = time.time() - start_time
        if self.debug:
            print(f"ML optimization completed in {optimization_time:.2f} seconds")
        
        return result
    
    def _optimize_with_clustering(self) -> Dict:
        """
        Use K-means clustering to group similar assets and allocate within clusters.
        
        Returns:
            Dictionary with optimization results
        """
        print("Using K-means clustering for portfolio optimization")
        
        # Prepare data for clustering
        # Features: returns, volatility, and correlations
        features = []
        
        # Add return and volatility for each asset
        for ticker in self.tickers:
            returns = self.mean_returns[ticker]
            volatility = np.sqrt(self.cov_matrix.loc[ticker, ticker])
            sharpe = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Average correlation with other assets
            correlations = self.cov_matrix[ticker].drop(ticker)
            avg_correlation = correlations.mean()
            
            features.append([returns, volatility, sharpe, avg_correlation])
        
        # Convert to numpy array and scale features
        features = np.array(features)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        n_assets = len(self.tickers)
        # Determine optimal number of clusters (between 4 and 12 or n_assets/10, whichever is smaller)
        max_clusters = min(12, max(4, n_assets // 10))
        
        # Using silhouette score to find optimal k would be better, but for simplicity:
        n_clusters = max(3, min(int(np.sqrt(len(self.tickers)) / 2), max_clusters))
        
        if self.debug:
            print(f"Creating {n_clusters} clusters from {len(self.tickers)} assets")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Allocate weights within each cluster
        weights = np.zeros(len(self.tickers))
        
        # Process clusters
        for cluster_id in range(n_clusters):
            # Get indices of assets in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_tickers = [self.tickers[i] for i in cluster_indices]
            
            if self.debug:
                print(f"Cluster {cluster_id} has {len(cluster_indices)} assets: {', '.join(cluster_tickers[:5])}" +
                      ("..." if len(cluster_tickers) > 5 else ""))
            
            # Extract return and covariance data for this cluster
            cluster_returns = self.mean_returns.iloc[cluster_indices].values
            cluster_cov = self.cov_matrix.iloc[cluster_indices, cluster_indices].values
            
            # Allocate within cluster based on Sharpe ratio
            cluster_vol = np.sqrt(np.diag(cluster_cov))
            cluster_sharpe = np.zeros_like(cluster_returns)
            
            # Calculate Sharpe ratio for each asset in the cluster
            for i in range(len(cluster_returns)):
                if cluster_vol[i] > 0:
                    cluster_sharpe[i] = (cluster_returns[i] - self.risk_free_rate) / cluster_vol[i]
            
            # Based on Sharpe ratios, determine allocations
            # Use softmax-like function to convert Sharpes to weights
            if self.shorts:
                # For short selling, we want negative Sharpe ratios for shorts
                # Assign shorts to negative Sharpe assets, longs to positive Sharpe
                pos_sharpe_indices = cluster_sharpe > 0
                neg_sharpe_indices = cluster_sharpe < 0
                
                # Skip neutral assets (Sharpe ≈ 0)
                pos_sharpe = cluster_sharpe[pos_sharpe_indices]
                neg_sharpe = cluster_sharpe[neg_sharpe_indices]
                
                # Allocate longs
                if np.sum(pos_sharpe_indices) > 0:
                    pos_weights = np.exp(pos_sharpe - np.max(pos_sharpe))
                    pos_weights = pos_weights / np.sum(pos_weights)
                    
                    # Scale by cluster importance and maximum long allocation
                    pos_scale = self.max_long * (len(cluster_indices) / len(self.tickers))
                    cluster_weights = np.zeros(len(cluster_indices))
                    cluster_weights[pos_sharpe_indices] = pos_weights * pos_scale
                    
                # Allocate shorts
                if np.sum(neg_sharpe_indices) > 0:
                    # Use absolute values for calculation, then negate
                    abs_neg_sharpe = np.abs(neg_sharpe)
                    neg_weights = np.exp(abs_neg_sharpe - np.max(abs_neg_sharpe))
                    neg_weights = neg_weights / np.sum(neg_weights)
                    
                    # Scale by cluster importance and maximum short allocation
                    neg_scale = self.max_short * (len(cluster_indices) / len(self.tickers))
                    cluster_weights[neg_sharpe_indices] = -neg_weights * neg_scale
            else:
                # Long-only: focus on positive Sharpe ratios
                pos_sharpe_indices = cluster_sharpe > 0
                
                if np.sum(pos_sharpe_indices) > 0:
                    # Use only positive Sharpe assets
                    pos_sharpe = cluster_sharpe[pos_sharpe_indices]
                    pos_weights = np.exp(pos_sharpe - np.max(pos_sharpe))
                    pos_weights = pos_weights / np.sum(pos_weights)
                    
                    # Scale weights by cluster size
                    pos_scale = len(cluster_indices) / len(self.tickers)
                    cluster_weights = np.zeros(len(cluster_indices))
                    cluster_weights[pos_sharpe_indices] = pos_weights * pos_scale
                else:
                    # If no positive Sharpe assets, use equal weights
                    cluster_weights = np.ones(len(cluster_indices)) / len(cluster_indices)
                    cluster_weights = cluster_weights * (len(cluster_indices) / len(self.tickers))
            
            # Assign cluster weights to global weights
            weights[cluster_indices] = cluster_weights
        
        # Normalize to ensure sum to 1
        # First check if we need to trim excesses
        total_long = np.sum(np.maximum(weights, 0))
        total_short = np.sum(np.abs(np.minimum(weights, 0)))
        
        if total_long > self.max_long:
            # Scale down long positions
            long_indices = weights > 0
            weights[long_indices] = weights[long_indices] * (self.max_long / total_long)
            
        if total_short > self.max_short:
            # Scale down short positions
            short_indices = weights < 0
            weights[short_indices] = weights[short_indices] * (self.max_short / total_short)
        
        # Ensure sum is exactly 1
        adjustment = (1.0 - np.sum(weights)) / len(weights)
        weights = weights + adjustment
        
        # Calculate portfolio metrics
        port_return = np.sum(self.mean_returns.values * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
        port_sharpe = (port_return - self.risk_free_rate) / port_vol
        
        # Calculate Kelly metrics
        kelly_metrics = self._calculate_kelly(port_return, port_vol)
        
        # Create a dictionary of weights
        weight_dict = dict(zip(self.tickers, weights))
        
        return {
            'weights': weight_dict,
            'return': port_return,
            'volatility': port_vol,
            'sharpe': port_sharpe,
            'kelly_metrics': kelly_metrics,
            'method': 'clustering'
        }
    
    def _optimize_with_pca(self) -> Dict:
        """
        Use Principal Component Analysis to reduce dimensionality and optimize.
        
        Returns:
            Dictionary with optimization results
        """
        print("Using PCA-based approach for portfolio optimization")
        
        # Prepare feature matrix with returns and scaled covariance
        returns = self.mean_returns.values
        
        # Use PCA to analyze the covariance structure
        pca = PCA(n_components=min(5, len(self.tickers) - 1))
        
        # Standardize returns for PCA
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns.reshape(-1, 1)).flatten()
        
        # Apply PCA to covariance matrix
        pca_result = pca.fit_transform(self.cov_matrix.values)
        
        # Get PCA components and eigenvalues (risk factors)
        components = pca.components_
        eigenvalues = pca.explained_variance_
        
        if self.debug:
            print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
            
        # Use PCA insights to allocate weights
        weights = np.zeros(len(self.tickers))
        
        # Calculate factor exposures and risk-adjusted returns
        factor_exposures = np.abs(components[0]) # Exposure to first principal component (market risk)
        
        # Calculate risk-adjusted returns (similar to Sharpe)
        risk_adjusted_returns = scaled_returns / factor_exposures
        
        # Assign weights based on risk-adjusted returns and PCA loadings
        if self.shorts:
            # Separate assets into long and short candidates
            long_candidates = risk_adjusted_returns > 0
            short_candidates = risk_adjusted_returns < 0
            
            # Allocate to long positions
            if np.sum(long_candidates) > 0:
                long_scores = risk_adjusted_returns[long_candidates]
                # Use softmax-like function to convert scores to weights
                long_weights = np.exp(long_scores - np.max(long_scores))
                long_weights = long_weights / np.sum(long_weights) * self.max_long
                weights[long_candidates] = long_weights
            
            # Allocate to short positions
            if np.sum(short_candidates) > 0:
                short_scores = -risk_adjusted_returns[short_candidates]  # Convert to positive for calculation
                # Use softmax-like function for shorts
                short_weights = np.exp(short_scores - np.max(short_scores))
                short_weights = short_weights / np.sum(short_weights) * -self.max_short  # Negative for shorts
                weights[short_candidates] = short_weights
        else:
            # Long-only optimization with risk adjustment
            # Ignore negative risk-adjusted returns
            positive_returns = np.maximum(risk_adjusted_returns, 0)
            
            # Skip if all returns are negative
            if np.sum(positive_returns) > 0:
                weights = positive_returns / np.sum(positive_returns)
            else:
                # Equally weight all assets
                weights = np.ones(len(self.tickers)) / len(self.tickers)
        
        # Ensure weights sum to 1
        adjustment = (1.0 - np.sum(weights)) / len(weights)
        weights = weights + adjustment
        
        # Calculate portfolio metrics
        port_return = np.sum(self.mean_returns.values * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
        port_sharpe = (port_return - self.risk_free_rate) / port_vol
        
        # Calculate Kelly metrics
        kelly_metrics = self._calculate_kelly(port_return, port_vol)
        
        # Create a dictionary of weights
        weight_dict = dict(zip(self.tickers, weights))
        
        return {
            'weights': weight_dict,
            'return': port_return,
            'volatility': port_vol,
            'sharpe': port_sharpe,
            'kelly_metrics': kelly_metrics,
            'method': 'pca'
        }
    
    def _calculate_kelly(self, portfolio_ret: float, portfolio_vol: float) -> Dict:
        """
        Calculate Kelly Criterion and leveraged metrics.
        
        Args:
            portfolio_ret: Portfolio return
            portfolio_vol: Portfolio volatility
            
        Returns:
            Dictionary with Kelly metrics
        """
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
