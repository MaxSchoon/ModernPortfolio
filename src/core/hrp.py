"""Hierarchical Risk Parity (HRP) allocation.

The clustering-based allocator from López de Prado (2016), "Building
Diversified Portfolios that Outperform Out-of-Sample" (Journal of Portfolio
Management 42(4)):

1. Convert the correlation matrix into the distance d = sqrt((1 - ρ) / 2).
2. Hierarchically cluster the assets on that distance (single linkage).
3. Quasi-diagonalize: reorder assets in dendrogram-leaf order so similar
   assets sit next to each other.
4. Recursively bisect the ordered list, splitting capital between the two
   halves in inverse proportion to their inverse-variance cluster risk.

HRP never inverts the covariance matrix and never uses expected returns, so
it stays stable where mean-variance optimization is noise-amplifying — and it
works on singular covariance matrices. It is long-only by construction: the
recursive bisection only ever splits capital, never signs it. For short or
market-neutral portfolios use ``MeanVarianceOptimizer``.

Expected returns are accepted only to *report* the resulting portfolio's
return and Sharpe ratio; they never influence the weights.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from src.core.exceptions import DataValidationError
from src.core.optimization import OptimizationMode, PortfolioResult

logger = logging.getLogger(__name__)

_VOL_FLOOR = 1e-12


class HRPOptimizer:
    """Hierarchical Risk Parity allocator over a fixed asset universe.

    Args:
        tickers: asset names, aligned with ``mean_returns`` / ``cov_matrix``.
        mean_returns: annualized expected returns (reporting only).
        cov_matrix: annualized covariance matrix; may be singular but must be
            symmetric with strictly positive variances.
        risk_free_rate: annual rate used in the reported Sharpe ratio.
    """

    def __init__(
        self,
        tickers: list[str] | tuple[str, ...],
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.tickers = tuple(tickers)
        self._n = len(self.tickers)
        if self._n == 0:
            raise DataValidationError("no assets to allocate")
        self.mean_returns = np.asarray(mean_returns, dtype=float).ravel()
        self.cov_matrix = np.asarray(cov_matrix, dtype=float)
        self.risk_free_rate = risk_free_rate

        if self.mean_returns.shape != (self._n,):
            raise DataValidationError(
                f"mean_returns has shape {self.mean_returns.shape}, expected ({self._n},)"
            )
        if self.cov_matrix.shape != (self._n, self._n):
            raise DataValidationError(
                f"covariance has shape {self.cov_matrix.shape}, expected ({self._n}, {self._n})"
            )
        if not np.isfinite(self.mean_returns).all() or not np.isfinite(self.cov_matrix).all():
            raise DataValidationError("inputs contain non-finite values")
        if not np.allclose(self.cov_matrix, self.cov_matrix.T, atol=1e-8):
            raise DataValidationError("covariance matrix is not symmetric")
        variances = np.diag(self.cov_matrix)
        if (variances <= 0).any():
            bad = [t for t, v in zip(self.tickers, variances, strict=True) if v <= 0]
            raise DataValidationError(
                "inverse-variance allocation requires strictly positive variances; "
                f"zero/negative variance for: {', '.join(bad)}"
            )
        near_zero = variances < 1e-7
        if near_zero.any():
            # Inverse-variance allocation concentrates almost everything in a
            # near-riskless asset — usually the synthetic CASH proxy. Valid
            # math, but rarely what the user meant; say so out loud.
            names = [t for t, flag in zip(self.tickers, near_zero, strict=True) if flag]
            logger.warning(
                "near-zero-variance assets present (%s); HRP will concentrate in them — "
                "consider --exclude-cash",
                ", ".join(names),
            )

    # ------------------------------------------------------------- algorithm

    def allocate(self) -> PortfolioResult:
        """Compute HRP weights and report the portfolio's headline metrics."""
        if self._n == 1:
            weights = np.array([1.0])
        else:
            order = self._quasi_diagonal_order()
            weights = self._recursive_bisection(order)

        expected_return = float(weights @ self.mean_returns)
        volatility = float(np.sqrt(max(weights @ self.cov_matrix @ weights, 0.0)))
        if volatility < _VOL_FLOOR:
            raise DataValidationError("HRP portfolio has zero volatility; Sharpe is undefined")
        sharpe = (expected_return - self.risk_free_rate) / volatility
        result = PortfolioResult(
            tickers=self.tickers,
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe=float(sharpe),
            mode=OptimizationMode.LONG_ONLY,
            net_exposure=float(weights.sum()),
            gross_exposure=float(np.abs(weights).sum()),
        )
        logger.info(
            "HRP: return=%.2f%% vol=%.2f%% sharpe=%.2f",
            result.expected_return * 100,
            result.volatility * 100,
            result.sharpe,
        )
        return result

    def _quasi_diagonal_order(self) -> list[int]:
        """Dendrogram-leaf order from single-linkage clustering on the
        correlation distance d = sqrt((1 - ρ) / 2)."""
        vols = np.sqrt(np.diag(self.cov_matrix))
        corr = self.cov_matrix / np.outer(vols, vols)
        corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
        distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        tree = linkage(condensed, method="single")
        return [int(i) for i in leaves_list(tree)]

    def _recursive_bisection(self, order: list[int]) -> np.ndarray:
        """Split capital top-down; each split is inverse to cluster variance."""
        weights = np.ones(self._n)
        stack: list[list[int]] = [order]
        while stack:
            cluster = stack.pop()
            if len(cluster) < 2:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            left_var = self._inverse_variance_cluster_risk(left)
            right_var = self._inverse_variance_cluster_risk(right)
            left_share = 1.0 - left_var / (left_var + right_var)
            weights[left] *= left_share
            weights[right] *= 1.0 - left_share
            stack.extend((left, right))
        return weights

    def _inverse_variance_cluster_risk(self, indices: list[int]) -> float:
        """Variance of the cluster under its own inverse-variance weighting."""
        sub_cov = self.cov_matrix[np.ix_(indices, indices)]
        inverse_variance = 1.0 / np.diag(sub_cov)
        inverse_variance /= inverse_variance.sum()
        return float(inverse_variance @ sub_cov @ inverse_variance)
