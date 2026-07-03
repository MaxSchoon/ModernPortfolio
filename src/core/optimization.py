"""Mean-variance optimization engine.

Supports three portfolio regimes:

- ``long-only``       — classic Markowitz: w_i >= 0, sum(w) = 1.
- ``long-short``      — shorts allowed with full use of proceeds: sum(w) = 1,
                        gross exposure sum(|w|) <= gross_limit (1.6 = "130/30",
                        2.0 = Reg-T margin).
- ``market-neutral``  — dollar-neutral: sum(w) = 0, normalized so that gross
                        exposure sum(|w|) = 1.

Shorts are modeled with the split-variable formulation ``w = w_long - w_short``
(both halves non-negative), which keeps every exposure constraint *linear* —
``abs()`` inside SLSQP constraints violates its smoothness assumptions and is a
documented source of silent convergence failures.

The Sharpe ratio of a market-neutral portfolio uses the raw expected return
(no risk-free subtraction): the portfolio is self-financing, so its return is
already an excess return. Because the ratio is scale-invariant, the optimizer
finds the best direction and then rescales gross exposure to exactly 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import minimize

from src.core.exceptions import ConfigurationError, DataValidationError, OptimizationError

logger = logging.getLogger(__name__)

_VOL_FLOOR = 1e-12
_WEIGHT_TOL = 1e-6
_ZERO_CLIP = 1e-8


class OptimizationMode(str, Enum):
    """Portfolio regime; decides sign constraints and exposure targets."""

    LONG_ONLY = "long-only"
    LONG_SHORT = "long-short"
    MARKET_NEUTRAL = "market-neutral"


@dataclass(frozen=True)
class OptimizerConfig:
    """Validated optimizer settings.

    Attributes:
        mode: portfolio regime.
        risk_free_rate: annual risk-free rate used in the Sharpe ratio.
        max_weight: per-asset cap on long weight (fraction of capital).
        max_short: per-asset cap on short weight, as a positive fraction.
        gross_limit: cap on sum(|w|). Only meaningful for long-short
            (market-neutral portfolios are normalized to gross = 1, and
            long-only portfolios have gross = 1 by construction).
        short_borrow_rate: annual borrow fee charged on short notional.
    """

    mode: OptimizationMode = OptimizationMode.LONG_ONLY
    risk_free_rate: float = 0.04
    max_weight: float = 1.0
    max_short: float = 0.3
    gross_limit: float = 1.6
    short_borrow_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.max_weight <= 10.0:
            raise ConfigurationError(f"max_weight must be in (0, 10], got {self.max_weight}")
        if not 0.0 <= self.max_short <= 10.0:
            raise ConfigurationError(f"max_short must be in [0, 10], got {self.max_short}")
        if self.short_borrow_rate < 0.0:
            raise ConfigurationError(
                f"short_borrow_rate must be >= 0, got {self.short_borrow_rate}"
            )
        if self.mode is OptimizationMode.LONG_SHORT and self.gross_limit < 1.0:
            raise ConfigurationError(
                "gross_limit must be >= 1 for long-short portfolios: with the "
                f"budget constraint sum(w) = 1, gross exposure is at least 1 (got {self.gross_limit})"
            )
        if self.mode is OptimizationMode.MARKET_NEUTRAL and self.max_short == 0.0:
            raise ConfigurationError("market-neutral portfolios require max_short > 0")

    @property
    def allows_shorts(self) -> bool:
        return self.mode is not OptimizationMode.LONG_ONLY

    @property
    def net_exposure_target(self) -> float:
        return 0.0 if self.mode is OptimizationMode.MARKET_NEUTRAL else 1.0

    @property
    def effective_gross_limit(self) -> float:
        if self.mode is OptimizationMode.LONG_ONLY:
            return 1.0
        if self.mode is OptimizationMode.MARKET_NEUTRAL:
            return 1.0
        return self.gross_limit

    @property
    def effective_risk_free(self) -> float:
        """Rate subtracted in the Sharpe numerator (0 for self-financing portfolios)."""
        return 0.0 if self.mode is OptimizationMode.MARKET_NEUTRAL else self.risk_free_rate


@dataclass(frozen=True)
class PortfolioResult:
    """Optimized portfolio with its exposures and headline metrics."""

    tickers: tuple[str, ...]
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    mode: OptimizationMode
    net_exposure: float = field(default=0.0)
    gross_exposure: float = field(default=0.0)

    @property
    def long_exposure(self) -> float:
        return float(self.weights[self.weights > 0].sum())

    @property
    def short_exposure(self) -> float:
        """Total short notional as a positive number."""
        return float(-self.weights[self.weights < 0].sum())

    def weights_by_ticker(self) -> dict[str, float]:
        return {t: float(w) for t, w in zip(self.tickers, self.weights, strict=True)}


@dataclass(frozen=True)
class FrontierPoint:
    expected_return: float
    volatility: float


def tangency_portfolio(
    mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float
) -> np.ndarray:
    """Closed-form unconstrained max-Sharpe (tangency) weights.

    t = Σ⁻¹(μ - rf·1) / |1'Σ⁻¹(μ - rf·1)|. Used as a warm start and as the
    analytic ground truth in tests. The absolute value in the normalizer keeps
    the orientation stable when the net of the unnormalized solution is
    negative (the classic sign-flip pitfall of naive normalization).
    """
    excess = mean_returns - risk_free_rate
    raw = np.linalg.solve(cov_matrix, excess)
    denom = raw.sum()
    if abs(denom) < _VOL_FLOOR:
        raise OptimizationError(
            "tangency portfolio is undefined: net weight of the unconstrained "
            "solution is zero (all assets have identical risk-adjusted excess returns?)"
        )
    return raw / abs(denom)


class MeanVarianceOptimizer:
    """Numerical mean-variance optimizer over a fixed asset universe.

    Inputs are validated once at construction; every public method either
    returns a feasible, verified result or raises a typed exception.
    """

    def __init__(
        self,
        tickers: list[str] | tuple[str, ...],
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        config: OptimizerConfig | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        self.tickers = tuple(tickers)
        self._n = len(self.tickers)
        if self._n == 0:
            raise DataValidationError("no assets to optimize")
        self.mean_returns = np.asarray(mean_returns, dtype=float).ravel()
        self.cov_matrix = self._validated_covariance(np.asarray(cov_matrix, dtype=float))
        self._rng = np.random.default_rng(42)

        if self.mean_returns.shape != (self._n,):
            raise DataValidationError(
                f"mean_returns has shape {self.mean_returns.shape}, expected ({self._n},)"
            )
        if not np.isfinite(self.mean_returns).all():
            bad = [
                t
                for t, r in zip(self.tickers, self.mean_returns, strict=True)
                if not np.isfinite(r)
            ]
            raise DataValidationError(f"non-finite expected returns for: {', '.join(bad)}")
        if (
            self.config.mode is OptimizationMode.LONG_ONLY
            and self._max_feasible_net() < 1.0 - _WEIGHT_TOL
        ):
            raise ConfigurationError(
                f"infeasible: {self._n} assets with max_weight={self.config.max_weight} "
                "cannot sum to 1; raise max_weight or add assets"
            )

    # ------------------------------------------------------------------ setup

    def _validated_covariance(self, cov: np.ndarray) -> np.ndarray:
        n = len(self.tickers)
        if cov.shape != (n, n):
            raise DataValidationError(f"covariance has shape {cov.shape}, expected ({n}, {n})")
        if not np.isfinite(cov).all():
            raise DataValidationError("covariance matrix contains non-finite values")
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise DataValidationError("covariance matrix is not symmetric")
        cov = (cov + cov.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(cov)
        min_eig, max_eig = eigenvalues[0], eigenvalues[-1]
        if max_eig <= 0:
            raise DataValidationError("covariance matrix has no positive variance direction")
        if min_eig < -1e-8 * max_eig:
            raise DataValidationError(
                f"covariance matrix is not positive semi-definite (min eigenvalue {min_eig:.3e}); "
                "check the return series for alignment or duplication problems"
            )
        if min_eig < 0:
            # Numerical noise only: project onto the PSD cone.
            logger.debug("clipping tiny negative eigenvalue %.3e in covariance", min_eig)
            vals, vecs = np.linalg.eigh(cov)
            cov = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
        return cov

    def _max_feasible_net(self) -> float:
        return self._n * self.config.max_weight

    # ------------------------------------------------------- split formulation

    def _weights_from(self, z: np.ndarray) -> np.ndarray:
        return z[: self._n] - z[self._n :]

    def _expected_return(self, z: np.ndarray) -> float:
        w = self._weights_from(z)
        borrow_cost = self.config.short_borrow_rate * z[self._n :].sum()
        return float(w @ self.mean_returns - borrow_cost)

    def _volatility(self, z: np.ndarray) -> float:
        w = self._weights_from(z)
        variance = float(w @ self.cov_matrix @ w)
        return float(np.sqrt(max(variance, 0.0)))

    def _bounds(self) -> list[tuple[float, float]]:
        cfg = self.config
        short_cap = cfg.max_short if cfg.allows_shorts else 0.0
        return [(0.0, cfg.max_weight)] * self._n + [(0.0, short_cap)] * self._n

    def _constraints(self, extra: list[dict] | None = None) -> list[dict]:
        cfg = self.config
        constraints: list[dict] = [
            {
                "type": "eq",
                "fun": lambda z: self._weights_from(z).sum() - cfg.net_exposure_target,
            }
        ]
        if cfg.allows_shorts:
            constraints.append(
                {"type": "ineq", "fun": lambda z: cfg.effective_gross_limit - z.sum()}
            )
        return constraints + (extra or [])

    def _starting_points(self) -> list[np.ndarray]:
        n, cfg = self._n, self.config
        starts: list[np.ndarray] = []

        def as_split(w: np.ndarray) -> np.ndarray:
            longs = np.clip(w, 0.0, cfg.max_weight)
            shorts = np.clip(-w, 0.0, cfg.max_short if cfg.allows_shorts else 0.0)
            return np.concatenate([longs, shorts])

        if cfg.mode is OptimizationMode.MARKET_NEUTRAL:
            direction = self.mean_returns - self.mean_returns.mean()
            gross = np.abs(direction).sum()
            if gross > _VOL_FLOOR:
                starts.append(as_split(0.5 * direction / gross))
            starts.append(np.zeros(2 * n))
        else:
            weight = min(1.0 / n, cfg.max_weight)
            starts.append(as_split(np.full(n, weight)))
            try:
                warm = tangency_portfolio(self.mean_returns, self.cov_matrix, cfg.risk_free_rate)
                starts.append(as_split(warm))
            except (OptimizationError, np.linalg.LinAlgError):
                logger.debug("tangency warm start unavailable; using heuristic starts only")
        for _ in range(2):
            w = self._rng.uniform(-0.5 if cfg.allows_shorts else 0.0, 1.0, size=n)
            starts.append(as_split(w))
        return starts

    # ---------------------------------------------------------------- solving

    def _solve(
        self,
        objective,
        extra_constraints: list[dict] | None = None,
        context: str = "optimization",
        warm_starts: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Run SLSQP from several starts; return the best feasible solution."""
        best: tuple[float, np.ndarray] | None = None
        failures: list[str] = []
        for start in (warm_starts or []) + self._starting_points():
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=self._bounds(),
                constraints=self._constraints(extra_constraints),
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            if not result.success:
                failures.append(str(result.message))
                continue
            if not self._is_feasible(result.x, extra_constraints):
                failures.append("converged to an infeasible point")
                continue
            if best is None or result.fun < best[0]:
                best = (result.fun, result.x)
        if best is None:
            raise OptimizationError(
                f"{context} failed from every starting point "
                f"({len(failures)} attempts; last error: {failures[-1] if failures else 'none'})"
            )
        return best[1]

    def _is_feasible(self, z: np.ndarray, extra_constraints: list[dict] | None = None) -> bool:
        cfg = self.config
        w = self._weights_from(z)
        if abs(w.sum() - cfg.net_exposure_target) > _WEIGHT_TOL:
            return False
        if np.abs(w).sum() > cfg.effective_gross_limit + _WEIGHT_TOL:
            return False
        if (w > cfg.max_weight + _WEIGHT_TOL).any():
            return False
        short_cap = cfg.max_short if cfg.allows_shorts else 0.0
        if (w < -short_cap - _WEIGHT_TOL).any():
            return False
        for constraint in extra_constraints or []:
            value = constraint["fun"](z)
            if constraint["type"] == "eq" and abs(value) > _WEIGHT_TOL:
                return False
            if constraint["type"] == "ineq" and value < -_WEIGHT_TOL:
                return False
        return True

    def _finalize(self, z: np.ndarray) -> PortfolioResult:
        cfg = self.config
        weights = self._weights_from(z)
        weights[np.abs(weights) < _ZERO_CLIP] = 0.0

        if cfg.mode is OptimizationMode.MARKET_NEUTRAL:
            gross = np.abs(weights).sum()
            if gross < _ZERO_CLIP:
                raise OptimizationError(
                    "market-neutral optimization converged to the zero portfolio; "
                    "expected returns may not differentiate the assets"
                )
            # The ratio objective is scale-invariant; pin gross exposure at 1.
            weights = weights / gross

        shorts = np.clip(-weights, 0.0, None)
        expected_return = float(
            weights @ self.mean_returns - self.config.short_borrow_rate * shorts.sum()
        )
        variance = float(weights @ self.cov_matrix @ weights)
        volatility = float(np.sqrt(max(variance, 0.0)))
        if volatility < _VOL_FLOOR:
            raise OptimizationError("optimal portfolio has zero volatility; Sharpe is undefined")
        sharpe = (expected_return - cfg.effective_risk_free) / volatility
        return PortfolioResult(
            tickers=self.tickers,
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe=float(sharpe),
            mode=cfg.mode,
            net_exposure=float(weights.sum()),
            gross_exposure=float(np.abs(weights).sum()),
        )

    # ----------------------------------------------------------------- public

    def max_sharpe(self) -> PortfolioResult:
        """Maximize the Sharpe ratio under the configured mode's constraints."""
        rf = self.config.effective_risk_free

        def negative_sharpe(z: np.ndarray) -> float:
            vol = self._volatility(z)
            if vol < _VOL_FLOOR:
                return 1e6
            return -(self._expected_return(z) - rf) / vol

        solution = self._solve(negative_sharpe, context="max-Sharpe optimization")
        result = self._finalize(solution)
        logger.info(
            "max-Sharpe (%s): return=%.2f%% vol=%.2f%% sharpe=%.2f net=%.2f gross=%.2f",
            result.mode.value,
            result.expected_return * 100,
            result.volatility * 100,
            result.sharpe,
            result.net_exposure,
            result.gross_exposure,
        )
        return result

    def min_volatility(self) -> PortfolioResult:
        """Minimize volatility under the configured mode's constraints.

        Not defined for market-neutral portfolios: the zero portfolio is
        always the trivial minimizer there.
        """
        if self.config.mode is OptimizationMode.MARKET_NEUTRAL:
            raise ConfigurationError(
                "min-volatility is degenerate for market-neutral portfolios "
                "(the zero portfolio always wins); use max_sharpe or the frontier"
            )
        solution = self._solve(lambda z: self._volatility(z) ** 2, context="min-volatility")
        return self._finalize(solution)

    def efficient_frontier(self, points: int = 30) -> list[FrontierPoint]:
        """Minimum-volatility portfolios across the achievable return range."""
        if points < 2:
            raise ConfigurationError(f"frontier needs at least 2 points, got {points}")
        low, high = self._achievable_return_range()
        frontier: list[FrontierPoint] = []
        # Continuation: each point warm-starts from its neighbor's solution.
        # Without it, isolated SLSQP runs land on feasible-but-suboptimal
        # points and the frontier comes out jagged instead of convex.
        previous: np.ndarray | None = None
        for target in np.linspace(low, high, points):
            constraint = {"type": "eq", "fun": lambda z, t=target: self._expected_return(z) - t}
            try:
                z = self._solve(
                    lambda z: self._volatility(z) ** 2,
                    extra_constraints=[constraint],
                    context=f"frontier point at return {target:.2%}",
                    warm_starts=[previous] if previous is not None else None,
                )
            except OptimizationError:
                logger.debug("skipping unreachable frontier target %.2f%%", target * 100)
                continue
            previous = z
            frontier.append(
                FrontierPoint(
                    expected_return=self._expected_return(z), volatility=self._volatility(z)
                )
            )
        if len(frontier) < 2:
            raise OptimizationError("efficient frontier collapsed: fewer than 2 feasible points")
        return frontier

    def _achievable_return_range(self) -> tuple[float, float]:
        z_max = self._solve(lambda z: -self._expected_return(z), context="max-return bound")
        z_min = self._solve(self._expected_return, context="min-return bound")
        low, high = self._expected_return(z_min), self._expected_return(z_max)
        if high - low < _VOL_FLOOR:
            raise OptimizationError("all feasible portfolios have identical expected return")
        return low, high


@dataclass(frozen=True)
class KellyMetrics:
    """Kelly-criterion leverage suggestion for an optimized portfolio."""

    kelly_fraction: float
    safe_kelly: float
    leveraged_return: float
    leveraged_volatility: float
    leveraged_sharpe: float


def kelly_metrics(
    expected_return: float,
    volatility: float,
    margin_cost_rate: float,
    risk_free_rate: float,
    leverage_cap: float = 2.0,
) -> KellyMetrics:
    """Portfolio-level Kelly leverage.

    kelly = (μ - c) / σ² where c is the margin borrow rate. The applied
    ("safe") leverage is clamped to [0, leverage_cap]. Margin interest is
    charged only on borrowed capital (leverage above 1); capital left
    uninvested at leverage below 1 earns the risk-free rate.
    """
    if volatility <= 0:
        raise DataValidationError(f"volatility must be positive, got {volatility}")
    if leverage_cap <= 0:
        raise ConfigurationError(f"leverage_cap must be positive, got {leverage_cap}")
    kelly = (expected_return - margin_cost_rate) / volatility**2
    safe = float(np.clip(kelly, 0.0, leverage_cap))
    borrowed = max(safe - 1.0, 0.0)
    idle = max(1.0 - safe, 0.0)
    lev_return = safe * expected_return - borrowed * margin_cost_rate + idle * risk_free_rate
    lev_vol = safe * volatility
    lev_sharpe = (lev_return - risk_free_rate) / lev_vol if lev_vol > _VOL_FLOOR else 0.0
    return KellyMetrics(
        kelly_fraction=float(kelly),
        safe_kelly=safe,
        leveraged_return=float(lev_return),
        leveraged_volatility=float(lev_vol),
        leveraged_sharpe=float(lev_sharpe),
    )
