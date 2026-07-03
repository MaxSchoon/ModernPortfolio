"""Falsification suite for the mean-variance engine.

Each test states an observation that would refute the optimizer's correctness:
an analytic solution it fails to reproduce, a constraint it violates, a
feasible portfolio it fails to beat, or a degenerate input it fails to reject.
"""

import numpy as np
import pytest
from src.core.exceptions import ConfigurationError, DataValidationError, OptimizationError
from src.core.optimization import (
    MeanVarianceOptimizer,
    OptimizationMode,
    OptimizerConfig,
    kelly_metrics,
    tangency_portfolio,
)

RF = 0.02
TOL = 1e-6


def make_optimizer(mu, cov, mode, **overrides):
    tickers = [f"A{i}" for i in range(len(mu))]
    defaults = {"mode": mode, "risk_free_rate": RF, "max_short": 1.0, "gross_limit": 3.0}
    defaults.update(overrides)
    return MeanVarianceOptimizer(tickers, np.array(mu), np.array(cov), OptimizerConfig(**defaults))


@pytest.fixture
def interior_universe():
    """Three assets whose unconstrained tangency portfolio is long-only and interior."""
    mu = [0.10, 0.07, 0.04]
    cov = np.diag([0.04, 0.03, 0.02])
    return mu, cov


@pytest.fixture
def short_universe():
    """Asset A2 has a negative excess return; the tangency solution shorts it."""
    mu = [0.10, 0.02, -0.06]
    cov = np.diag([0.04, 0.03, 0.05])
    return mu, cov


def sharpe_of(weights, mu, cov, rf=RF):
    ret = weights @ np.array(mu)
    vol = np.sqrt(weights @ np.array(cov) @ weights)
    return (ret - rf) / vol


# --------------------------------------------------------------- ground truth


class TestAnalyticGroundTruth:
    def test_long_only_recovers_interior_tangency(self, interior_universe):
        mu, cov = interior_universe
        analytic = tangency_portfolio(np.array(mu), cov, RF)
        result = make_optimizer(mu, cov, OptimizationMode.LONG_ONLY).max_sharpe()
        np.testing.assert_allclose(result.weights, analytic, atol=1e-4)

    def test_long_short_recovers_interior_tangency(self, interior_universe):
        mu, cov = interior_universe
        analytic = tangency_portfolio(np.array(mu), cov, RF)
        result = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT).max_sharpe()
        np.testing.assert_allclose(result.weights, analytic, atol=1e-4)

    def test_long_short_matches_analytic_sharpe_with_shorts(self, short_universe):
        # Caps widened so the analytic solution (weights ≈ [5, 0, -4]) is interior.
        mu, cov = short_universe
        analytic = tangency_portfolio(np.array(mu), cov, RF)
        assert analytic[2] < 0, "test premise: analytic solution must short A2"
        result = make_optimizer(
            mu, cov, OptimizationMode.LONG_SHORT, max_weight=10.0, max_short=10.0, gross_limit=10.0
        ).max_sharpe()
        assert result.sharpe == pytest.approx(sharpe_of(analytic, mu, cov), abs=1e-4)

    def test_tangency_undefined_when_net_is_zero(self):
        # Symmetric universe: excess returns sum to zero under Σ⁻¹.
        mu = np.array([RF + 0.05, RF - 0.05])
        cov = np.diag([0.04, 0.04])
        with pytest.raises(OptimizationError, match="undefined"):
            tangency_portfolio(mu, cov, RF)


# ------------------------------------------------------------------ regimes


class TestLongOnly:
    def test_never_shorts_a_losing_asset(self, short_universe):
        mu, cov = short_universe
        result = make_optimizer(mu, cov, OptimizationMode.LONG_ONLY).max_sharpe()
        assert (result.weights >= -TOL).all()
        assert result.weights[2] == pytest.approx(0.0, abs=1e-4)
        assert result.net_exposure == pytest.approx(1.0, abs=TOL)
        assert result.gross_exposure == pytest.approx(1.0, abs=TOL)

    def test_single_asset_gets_full_weight(self):
        result = make_optimizer([0.08], [[0.04]], OptimizationMode.LONG_ONLY).max_sharpe()
        assert result.weights[0] == pytest.approx(1.0, abs=TOL)

    def test_respects_max_weight_cap(self, interior_universe):
        mu, cov = interior_universe
        result = make_optimizer(mu, cov, OptimizationMode.LONG_ONLY, max_weight=0.4).max_sharpe()
        assert (result.weights <= 0.4 + TOL).all()
        assert result.net_exposure == pytest.approx(1.0, abs=TOL)


class TestLongShort:
    def test_shorts_the_losing_asset(self, short_universe):
        mu, cov = short_universe
        result = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT).max_sharpe()
        assert result.weights[2] < -0.01, "optimizer left Sharpe on the table by not shorting"
        assert result.net_exposure == pytest.approx(1.0, abs=TOL)

    def test_respects_gross_limit(self, short_universe):
        mu, cov = short_universe
        result = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT, gross_limit=1.6).max_sharpe()
        assert result.gross_exposure <= 1.6 + TOL

    def test_gross_limit_of_one_forbids_shorting(self, short_universe):
        mu, cov = short_universe
        result = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT, gross_limit=1.0).max_sharpe()
        assert (result.weights >= -TOL).all()

    def test_dominates_long_only(self, short_universe):
        mu, cov = short_universe
        long_only = make_optimizer(mu, cov, OptimizationMode.LONG_ONLY).max_sharpe()
        long_short = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT).max_sharpe()
        assert long_short.sharpe >= long_only.sharpe - TOL

    def test_respects_max_short_per_asset(self, short_universe):
        mu, cov = short_universe
        result = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT, max_short=0.1).max_sharpe()
        assert (result.weights >= -0.1 - TOL).all()

    def test_borrow_cost_discourages_shorting(self, short_universe):
        mu, cov = short_universe
        free = make_optimizer(mu, cov, OptimizationMode.LONG_SHORT).max_sharpe()
        costly = make_optimizer(
            mu, cov, OptimizationMode.LONG_SHORT, short_borrow_rate=1.0
        ).max_sharpe()
        assert costly.short_exposure < free.short_exposure + TOL
        assert costly.short_exposure == pytest.approx(0.0, abs=1e-3)


class TestMarketNeutral:
    def test_is_dollar_neutral_with_unit_gross(self, short_universe):
        mu, cov = short_universe
        result = make_optimizer(mu, cov, OptimizationMode.MARKET_NEUTRAL).max_sharpe()
        assert result.net_exposure == pytest.approx(0.0, abs=1e-4)
        assert result.gross_exposure == pytest.approx(1.0, abs=1e-4)

    def test_longs_winners_and_shorts_losers(self):
        mu = [0.12, -0.02]
        cov = np.diag([0.04, 0.04])
        result = make_optimizer(mu, cov, OptimizationMode.MARKET_NEUTRAL).max_sharpe()
        assert result.weights[0] > 0.0
        assert result.weights[1] < 0.0
        assert result.sharpe > 0.0

    def test_undifferentiated_universe_yields_no_fake_edge(self):
        # With identical expected returns every dollar-neutral portfolio earns 0:
        # the optimizer must either refuse or report a Sharpe of ~0 — a positive
        # Sharpe here would be fabricated.
        mu = [0.05, 0.05]
        cov = np.diag([0.04, 0.04])
        try:
            result = make_optimizer(mu, cov, OptimizationMode.MARKET_NEUTRAL).max_sharpe()
        except OptimizationError:
            return
        assert abs(result.sharpe) < 1e-6
        assert result.net_exposure == pytest.approx(0.0, abs=1e-4)

    def test_min_volatility_is_rejected_as_degenerate(self, short_universe):
        mu, cov = short_universe
        with pytest.raises(ConfigurationError, match="degenerate"):
            make_optimizer(mu, cov, OptimizationMode.MARKET_NEUTRAL).min_volatility()


# ------------------------------------------------------------ stress / fuzz


class TestStress:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize(
        "mode",
        [OptimizationMode.LONG_ONLY, OptimizationMode.LONG_SHORT, OptimizationMode.MARKET_NEUTRAL],
    )
    def test_random_universes_satisfy_invariants(self, seed, mode):
        rng = np.random.default_rng(seed)
        n = 8
        mu = rng.normal(0.06, 0.08, size=n)
        factors = rng.normal(size=(n, 3))
        cov = factors @ factors.T * 0.01 + np.diag(rng.uniform(0.01, 0.09, size=n))
        opt = make_optimizer(list(mu), cov, mode, gross_limit=2.0)
        result = opt.max_sharpe()

        assert np.isfinite(result.weights).all()
        assert result.net_exposure == pytest.approx(opt.config.net_exposure_target, abs=1e-4)
        assert result.gross_exposure <= opt.config.effective_gross_limit + 1e-4
        assert (result.weights <= opt.config.max_weight + TOL).all()
        if mode is OptimizationMode.LONG_ONLY:
            assert (result.weights >= -TOL).all()
            equal_weight = np.full(n, 1.0 / n)
            assert result.sharpe >= sharpe_of(equal_weight, mu, cov) - 1e-6
        assert result.volatility > 0

    @pytest.mark.parametrize("mode", [OptimizationMode.LONG_ONLY, OptimizationMode.LONG_SHORT])
    def test_frontier_brackets_the_max_sharpe_portfolio(self, short_universe, mode):
        mu, cov = short_universe
        opt = make_optimizer(mu, cov, mode, gross_limit=2.0)
        frontier = opt.efficient_frontier(points=15)
        assert len(frontier) >= 2
        assert all(p.volatility > 0 for p in frontier)
        best = opt.max_sharpe()
        returns = [p.expected_return for p in frontier]
        assert min(returns) <= best.expected_return <= max(returns) + 1e-4

    @pytest.mark.parametrize(
        "mode",
        [OptimizationMode.LONG_ONLY, OptimizationMode.LONG_SHORT, OptimizationMode.MARKET_NEUTRAL],
    )
    def test_frontier_is_not_jagged(self, short_universe, mode):
        # A mean-variance frontier is convex: sorted by return, volatility may
        # descend to the minimum-variance point and then ascend — one sign
        # change. A sawtooth here means individual points converged to
        # suboptimal solutions (the bug the continuation warm start fixes).
        mu, cov = short_universe
        opt = make_optimizer(mu, cov, mode, gross_limit=2.0)
        frontier = sorted(opt.efficient_frontier(points=20), key=lambda p: p.expected_return)
        vols = np.array([p.volatility for p in frontier])
        diffs = np.diff(vols)
        signs = np.sign(diffs[np.abs(diffs) > 1e-6])
        sign_changes = int(np.sum(signs[1:] != signs[:-1]))
        assert sign_changes <= 1, f"jagged frontier: {vols.round(4).tolist()}"

    def test_rank_deficient_but_psd_covariance_still_solves(self):
        # Two perfectly correlated assets plus one independent: singular Σ.
        base = np.array([[0.04, 0.04, 0.0], [0.04, 0.04, 0.0], [0.0, 0.0, 0.02]])
        result = make_optimizer([0.08, 0.08, 0.05], base, OptimizationMode.LONG_ONLY).max_sharpe()
        assert result.net_exposure == pytest.approx(1.0, abs=TOL)


# ----------------------------------------------------------- degenerate input


class TestValidation:
    def test_rejects_nan_returns(self):
        with pytest.raises(DataValidationError, match="non-finite"):
            make_optimizer([0.05, np.nan], np.diag([0.02, 0.02]), OptimizationMode.LONG_ONLY)

    def test_rejects_asymmetric_covariance(self):
        cov = np.array([[0.04, 0.01], [0.03, 0.04]])
        with pytest.raises(DataValidationError, match="symmetric"):
            make_optimizer([0.05, 0.06], cov, OptimizationMode.LONG_ONLY)

    def test_rejects_indefinite_covariance(self):
        cov = np.array([[0.04, 0.09], [0.09, 0.04]])  # off-diagonal exceeds diagonal
        with pytest.raises(DataValidationError, match="positive semi-definite"):
            make_optimizer([0.05, 0.06], cov, OptimizationMode.LONG_ONLY)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(DataValidationError, match="shape"):
            make_optimizer([0.05, 0.06], np.diag([0.02, 0.02, 0.02]), OptimizationMode.LONG_ONLY)

    def test_rejects_empty_universe(self):
        with pytest.raises(DataValidationError, match="no assets"):
            MeanVarianceOptimizer([], np.array([]), np.empty((0, 0)), OptimizerConfig())

    def test_rejects_zero_volatility_optimum(self):
        # Defense-in-depth: a rank-deficient universe where the only feasible
        # portfolio (sum w = 1 with tight caps) has zero variance.
        cov = np.array([[0.04, -0.04], [-0.04, 0.04]])  # perfectly anti-correlated
        with pytest.raises(OptimizationError, match="zero volatility"):
            make_optimizer(
                [0.05, 0.05], cov, OptimizationMode.LONG_ONLY, max_weight=0.5
            ).max_sharpe()

    def test_rejects_infeasible_max_weight(self):
        with pytest.raises(ConfigurationError, match="infeasible"):
            make_optimizer(
                [0.05, 0.06], np.diag([0.02, 0.02]), OptimizationMode.LONG_ONLY, max_weight=0.3
            )

    def test_rejects_all_zero_variance_universe(self):
        with pytest.raises(DataValidationError, match="no positive variance"):
            make_optimizer([0.05], [[0.0]], OptimizationMode.LONG_ONLY)

    def test_config_rejects_gross_limit_below_one_for_long_short(self):
        with pytest.raises(ConfigurationError, match="gross_limit"):
            OptimizerConfig(mode=OptimizationMode.LONG_SHORT, gross_limit=0.8)

    def test_config_rejects_market_neutral_without_shorts(self):
        with pytest.raises(ConfigurationError, match="max_short"):
            OptimizerConfig(mode=OptimizationMode.MARKET_NEUTRAL, max_short=0.0)


# -------------------------------------------------------------------- kelly


class TestKelly:
    def test_matches_hand_computed_leverage(self):
        # kelly = (0.10 - 0.06) / 0.2² = 1.0 → no borrowing, no idle cash.
        k = kelly_metrics(0.10, 0.20, margin_cost_rate=0.06, risk_free_rate=RF)
        assert k.kelly_fraction == pytest.approx(1.0)
        assert k.safe_kelly == pytest.approx(1.0)
        assert k.leveraged_return == pytest.approx(0.10)

    def test_borrowing_is_charged_only_above_one_x(self):
        # kelly = (0.12 - 0.04) / 0.2² = 2.0 → borrow 1x at 4%.
        k = kelly_metrics(0.12, 0.20, margin_cost_rate=0.04, risk_free_rate=RF)
        assert k.safe_kelly == pytest.approx(2.0)
        assert k.leveraged_return == pytest.approx(2 * 0.12 - 1 * 0.04)

    def test_negative_kelly_parks_capital_at_risk_free(self):
        # Expected return below margin cost: do not lever; idle capital earns rf.
        k = kelly_metrics(0.03, 0.30, margin_cost_rate=0.06, risk_free_rate=RF)
        assert k.kelly_fraction < 0
        assert k.safe_kelly == 0.0
        assert k.leveraged_return == pytest.approx(RF)

    def test_cap_clamps_runaway_leverage(self):
        k = kelly_metrics(0.50, 0.10, margin_cost_rate=0.02, risk_free_rate=RF, leverage_cap=2.0)
        assert k.kelly_fraction > 2.0
        assert k.safe_kelly == 2.0

    def test_rejects_zero_volatility(self):
        with pytest.raises(DataValidationError):
            kelly_metrics(0.10, 0.0, margin_cost_rate=0.06, risk_free_rate=RF)
