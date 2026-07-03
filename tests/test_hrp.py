"""Falsification suite for the Hierarchical Risk Parity allocator.

Ground truths: for uncorrelated assets HRP must reduce to exact
inverse-variance weighting; correlated assets must cluster and share one
risk budget; and — HRP's selling point — a singular covariance matrix must
still allocate (mean-variance methods cannot).
"""

import numpy as np
import pytest
from src.core.exceptions import DataValidationError
from src.core.hrp import HRPOptimizer
from src.core.optimization import OptimizationMode

RF = 0.02


def make_hrp(mu, cov):
    tickers = [f"A{i}" for i in range(len(mu))]
    return HRPOptimizer(tickers, np.array(mu, dtype=float), np.array(cov, dtype=float), RF)


class TestGroundTruth:
    def test_uncorrelated_assets_reduce_to_inverse_variance(self):
        # With a diagonal covariance every bisection split is inverse-variance,
        # so the final weights must equal 1/σ² normalized — exactly.
        variances = np.array([0.04, 0.02, 0.08, 0.01])
        result = make_hrp([0.05] * 4, np.diag(variances)).allocate()
        expected = (1 / variances) / (1 / variances).sum()
        np.testing.assert_allclose(np.sort(result.weights), np.sort(expected), atol=1e-10)
        # Not just the multiset: each asset gets its own inverse-variance weight.
        np.testing.assert_allclose(result.weights, expected, atol=1e-10)

    def test_two_assets_exact_split(self):
        cov = np.diag([0.09, 0.01])
        result = make_hrp([0.05, 0.05], cov).allocate()
        np.testing.assert_allclose(result.weights, [0.1, 0.9], atol=1e-10)

    def test_correlated_pair_shares_one_risk_budget(self):
        # A2 and A3 are near-duplicates; A1 is independent with equal variance.
        # The pair must cluster and split one budget: each member gets less
        # than the standalone asset.
        cov = np.array(
            [
                [0.04, 0.0, 0.0],
                [0.0, 0.04, 0.038],
                [0.0, 0.038, 0.04],
            ]
        )
        result = make_hrp([0.05] * 3, cov).allocate()
        standalone, dup_a, dup_b = result.weights
        assert standalone > dup_a
        assert standalone > dup_b
        assert dup_a + dup_b == pytest.approx(1.0 - standalone, abs=1e-9)


class TestInvariants:
    def test_long_only_fully_invested_deterministic(self):
        rng = np.random.default_rng(3)
        factors = rng.normal(size=(10, 3))
        cov = factors @ factors.T * 0.01 + np.diag(rng.uniform(0.01, 0.05, size=10))
        mu = rng.normal(0.06, 0.05, size=10)
        first = make_hrp(list(mu), cov).allocate()
        second = make_hrp(list(mu), cov).allocate()
        assert (first.weights >= 0).all()
        assert first.net_exposure == pytest.approx(1.0, abs=1e-9)
        assert first.gross_exposure == pytest.approx(1.0, abs=1e-9)
        assert first.mode is OptimizationMode.LONG_ONLY
        np.testing.assert_array_equal(first.weights, second.weights)

    def test_singular_covariance_still_allocates(self):
        # Two perfectly correlated assets: covariance is singular, so any
        # method needing Σ⁻¹ fails here. HRP must not.
        cov = np.array(
            [
                [0.04, 0.04, 0.0],
                [0.04, 0.04, 0.0],
                [0.0, 0.0, 0.02],
            ]
        )
        result = make_hrp([0.06, 0.06, 0.05], cov).allocate()
        assert result.net_exposure == pytest.approx(1.0, abs=1e-9)
        assert (result.weights >= 0).all()

    def test_weights_ignore_expected_returns(self):
        # HRP is risk-only: doubling every expected return must not move a
        # single weight (only the reported return/Sharpe change).
        cov = np.array([[0.04, 0.01], [0.01, 0.03]])
        low = make_hrp([0.02, 0.03], cov).allocate()
        high = make_hrp([0.04, 0.06], cov).allocate()
        np.testing.assert_array_equal(low.weights, high.weights)
        assert high.expected_return > low.expected_return

    def test_single_asset_gets_everything(self):
        result = make_hrp([0.07], [[0.04]]).allocate()
        assert result.weights[0] == pytest.approx(1.0)


class TestValidation:
    def test_rejects_zero_variance_asset(self):
        with pytest.raises(DataValidationError, match="positive variances"):
            make_hrp([0.05, 0.04], np.diag([0.04, 0.0]))

    def test_rejects_asymmetric_covariance(self):
        cov = np.array([[0.04, 0.02], [0.01, 0.04]])
        with pytest.raises(DataValidationError, match="symmetric"):
            make_hrp([0.05, 0.04], cov)

    def test_rejects_indefinite_covariance(self):
        # Off-diagonal exceeding the diagonal: not a valid covariance. Review
        # fuzzing showed such inputs made bisection shares leave [0, 1] and
        # emit negative weights on a portfolio labeled long-only.
        cov = np.array([[0.04, 0.09], [0.09, 0.04]])
        with pytest.raises(DataValidationError, match="positive semi-definite"):
            make_hrp([0.05, 0.06], cov)

    def test_fuzzed_indefinite_inputs_never_yield_signed_weights(self):
        # The reviewer's fuzz case: random symmetric-but-indefinite matrices
        # must be rejected at construction, never allocated with signs.
        rng = np.random.default_rng(0)
        rejected = 0
        for _ in range(200):
            raw = rng.uniform(-0.2, 0.2, size=(5, 5))
            cov = (raw + raw.T) / 2
            np.fill_diagonal(cov, rng.uniform(0.01, 0.05, size=5))
            if np.linalg.eigvalsh(cov)[0] >= -1e-8:
                continue
            with pytest.raises(DataValidationError):
                make_hrp(list(rng.normal(0.05, 0.02, 5)), cov)
            rejected += 1
        assert rejected > 0, "fuzz premise failed: no indefinite matrices generated"

    def test_rejects_nan_inputs(self):
        with pytest.raises(DataValidationError, match="non-finite"):
            make_hrp([0.05, np.nan], np.diag([0.04, 0.04]))

    def test_rejects_empty_universe(self):
        with pytest.raises(DataValidationError, match="no assets"):
            HRPOptimizer([], np.array([]), np.empty((0, 0)), RF)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(DataValidationError, match="shape"):
            make_hrp([0.05, 0.04], np.diag([0.04, 0.04, 0.04]))
