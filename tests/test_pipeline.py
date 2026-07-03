"""Tests for the data pipeline (PortfolioAnalyzer) with injected data — no network."""

import numpy as np
import pandas as pd
import pytest
from src.core.exceptions import DataValidationError
from src.core.ModernPortfolio import PortfolioAnalyzer
from src.core.optimization import OptimizationMode, OptimizerConfig


def make_analyzer(tmp_path, tickers, price_data, div_data=None, **kwargs):
    analyzer = PortfolioAnalyzer(tickers, cache_dir=str(tmp_path / "cache"), **kwargs)
    analyzer.price_data = price_data
    analyzer.div_data = div_data if div_data is not None else pd.DataFrame()
    return analyzer


@pytest.fixture
def price_history():
    """Two years of daily prices: A trends up, B trends down, C is flat-ish."""
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=504)
    rng = np.random.default_rng(7)
    steps = rng.normal(0, 0.01, size=(504, 3)) + np.array([0.0008, -0.0006, 0.0001])
    prices = 100 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=index, columns=["AAA", "BBB", "CCC"])


class TestReturns:
    def test_summary_has_finite_annualized_stats(self, tmp_path, price_history):
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CCC"], price_history)
        summary = analyzer.calculate_returns()
        assert list(summary.index) == ["AAA", "BBB", "CCC"]
        assert np.isfinite(summary.to_numpy()).all()
        assert summary.loc["AAA", "AnnReturn"] > summary.loc["BBB", "AnnReturn"]

    def test_dividends_add_to_total_return(self, tmp_path, price_history):
        no_div = make_analyzer(tmp_path, ["AAA"], price_history[["AAA"]])
        base = no_div.calculate_returns().loc["AAA", "AnnReturn"]

        divs = pd.DataFrame(0.0, index=price_history.index, columns=["AAA"])
        divs.iloc[::63, 0] = 1.0  # quarterly dividend
        with_div = make_analyzer(tmp_path, ["AAA"], price_history[["AAA"]], divs)
        boosted = with_div.calculate_returns().loc["AAA", "AnnReturn"]
        assert boosted > base

    def test_synthetic_assets_are_deterministic_and_decorrelated(self, tmp_path, price_history):
        analyzer = make_analyzer(
            tmp_path, ["AAA", "CASH"], price_history[["AAA"]], risk_free_rate=0.04
        )
        analyzer.calculate_returns()
        # Arithmetic annualization of the geometric daily rate: 252·((1+rf)^(1/252)−1).
        expected = 252 * ((1.04) ** (1 / 252) - 1)
        assert analyzer.mean_returns["CASH"] == pytest.approx(expected, rel=1e-6)
        assert analyzer.cov_matrix.loc["CASH", "AAA"] == 0.0
        assert analyzer.cov_matrix.loc["CASH", "CASH"] > 0.0

    def test_future_dated_rows_are_dropped(self, tmp_path, price_history):
        future = price_history.copy()
        future.index = future.index + pd.Timedelta(days=365)  # half the rows in the future
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CCC"], future)
        analyzer.calculate_returns()
        assert analyzer.total_returns.index.max() <= pd.Timestamp.now()


class TestQuality:
    def test_drops_ticker_with_excessive_gaps(self, tmp_path, price_history):
        holey = price_history.copy()
        holey.iloc[: int(len(holey) * 0.6), 1] = np.nan  # BBB 60% missing
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CCC"], holey)
        summary = analyzer.calculate_returns()
        assert "BBB" not in summary.index
        assert {"AAA", "CCC"} <= set(summary.index)

    def test_salvages_common_window_when_all_gappy(self, tmp_path, price_history):
        # Every ticker misses >45% overall, but all overlap on rows 200:400 —
        # the pipeline must recover that window instead of dropping everything.
        windowed = price_history.copy()
        windowed.iloc[:200] = np.nan
        windowed.iloc[400:] = np.nan
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CCC"], windowed)
        summary = analyzer.calculate_returns()
        assert set(summary.index) == {"AAA", "BBB", "CCC"}

    def test_raises_when_no_common_window_exists(self, tmp_path, price_history):
        # AAA and BBB never overlap: salvage is impossible; silently optimizing
        # on garbage would be the bug.
        disjoint = price_history.copy()
        disjoint.iloc[:350, 0] = np.nan
        disjoint.iloc[150:, 1] = np.nan
        disjoint.iloc[:450, 2] = np.nan
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CCC"], disjoint)
        with pytest.raises(DataValidationError):
            analyzer.calculate_returns()

    def test_raises_without_fetch(self, tmp_path):
        analyzer = PortfolioAnalyzer(["AAA"], cache_dir=str(tmp_path))
        with pytest.raises(DataValidationError, match="fetch_data"):
            analyzer.calculate_returns()

    def test_rejects_empty_ticker_list(self, tmp_path):
        with pytest.raises(DataValidationError, match="empty"):
            PortfolioAnalyzer([], cache_dir=str(tmp_path))

    def test_deduplicates_tickers(self, tmp_path):
        analyzer = PortfolioAnalyzer(["AAA", "AAA", "BBB"], cache_dir=str(tmp_path))
        assert analyzer.tickers == ["AAA", "BBB"]


class TestEndToEndOptimization:
    @pytest.mark.parametrize(
        "mode",
        [OptimizationMode.LONG_ONLY, OptimizationMode.LONG_SHORT, OptimizationMode.MARKET_NEUTRAL],
    )
    def test_pipeline_feeds_all_three_modes(self, tmp_path, price_history, mode):
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CCC"], price_history)
        analyzer.calculate_returns()
        config = OptimizerConfig(mode=mode, max_short=1.0, gross_limit=2.0)
        result = analyzer.build_optimizer(config).max_sharpe()
        target = 0.0 if mode is OptimizationMode.MARKET_NEUTRAL else 1.0
        assert result.net_exposure == pytest.approx(target, abs=1e-4)
        if mode is not OptimizationMode.LONG_ONLY:
            # BBB trends down: a strictly negative weight, or shorting is broken.
            assert result.weights_by_ticker()["BBB"] < -0.01

    def test_exclude_cash_removes_synthetics(self, tmp_path, price_history):
        analyzer = make_analyzer(tmp_path, ["AAA", "BBB", "CASH"], price_history[["AAA", "BBB"]])
        analyzer.calculate_returns()
        optimizer = analyzer.build_optimizer(OptimizerConfig(), exclude_cash=True)
        assert "CASH" not in optimizer.tickers
