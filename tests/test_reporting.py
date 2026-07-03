from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.core.exceptions import DataValidationError
from src.core.optimization import (
    FrontierPoint,
    KellyMetrics,
    OptimizationMode,
    PortfolioResult,
)
from src.reporting.html_report import write_html_report
from src.reporting.plots import (
    plot_allocation,
    plot_efficient_frontier,
    plot_price_history,
    plot_risk_return,
)


@pytest.fixture
def portfolio_result() -> PortfolioResult:
    weights = np.array([0.70, 0.45, -0.15])
    return PortfolioResult(
        tickers=("AAA", "BBB", "CCC"),
        weights=weights,
        expected_return=0.112,
        volatility=0.184,
        sharpe=0.50,
        mode=OptimizationMode.LONG_SHORT,
        net_exposure=float(weights.sum()),
        gross_exposure=float(np.abs(weights).sum()),
    )


@pytest.fixture
def kelly_metrics() -> KellyMetrics:
    return KellyMetrics(
        kelly_fraction=1.35,
        safe_kelly=1.20,
        leveraged_return=0.126,
        leveraged_volatility=0.221,
        leveraged_sharpe=0.47,
    )


@pytest.fixture
def returns_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AnnReturn": [10.0, 7.5, -2.5],
            "AnnVolatility": [18.0, 14.0, 21.0],
            "Sharpe": [0.44, 0.39, -0.21],
        },
        index=["AAA", "BBB", "CCC"],
    )


@pytest.fixture
def frontier() -> list[FrontierPoint]:
    return [
        FrontierPoint(expected_return=0.04, volatility=0.10),
        FrontierPoint(expected_return=0.08, volatility=0.14),
        FrontierPoint(expected_return=0.12, volatility=0.20),
    ]


def assert_nonempty_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"\x89PNG")


def test_plot_allocation_writes_png_for_mixed_long_short(tmp_path: Path) -> None:
    path = tmp_path / "allocation-mixed.png"
    plot_allocation({"AAA": 0.70, "BBB": 0.45, "CCC": -0.15, "CASH": 0.0005}, "long-short", path)

    assert_nonempty_png(path)


def test_plot_allocation_writes_png_for_all_long(tmp_path: Path) -> None:
    path = tmp_path / "allocation-long.png"
    plot_allocation({"AAA": 0.55, "BBB": 0.30, "CCC": 0.15}, "long-only", path)

    assert_nonempty_png(path)


def test_plot_allocation_rejects_empty_weights(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        plot_allocation({}, "long-short", tmp_path / "empty.png")


def test_plot_efficient_frontier_writes_png(
    tmp_path: Path,
    frontier: list[FrontierPoint],
    portfolio_result: PortfolioResult,
) -> None:
    path = tmp_path / "frontier.png"
    asset_returns = pd.Series([0.10, 0.075, -0.025], index=["AAA", "BBB", "CCC"])
    asset_volatilities = pd.Series([0.18, 0.14, 0.21], index=["AAA", "BBB", "CCC"])

    plot_efficient_frontier(frontier, portfolio_result, asset_returns, asset_volatilities, path)

    assert_nonempty_png(path)


def test_plot_risk_return_writes_png(tmp_path: Path, returns_summary: pd.DataFrame) -> None:
    path = tmp_path / "risk-return.png"

    plot_risk_return(returns_summary, 0.02, path)

    assert_nonempty_png(path)


def test_plot_price_history_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "price-history.png"
    prices = pd.Series(
        [100.0, 101.5, 99.0, 104.0],
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )

    plot_price_history("AAA", prices, path)

    assert_nonempty_png(path)


def test_write_html_report_is_self_contained_and_renders_shorts(
    tmp_path: Path,
    portfolio_result: PortfolioResult,
    kelly_metrics: KellyMetrics,
    returns_summary: pd.DataFrame,
    frontier: list[FrontierPoint],
) -> None:
    chart_path = tmp_path / "allocation.png"
    plot_allocation(portfolio_result.weights_by_ticker(), "long-short", chart_path)
    report_path = write_html_report(
        tmp_path / "report.html",
        result=portfolio_result,
        kelly=kelly_metrics,
        returns_summary=returns_summary,
        frontier=frontier,
        run_config={"mode": "long-short", "risk_free_rate": 0.02},
        chart_paths={"Allocation": chart_path},
    )

    html = report_path.read_text(encoding="utf-8")
    assert "AAA" in html
    assert "BBB" in html
    assert "CCC" in html
    assert "Sharpe" in html
    assert "-15.00%" in html
    assert "data:image/png;base64," in html
    assert "<script" not in html.lower()
