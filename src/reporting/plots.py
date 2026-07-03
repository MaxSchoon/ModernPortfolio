"""Matplotlib chart generation for portfolio reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.core.exceptions import DataValidationError

matplotlib.use("Agg")

if TYPE_CHECKING:
    from src.core.optimization import FrontierPoint, PortfolioResult

logger = logging.getLogger(__name__)

_MIN_VISIBLE_WEIGHT = 0.001
_LONG_COLOR = "#2563eb"
_SHORT_COLOR = "#dc2626"
_ASSET_COLOR = "#64748b"
_FRONTIER_COLOR = "#0f766e"
_OPTIMAL_COLOR = "#f59e0b"


def plot_allocation(weights: dict[str, float], mode: str, path: Path | str) -> None:
    """Write a signed horizontal allocation bar chart."""
    rows = _valid_weight_rows(weights)
    visible = [(ticker, weight) for ticker, weight in rows if abs(weight) >= _MIN_VISIBLE_WEIGHT]
    if not visible:
        raise DataValidationError("allocation has no weights above the 0.1% display threshold")

    visible.sort(key=lambda item: item[1])
    labels = [ticker for ticker, _ in visible]
    values = np.array([weight for _, weight in visible], dtype=float) * 100.0
    colors = [_LONG_COLOR if value >= 0 else _SHORT_COLOR for value in values]

    fig: Figure | None = None
    try:
        fig = Figure(figsize=(9, max(3.5, 0.45 * len(values) + 1.5)))
        ax = fig.subplots()
        y_positions = np.arange(len(values))
        ax.barh(y_positions, values, color=colors, alpha=0.9)
        ax.axvline(0, color="#111827", linewidth=1.0, alpha=0.75)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Portfolio weight")
        ax.set_title(f"Allocation ({mode})")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        _annotate_bars(ax, y_positions, values)
        # Leave headroom on both sides so the outside-the-bar value labels
        # are never clipped at the figure edge.
        headroom = 0.20 * float(np.max(np.abs(values)))
        ax.set_xlim(
            min(0.0, float(values.min())) - headroom,
            max(0.0, float(values.max())) + headroom,
        )
        _style_axes(ax)
        fig.tight_layout()
        _save_figure(fig, path)
        logger.info("wrote allocation chart to %s", path)
    finally:
        if fig is not None:
            fig.clear()


def plot_efficient_frontier(
    frontier: list[FrontierPoint],
    optimal: PortfolioResult,
    asset_returns: pd.Series,
    asset_volatilities: pd.Series,
    path: Path | str,
) -> None:
    """Write an efficient frontier chart with optimal and per-asset markers."""
    frontier_points = _valid_frontier(frontier)
    returns, volatilities = _valid_asset_series(asset_returns, asset_volatilities)
    optimal_return = _valid_finite_number(optimal.expected_return, "optimal expected return")
    optimal_volatility = _valid_positive_number(optimal.volatility, "optimal volatility")

    # Sort by return, not volatility: the frontier is a function of return
    # (one volatility per target return), while sorting by volatility
    # interleaves the upper and lower branches and draws a sawtooth.
    frontier_points.sort(key=lambda point: point[0])
    frontier_returns = np.array([point[0] for point in frontier_points], dtype=float) * 100.0
    frontier_vols = np.array([point[1] for point in frontier_points], dtype=float) * 100.0

    fig: Figure | None = None
    try:
        fig = Figure(figsize=(9, 5.5))
        ax = fig.subplots()
        ax.plot(frontier_vols, frontier_returns, color=_FRONTIER_COLOR, linewidth=2.2)
        ax.scatter(
            volatilities.to_numpy(dtype=float) * 100.0,
            returns.to_numpy(dtype=float) * 100.0,
            color=_ASSET_COLOR,
            s=48,
            alpha=0.85,
            label="Assets",
        )
        for ticker, ann_return, ann_volatility in zip(
            returns.index,
            returns.to_numpy(dtype=float),
            volatilities.to_numpy(dtype=float),
            strict=True,
        ):
            _annotate_point(ax, ticker, ann_volatility * 100.0, ann_return * 100.0)
        ax.scatter(
            optimal_volatility * 100.0,
            optimal_return * 100.0,
            marker="*",
            s=220,
            color=_OPTIMAL_COLOR,
            edgecolor="#92400e",
            linewidth=0.9,
            label="Optimal portfolio",
            zorder=4,
        )
        ax.set_xlabel("Annualized volatility")
        ax.set_ylabel("Expected annual return")
        ax.set_title("Efficient Frontier")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        _save_figure(fig, path)
        logger.info("wrote efficient frontier chart to %s", path)
    finally:
        if fig is not None:
            fig.clear()


def plot_risk_return(
    returns_summary: pd.DataFrame,
    risk_free_rate: float,
    path: Path | str,
) -> None:
    """Write an asset risk/return scatter chart with a risk-free reference line."""
    summary = _valid_returns_summary(returns_summary)
    risk_free_pct = _valid_finite_number(risk_free_rate, "risk-free rate") * 100.0

    fig: Figure | None = None
    try:
        fig = Figure(figsize=(9, 5.5))
        ax = fig.subplots()
        scatter = ax.scatter(
            summary["AnnVolatility"],
            summary["AnnReturn"],
            c=summary["Sharpe"],
            cmap="viridis",
            s=58,
            alpha=0.9,
            edgecolor="#0f172a",
            linewidth=0.35,
        )
        for ticker, row in summary.iterrows():
            _annotate_point(ax, ticker, float(row["AnnVolatility"]), float(row["AnnReturn"]))
        ax.axhline(
            risk_free_pct,
            color="#b45309",
            linewidth=1.2,
            linestyle="--",
            label=f"Risk-free rate ({risk_free_pct:.2f}%)",
        )
        ax.set_xlabel("Annualized volatility")
        ax.set_ylabel("Annualized return")
        ax.set_title("Risk and Return by Asset")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        fig.colorbar(scatter, ax=ax, label="Sharpe")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        _save_figure(fig, path)
        logger.info("wrote risk/return chart to %s", path)
    finally:
        if fig is not None:
            fig.clear()


def plot_price_history(ticker: str, prices: pd.Series, path: Path | str) -> None:
    """Write a simple price history line chart."""
    label = _safe_label(ticker) or "Asset"
    price_series = _valid_price_series(prices)

    fig: Figure | None = None
    try:
        fig = Figure(figsize=(9, 4.8))
        ax = fig.subplots()
        ax.plot(
            price_series.index, price_series.to_numpy(dtype=float), color=_LONG_COLOR, linewidth=1.8
        )
        ax.set_title(f"{label} Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        _style_axes(ax)
        fig.autofmt_xdate()
        fig.tight_layout()
        _save_figure(fig, path)
        logger.info("wrote price history chart to %s", path)
    finally:
        if fig is not None:
            fig.clear()


def _save_figure(fig: Figure, path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")


def _valid_weight_rows(weights: dict[str, float]) -> list[tuple[str, float]]:
    if not weights:
        raise DataValidationError("allocation weights are empty")
    rows: list[tuple[str, float]] = []
    for ticker, weight in weights.items():
        numeric_weight = _valid_finite_number(weight, "allocation weight")
        label = _safe_label(ticker)
        if label is None or not label.strip():
            logger.warning("skipping allocation weight with unusable ticker label")
            continue
        rows.append((label, numeric_weight))
    if not rows:
        raise DataValidationError("allocation weights contain no usable ticker labels")
    return rows


def _valid_frontier(frontier: list[FrontierPoint]) -> list[tuple[float, float]]:
    if not frontier:
        raise DataValidationError("efficient frontier is empty")
    points: list[tuple[float, float]] = []
    for point in frontier:
        expected_return = _valid_finite_number(point.expected_return, "frontier expected return")
        volatility = _valid_positive_number(point.volatility, "frontier volatility")
        points.append((expected_return, volatility))
    return points


def _valid_asset_series(
    asset_returns: pd.Series,
    asset_volatilities: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    returns = _valid_numeric_series(asset_returns, "asset returns")
    volatilities = _valid_numeric_series(asset_volatilities, "asset volatilities")
    if not returns.index.equals(volatilities.index):
        raise DataValidationError("asset returns and volatilities must have matching indexes")
    if (volatilities <= 0).any():
        raise DataValidationError("asset volatilities must be positive")
    return returns, volatilities


def _valid_returns_summary(returns_summary: pd.DataFrame) -> pd.DataFrame:
    if returns_summary.empty:
        raise DataValidationError("returns summary is empty")
    required = ["AnnReturn", "AnnVolatility", "Sharpe"]
    missing = [column for column in required if column not in returns_summary.columns]
    if missing:
        raise DataValidationError(f"returns summary is missing columns: {', '.join(missing)}")
    summary = returns_summary.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(summary.to_numpy(dtype=float)).all():
        raise DataValidationError("returns summary contains non-finite values")
    if (summary["AnnVolatility"] <= 0).any():
        raise DataValidationError("annualized volatility must be positive")
    return summary


def _valid_price_series(prices: pd.Series) -> pd.Series:
    price_series = _valid_numeric_series(prices, "price history")
    if (price_series <= 0).any():
        raise DataValidationError("price history must contain positive prices")
    return price_series


def _valid_numeric_series(series: pd.Series, name: str) -> pd.Series:
    if series.empty:
        raise DataValidationError(f"{name} is empty")
    numeric = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataValidationError(f"{name} contains non-finite values")
    return numeric


def _valid_finite_number(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be numeric") from exc
    if not np.isfinite(numeric):
        raise DataValidationError(f"{name} must be finite")
    return numeric


def _valid_positive_number(value: object, name: str) -> float:
    numeric = _valid_finite_number(value, name)
    if numeric <= 0:
        raise DataValidationError(f"{name} must be positive")
    return numeric


def _annotate_bars(ax: Axes, y_positions: np.ndarray, values: np.ndarray) -> None:
    max_width = float(np.max(np.abs(values)))
    offset = max(max_width * 0.015, 0.08)
    for y_position, value in zip(y_positions, values, strict=True):
        horizontal_alignment = "left" if value >= 0 else "right"
        x_position = value + offset if value >= 0 else value - offset
        ax.text(
            x_position,
            y_position,
            f"{value:.2f}%",
            va="center",
            ha=horizontal_alignment,
            fontsize=9,
            color="#111827",
        )


def _annotate_point(ax: Axes, label: object, x_value: float, y_value: float) -> None:
    text = _safe_label(label)
    if text is None:
        return
    ax.annotate(
        text,
        (x_value, y_value),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        color="#334155",
    )


def _safe_label(label: object) -> str | None:
    try:
        return str(label)
    except Exception:
        logger.warning("skipping label that could not be stringified", exc_info=True)
        return None


def _style_axes(ax: Axes) -> None:
    ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
