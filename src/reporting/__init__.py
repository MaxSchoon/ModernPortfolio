"""Reporting helpers for charts and self-contained HTML output."""

from src.reporting.html_report import write_html_report
from src.reporting.plots import (
    plot_allocation,
    plot_efficient_frontier,
    plot_price_history,
    plot_risk_return,
)

__all__ = [
    "plot_allocation",
    "plot_efficient_frontier",
    "plot_price_history",
    "plot_risk_return",
    "write_html_report",
]
