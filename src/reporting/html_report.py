"""Self-contained HTML report generation for optimized portfolios."""

from __future__ import annotations

import base64
import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError
from src.core.optimization import FrontierPoint, KellyMetrics, PortfolioResult

logger = logging.getLogger(__name__)

_MIN_TABLE_WEIGHT = 0.0001
_REQUIRED_SUMMARY_COLUMNS = ("AnnReturn", "AnnVolatility", "Sharpe")


def write_html_report(
    path: Path | str,
    *,
    result: PortfolioResult,
    kelly: KellyMetrics,
    returns_summary: pd.DataFrame,
    frontier: list[FrontierPoint],
    run_config: dict[str, Any],
    chart_paths: dict[str, Path] | None = None,
) -> Path:
    """Write a single-file HTML portfolio report and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _valid_returns_summary(returns_summary)
    chart_images = _embedded_charts(chart_paths or {})
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    mode = getattr(result.mode, "value", str(result.mode))

    html_text = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(mode)} Portfolio Report</title>",
            f"<style>{_stylesheet()}</style>",
            "</head>",
            "<body>",
            '<main class="report-shell">',
            _header(mode, generated_at),
            _metric_cards(result),
            _kelly_card(kelly),
            _allocation_table(result),
            _charts_section(chart_images),
            _frontier_note(frontier),
            _asset_stats_table(summary),
            _run_config_section(run_config),
            _footer(),
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    output_path.write_text(html_text, encoding="utf-8")
    logger.info("wrote HTML report to %s", output_path)
    return output_path


def _header(mode: str, generated_at: str) -> str:
    return f"""
<header class="report-header">
  <div>
    <p class="eyebrow">Modern Portfolio Report</p>
    <h1>Optimization Summary</h1>
  </div>
  <div class="header-meta">
    <span class="mode-badge">{_escape(mode)}</span>
    <span>{_escape(generated_at)}</span>
  </div>
</header>
"""


def _metric_cards(result: PortfolioResult) -> str:
    cards = [
        ("Expected Return", _format_pct(result.expected_return)),
        ("Volatility", _format_pct(result.volatility)),
        ("Sharpe", _format_ratio(result.sharpe)),
        ("Net Exposure", _format_pct(result.net_exposure)),
        ("Gross Exposure", _format_pct(result.gross_exposure)),
        ("Long Exposure", _format_pct(result.long_exposure)),
        ("Short Exposure", _format_pct(result.short_exposure)),
    ]
    return f"""
<section class="card-grid" aria-label="Portfolio metrics">
  {_metric_card_items(cards)}
</section>
"""


def _metric_card_items(cards: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"""
  <article class="metric-card">
    <span>{_escape(label)}</span>
    <strong>{_escape(value)}</strong>
  </article>
"""
        for label, value in cards
    )


def _kelly_card(kelly: KellyMetrics) -> str:
    cards = [
        ("Kelly Fraction", _format_leverage(kelly.kelly_fraction)),
        ("Safe Leverage", _format_leverage(kelly.safe_kelly)),
        ("Leveraged Return", _format_pct(kelly.leveraged_return)),
        ("Leveraged Volatility", _format_pct(kelly.leveraged_volatility)),
        ("Leveraged Sharpe", _format_ratio(kelly.leveraged_sharpe)),
    ]
    return f"""
<section class="section-panel">
  <div class="section-heading">
    <h2>Kelly Metrics</h2>
  </div>
  <div class="card-grid compact">
    {_metric_card_items(cards)}
  </div>
</section>
"""


def _allocation_table(result: PortfolioResult) -> str:
    rows = sorted(
        (
            (ticker, float(weight))
            for ticker, weight in result.weights_by_ticker().items()
            if abs(float(weight)) >= _MIN_TABLE_WEIGHT
        ),
        key=lambda row: abs(row[1]),
        reverse=True,
    )
    if not rows:
        body = '<tr><td colspan="3">No allocation above 0.01%.</td></tr>'
    else:
        body = "\n".join(_allocation_row(ticker, weight) for ticker, weight in rows)
    return f"""
<section class="section-panel">
  <div class="section-heading">
    <h2>Allocation</h2>
  </div>
  <div class="table-scroll">
    <table>
      <thead>
        <tr><th>Ticker</th><th>Side</th><th>Weight</th></tr>
      </thead>
      <tbody>
        {body}
      </tbody>
    </table>
  </div>
</section>
"""


def _allocation_row(ticker: str, weight: float) -> str:
    side = "Long" if weight >= 0 else "Short"
    css_class = "long" if weight >= 0 else "short"
    rendered_weight = f"{weight * 100:.2f}%"
    return f"""
<tr>
  <td>{_escape(ticker)}</td>
  <td><span class="side {css_class}">{_escape(side)}</span></td>
  <td class="number {css_class}">{_escape(rendered_weight)}</td>
</tr>
"""


def _charts_section(chart_images: list[tuple[str, str]]) -> str:
    if not chart_images:
        return ""
    charts = "\n".join(
        f"""
    <figure>
      <img src="{data_uri}" alt="{_escape(name)} chart">
      <figcaption>{_escape(name)}</figcaption>
    </figure>
"""
        for name, data_uri in chart_images
    )
    return f"""
<section class="section-panel">
  <div class="section-heading">
    <h2>Charts</h2>
  </div>
  <div class="chart-grid">
    {charts}
  </div>
</section>
"""


def _asset_stats_table(summary: pd.DataFrame) -> str:
    rows = "\n".join(
        f"""
<tr>
  <td>{_escape(ticker)}</td>
  <td class="number">{_escape(_format_percent_value(row["AnnReturn"]))}</td>
  <td class="number">{_escape(_format_percent_value(row["AnnVolatility"]))}</td>
  <td class="number">{_escape(_format_ratio(row["Sharpe"]))}</td>
</tr>
"""
        for ticker, row in summary.iterrows()
    )
    return f"""
<section class="section-panel">
  <div class="section-heading">
    <h2>Per-Asset Statistics</h2>
  </div>
  <div class="table-scroll">
    <table>
      <thead>
        <tr><th>Ticker</th><th>AnnReturn</th><th>AnnVolatility</th><th>Sharpe</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</section>
"""


def _run_config_section(run_config: dict[str, Any]) -> str:
    if not run_config:
        return ""
    rows = "\n".join(
        f"<tr><td>{_escape(key)}</td><td>{_escape(_format_config_value(value))}</td></tr>"
        for key, value in sorted(run_config.items(), key=lambda item: str(item[0]))
    )
    return f"""
<section class="section-panel">
  <div class="section-heading">
    <h2>Run Configuration</h2>
  </div>
  <div class="table-scroll">
    <table>
      <thead><tr><th>Setting</th><th>Value</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>
"""


def _frontier_note(frontier: list[FrontierPoint]) -> str:
    point_count = len(frontier)
    return f'<p class="frontier-note">Efficient frontier points: {_escape(point_count)}</p>'


def _footer() -> str:
    return """
<footer>
  experimental / educational, not investment advice
</footer>
"""


def _embedded_charts(chart_paths: dict[str, Path]) -> list[tuple[str, str]]:
    charts: list[tuple[str, str]] = []
    for name, path in chart_paths.items():
        chart_path = Path(path)
        if not chart_path.exists() or not chart_path.is_file():
            logger.warning("skipping missing chart path: %s", chart_path)
            continue
        try:
            payload = base64.b64encode(chart_path.read_bytes()).decode("ascii")
        except OSError as exc:
            logger.warning("skipping unreadable chart path %s: %s", chart_path, exc)
            continue
        mime_type = "image/svg+xml" if chart_path.suffix.lower() == ".svg" else "image/png"
        charts.append((str(name), f"data:{mime_type};base64,{payload}"))
    return charts


def _valid_returns_summary(returns_summary: pd.DataFrame) -> pd.DataFrame:
    if returns_summary.empty:
        raise DataValidationError("returns summary is empty")
    missing = [column for column in _REQUIRED_SUMMARY_COLUMNS if column not in returns_summary]
    if missing:
        raise DataValidationError(f"returns summary is missing columns: {', '.join(missing)}")
    summary = returns_summary.loc[:, list(_REQUIRED_SUMMARY_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(summary.to_numpy(dtype=float)).all():
        raise DataValidationError("returns summary contains non-finite values")
    return summary


def _format_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _format_percent_value(value: float) -> str:
    return f"{float(value):.2f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.2f}"


def _format_leverage(value: float) -> str:
    return f"{float(value):.2f}x"


def _format_config_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, bool)):
        return str(value)
    return str(value)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _stylesheet() -> str:
    return """
:root {
  color-scheme: light dark;
  --bg: #f8fafc;
  --panel: #ffffff;
  --panel-muted: #f1f5f9;
  --text: #0f172a;
  --muted: #64748b;
  --line: #cbd5e1;
  --accent: #0f766e;
  --long: #2563eb;
  --short: #dc2626;
  --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b1120;
    --panel: #111827;
    --panel-muted: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --line: #334155;
    --accent: #2dd4bf;
    --long: #60a5fa;
    --short: #f87171;
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
  }
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}

.report-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 40px 0;
}

.report-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 28px;
}

.eyebrow,
.header-meta,
footer,
.frontier-note {
  color: var(--muted);
  font-size: 0.9rem;
}

.eyebrow {
  margin: 0 0 6px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1,
h2 {
  margin: 0;
  letter-spacing: 0;
}

h1 {
  font-size: clamp(2rem, 5vw, 3.4rem);
  line-height: 1.05;
}

h2 {
  font-size: 1.05rem;
}

.header-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
}

.mode-badge,
.side {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  font-weight: 700;
  white-space: nowrap;
}

.mode-badge {
  padding: 8px 12px;
  background: color-mix(in srgb, var(--accent) 16%, transparent);
  color: var(--accent);
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}

.card-grid.compact {
  margin-bottom: 0;
}

.metric-card,
.section-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.metric-card {
  padding: 16px;
}

.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 0.84rem;
}

.metric-card strong {
  display: block;
  margin-top: 6px;
  font-size: 1.35rem;
}

.section-panel {
  margin-top: 18px;
  padding: 20px;
}

.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.table-scroll {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 560px;
}

th,
td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
}

th {
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.number {
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.side {
  padding: 3px 9px;
  font-size: 0.82rem;
}

.long {
  color: var(--long);
}

.side.long {
  background: color-mix(in srgb, var(--long) 14%, transparent);
}

.short {
  color: var(--short);
}

.side.short {
  background: color-mix(in srgb, var(--short) 14%, transparent);
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}

figure {
  margin: 0;
  padding: 12px;
  background: var(--panel-muted);
  border: 1px solid var(--line);
  border-radius: 8px;
}

img {
  display: block;
  width: 100%;
  height: auto;
}

figcaption {
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.88rem;
}

.frontier-note {
  margin: 16px 0 0;
}

footer {
  margin-top: 28px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
}

@media (max-width: 720px) {
  .report-shell {
    width: min(100% - 20px, 1180px);
    padding: 24px 0;
  }

  .report-header {
    display: block;
  }

  .header-meta {
    justify-content: flex-start;
    margin-top: 14px;
  }

  .section-panel {
    padding: 14px;
  }
}
"""
