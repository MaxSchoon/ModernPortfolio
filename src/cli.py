"""Command-line interface for the Modern Portfolio optimizer.

This is the only module that prints, colors, and maps exceptions to exit
codes. Run as ``python -m src.cli`` from the repo root.

Exit codes: 0 success, 1 portfolio/data error, 2 usage error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

from src.core.exceptions import PortfolioError
from src.core.ModernPortfolio import SYNTHETIC_TICKERS, PortfolioAnalyzer
from src.core.optimization import (
    FrontierPoint,
    KellyMetrics,
    OptimizationMode,
    OptimizerConfig,
    PortfolioResult,
    kelly_metrics,
)
from src.utils.utils import load_tickers

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


# ------------------------------------------------------------------ styling


class Style:
    """ANSI styling with graceful fallback when not writing to a terminal."""

    def __init__(self, enabled: bool) -> None:
        def code(value: str) -> str:
            return value if enabled else ""

        self.reset = code("\033[0m")
        self.bold = code("\033[1m")
        self.dim = code("\033[2m")
        self.green = code("\033[32m")
        self.red = code("\033[31m")
        self.cyan = code("\033[36m")
        self.yellow = code("\033[33m")

    def heading(self, text: str) -> str:
        return (
            f"\n{self.bold}{self.cyan}{text}{self.reset}\n{self.dim}{'─' * len(text)}{self.reset}"
        )

    def weight(self, value: float) -> str:
        color = self.green if value >= 0 else self.red
        return f"{color}{value * 100:+8.2f}%{self.reset}"


# ---------------------------------------------------------------- arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modern-portfolio",
        description=(
            "Optimize a portfolio with Modern Portfolio Theory: long-only, "
            "long-short, or market-neutral. Experimental and educational — "
            "not investment advice."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--tickers-file",
        type=Path,
        default=Path("src/data/tickers.csv"),
        help="CSV file with a 'ticker' column",
    )
    data.add_argument("--years", type=int, default=5, help="years of history to use")
    data.add_argument("--no-cache", action="store_true", help="bypass the local data cache")
    data.add_argument("--clear-cache", action="store_true", help="clear the cache before running")
    data.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data_cache"),
        help="directory for cached market data",
    )
    data.add_argument("--batch-size", type=int, default=50, help="tickers per fetch batch")
    data.add_argument("--workers", type=int, default=3, help="parallel fetch workers")
    data.add_argument(
        "--no-standardize", action="store_true", help="skip the cache date-standardization pass"
    )

    strategy = parser.add_argument_group("strategy")
    strategy.add_argument(
        "--mode",
        type=OptimizationMode,
        choices=list(OptimizationMode),
        metavar="{long-only,long-short,market-neutral}",
        default=OptimizationMode.LONG_ONLY,
        help="portfolio regime: long-only, long-short (shorts allowed with "
        "full use of proceeds), or market-neutral (net exposure 0, gross 1)",
    )
    strategy.add_argument(
        "--risk-free", type=float, default=0.04, metavar="RATE", help="annual risk-free rate"
    )
    strategy.add_argument(
        "--margin-cost",
        type=float,
        default=0.065,
        metavar="RATE",
        help="annual margin borrow rate (Kelly leverage)",
    )
    strategy.add_argument(
        "--max-weight", type=float, default=1.0, metavar="W", help="per-asset cap on long weight"
    )
    strategy.add_argument(
        "--max-short",
        type=float,
        default=0.3,
        metavar="W",
        help="per-asset cap on short weight (positive fraction)",
    )
    strategy.add_argument(
        "--gross-limit",
        type=float,
        default=1.6,
        metavar="G",
        help="cap on gross exposure sum(|w|) for long-short "
        "(1.6 = a 130/30 fund, 2.0 = Reg-T margin)",
    )
    strategy.add_argument(
        "--borrow-rate",
        type=float,
        default=0.0,
        metavar="RATE",
        help="annual borrow fee charged on short notional",
    )
    strategy.add_argument(
        "--exclude-cash",
        action="store_true",
        help="exclude synthetic CASH/TBILLS from optimization",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path("portfolio_analysis"),
        help="directory for results, charts, and the report",
    )
    output.add_argument("--skip-plots", action="store_true", help="skip chart generation")
    output.add_argument(
        "--no-html-report", action="store_true", help="skip the self-contained HTML report"
    )
    output.add_argument(
        "--frontier-points", type=int, default=30, help="resolution of the efficient frontier"
    )
    output.add_argument(
        "--price-charts", action="store_true", help="also write one price-history chart per ticker"
    )
    output.add_argument(
        "--fast", action="store_true", help="fastest run: implies --skip-plots and --no-html-report"
    )
    verbosity = output.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="errors only")
    verbosity.add_argument("--debug", action="store_true", help="verbose diagnostics")
    output.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return parser


def configure_logging(quiet: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    if not debug:
        # Chatty third-party loggers drown the useful pipeline messages.
        for noisy in ("yfinance", "peewee", "urllib3", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------- presentation


def print_result(
    result: PortfolioResult, kelly: KellyMetrics, style: Style, risk_free_rate: float
) -> None:
    print(style.heading(f"Optimal allocation — {result.mode.value}"))
    weights = sorted(result.weights_by_ticker().items(), key=lambda kv: -abs(kv[1]))
    for ticker, weight in weights:
        if abs(weight) < 1e-4:
            continue
        tag = f"{style.dim}(short){style.reset}" if weight < 0 else "       "
        print(f"  {ticker:<10} {style.weight(weight)}  {tag}")

    print(style.heading("Portfolio metrics"))
    rows = [
        ("Expected return", f"{result.expected_return * 100:.2f}%"),
        ("Volatility", f"{result.volatility * 100:.2f}%"),
        ("Sharpe ratio", f"{result.sharpe:.2f}"),
        ("Net exposure", f"{result.net_exposure * 100:.1f}%"),
        ("Gross exposure", f"{result.gross_exposure * 100:.1f}%"),
        (
            "Long / short",
            f"{result.long_exposure * 100:.1f}% / {result.short_exposure * 100:.1f}%",
        ),
        ("Risk-free rate", f"{risk_free_rate * 100:.2f}%"),
    ]
    for label, value in rows:
        print(f"  {label:<18} {style.bold}{value}{style.reset}")

    print(style.heading("Kelly leverage"))
    kelly_rows = [
        ("Kelly fraction", f"{kelly.kelly_fraction:.2f}x"),
        ("Suggested leverage", f"{kelly.safe_kelly:.2f}x"),
        ("Leveraged return", f"{kelly.leveraged_return * 100:.2f}%"),
        ("Leveraged volatility", f"{kelly.leveraged_volatility * 100:.2f}%"),
        ("Leveraged Sharpe", f"{kelly.leveraged_sharpe:.2f}"),
    ]
    for label, value in kelly_rows:
        print(f"  {label:<20} {value}")
    if kelly.kelly_fraction > kelly.safe_kelly:
        print(
            f"  {style.yellow}Kelly suggests {kelly.kelly_fraction:.1f}x; "
            f"capped at {kelly.safe_kelly:.1f}x for safety.{style.reset}"
        )


def save_outputs(
    output_dir: Path,
    result: PortfolioResult,
    analyzer: PortfolioAnalyzer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_pct = {t: w * 100 for t, w in result.weights_by_ticker().items()}
    (output_dir / "optimal_weights.json").write_text(json.dumps(weights_pct, indent=2))
    if analyzer.returns_summary is not None:
        analyzer.returns_summary.to_csv(output_dir / "returns_summary.csv")
    if analyzer.total_returns is not None:
        analyzer.total_returns.corr().to_csv(output_dir / "correlations.csv")
    logger.info("results saved to %s", output_dir)


def generate_charts(
    args: argparse.Namespace,
    result: PortfolioResult,
    frontier: list[FrontierPoint],
    analyzer: PortfolioAnalyzer,
) -> dict[str, Path]:
    """Write charts and return their paths for embedding in the HTML report."""
    from src.reporting import plots

    charts: dict[str, Path] = {}
    summary = analyzer.returns_summary
    out = args.output_dir

    allocation_path = out / "portfolio_allocation.png"
    plots.plot_allocation(result.weights_by_ticker(), result.mode.value, allocation_path)
    charts["Allocation"] = allocation_path

    if frontier:
        frontier_path = out / "efficient_frontier.png"
        universe = list(result.tickers)
        # plot_efficient_frontier expects raw fractions; the summary columns are in %.
        plots.plot_efficient_frontier(
            frontier,
            result,
            summary.loc[universe, "AnnReturn"] / 100.0,
            summary.loc[universe, "AnnVolatility"] / 100.0,
            frontier_path,
        )
        charts["Efficient frontier"] = frontier_path

    risk_return_path = out / "risk_return_profile.png"
    plots.plot_risk_return(summary, args.risk_free, risk_return_path)
    charts["Risk / return"] = risk_return_path

    if args.price_charts and analyzer.price_data is not None:
        charts_dir = out / "price_charts"
        for ticker in analyzer.tickers:
            if ticker in SYNTHETIC_TICKERS or ticker not in analyzer.price_data.columns:
                continue
            plots.plot_price_history(
                ticker, analyzer.price_data[ticker], charts_dir / f"{ticker}_price.png"
            )
    return charts


# ---------------------------------------------------------------------- run


@contextlib.contextmanager
def _legacy_output_silenced(quiet: bool):
    """Suppress raw print() chatter from legacy data modules in quiet mode.

    The fetch/cache layer still prints progress directly; until it is ported
    to logging, quiet mode redirects its stdout. Errors are unaffected: they
    surface as exceptions and logging on stderr.
    """
    if not quiet:
        yield
        return
    os.environ.setdefault("TQDM_DISABLE", "1")  # progress bars write to stderr
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def run(args: argparse.Namespace) -> int:
    style = Style(enabled=sys.stdout.isatty() and not args.no_color)

    if args.fast:
        args.skip_plots = True
        args.no_html_report = True

    # Validate strategy settings before any (slow, networked) data work.
    config = OptimizerConfig(
        mode=args.mode,
        risk_free_rate=args.risk_free,
        max_weight=args.max_weight,
        max_short=args.max_short,
        gross_limit=args.gross_limit,
        short_borrow_rate=args.borrow_rate,
    )

    if args.clear_cache:
        from src.cache.csv_cache_manager import CSVDataCache

        CSVDataCache(str(args.cache_dir)).clear_cache()
        logger.info("cache cleared")

    with _legacy_output_silenced(args.quiet):
        tickers = load_tickers(str(args.tickers_file))
        analyzer = PortfolioAnalyzer(
            tickers,
            risk_free_rate=args.risk_free,
            years=args.years,
            cache_dir=str(args.cache_dir),
        )
        analyzer.fetch_data(
            use_cache=not args.no_cache, batch_size=args.batch_size, max_workers=args.workers
        )

        if not args.no_standardize:
            standardize_cache_safely(analyzer, args)

    analyzer.calculate_returns()

    optimizer = analyzer.build_optimizer(config, exclude_cash=args.exclude_cash)
    result = optimizer.max_sharpe()
    kelly = kelly_metrics(
        result.expected_return,
        result.volatility,
        margin_cost_rate=args.margin_cost,
        risk_free_rate=args.risk_free,
    )

    frontier: list[FrontierPoint] = []
    if not args.skip_plots or not args.no_html_report:
        try:
            frontier = optimizer.efficient_frontier(points=args.frontier_points)
        except PortfolioError as exc:
            logger.warning("efficient frontier unavailable: %s", exc)

    save_outputs(args.output_dir, result, analyzer)

    charts: dict[str, Path] = {}
    if not args.skip_plots:
        charts = generate_charts(args, result, frontier, analyzer)

    if not args.no_html_report:
        from src.reporting.html_report import write_html_report

        report_path = write_html_report(
            args.output_dir / "report.html",
            result=result,
            kelly=kelly,
            returns_summary=analyzer.returns_summary,
            frontier=frontier,
            run_config={
                "Mode": args.mode.value,
                "Years of history": args.years,
                "Risk-free rate": f"{args.risk_free:.2%}",
                "Margin cost": f"{args.margin_cost:.2%}",
                "Short borrow rate": f"{args.borrow_rate:.2%}",
                "Gross limit": args.gross_limit,
                "Max weight": args.max_weight,
                "Max short": args.max_short,
                "Tickers file": str(args.tickers_file),
            },
            chart_paths=charts or None,
        )
        print(f"\n{style.dim}HTML report:{style.reset} {report_path}")

    if not args.quiet:
        print_result(result, kelly, style, args.risk_free)
        print(f"\n{style.dim}Outputs in {args.output_dir}{style.reset}")
    return EXIT_OK


def standardize_cache_safely(analyzer: PortfolioAnalyzer, args: argparse.Namespace) -> None:
    """Run the optional cache standardization pass; never fail the run for it."""
    try:
        from src.cache.cache_standardize import standardize_cache
    except ImportError:
        logger.debug("cache_standardize module not available; skipping")
        return
    try:
        stats = standardize_cache(str(args.cache_dir), remove_future=True)
        if stats:
            logger.info(
                "standardized %d price and %d dividend cache files",
                stats.get("price_processed", 0),
                stats.get("div_processed", 0),
            )
    except Exception:
        # Standardization is best-effort maintenance of cache file formats;
        # a failure here must not abort the analysis. Full context to the log.
        logger.exception("cache standardization failed; continuing with raw cache")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.quiet, args.debug)
    try:
        return run(args)
    except PortfolioError as exc:
        logger.debug("failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
