"""Data pipeline and orchestration for the Modern Portfolio optimizer.

This module owns getting from a ticker list to annualized return/covariance
estimates. The numerical optimization lives in ``src.core.optimization``;
presentation (CLI output, charts, HTML report) lives in ``src.cli`` and
``src.reporting``.

Run the tool with ``python -m src.cli`` (see README).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.cache.csv_cache_manager import CSVDataCache
from src.core.exceptions import DataValidationError
from src.core.optimization import MeanVarianceOptimizer, OptimizerConfig
from src.fetching.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

#: Synthetic assets modeled as deterministic compounding at (a spread over)
#: the risk-free rate rather than fetched from Yahoo Finance.
SYNTHETIC_TICKERS = ("CASH", "TBILLS")
TRADING_DAYS_PER_YEAR = 252
#: Tickers missing more than this share of prices are dropped from the universe.
MAX_MISSING_PCT = 45.0
_TBILLS_SPREAD = 1.02  # TBILLS yield ≈ 2% over CASH, as in the original model


class PortfolioAnalyzer:
    """Loads price/dividend history and estimates the optimizer's inputs.

    Typical flow::

        analyzer = PortfolioAnalyzer(tickers, risk_free_rate=0.04, years=5)
        analyzer.fetch_data()
        analyzer.calculate_returns()
        result = analyzer.build_optimizer(config).max_sharpe()
    """

    def __init__(
        self,
        tickers: list[str],
        risk_free_rate: float = 0.04,
        years: int = 5,
        cache_dir: str = "data_cache",
    ) -> None:
        if not tickers:
            raise DataValidationError("ticker list is empty")
        if years < 1:
            raise DataValidationError(f"years must be >= 1, got {years}")
        # Deduplicate while preserving order; duplicated tickers would make the
        # covariance matrix singular by construction.
        self.tickers = list(dict.fromkeys(tickers))
        if len(self.tickers) < len(tickers):
            logger.warning("removed %d duplicate tickers", len(tickers) - len(self.tickers))
        self.risk_free_rate = risk_free_rate
        self.years = years
        self.cache_dir = cache_dir
        self.cache = CSVDataCache(cache_dir)
        self.price_data: pd.DataFrame | None = None
        self.div_data: pd.DataFrame | None = None
        self.mean_returns: pd.Series | None = None
        self.cov_matrix: pd.DataFrame | None = None
        self.returns_summary: pd.DataFrame | None = None
        self.total_returns: pd.DataFrame | None = None

    # ---------------------------------------------------------------- fetch

    def fetch_data(
        self,
        use_cache: bool = True,
        batch_size: int = 50,
        max_workers: int = 3,
    ) -> None:
        """Fetch (or load cached) price and dividend history for the universe.

        Raises:
            DataValidationError: if no ticker yields usable price data.
        """
        real_tickers = [t for t in self.tickers if t not in SYNTHETIC_TICKERS]
        started = time.time()
        logger.info("fetching %d tickers over %d years", len(real_tickers), self.years)

        results: dict[str, str] = {}
        if real_tickers:
            fetcher = DataFetcher(
                cache_dir=self.cache_dir,
                batch_size=batch_size,
                years=self.years,
                max_workers=max_workers,
            )
            results = fetcher.fetch_all(real_tickers, use_cache)

        prices: dict[str, pd.Series] = {}
        dividends: dict[str, pd.Series] = {}
        failed: list[str] = []
        for ticker in real_tickers:
            status = results.get(ticker, "")
            cached_price = self.cache.get_price_data(ticker) if status.startswith("✅") else None
            if cached_price is None or cached_price.dropna().empty:
                failed.append(ticker)
                logger.warning(
                    "no usable price data for %s (fetch status: %s)", ticker, status or "missing"
                )
                continue
            prices[ticker] = cached_price
            cached_div = self.cache.get_div_data(ticker)
            if cached_div is not None:
                dividends[ticker] = cached_div

        if not prices and not any(t in SYNTHETIC_TICKERS for t in self.tickers):
            raise DataValidationError(
                f"none of the {len(real_tickers)} tickers produced usable price data; "
                "check ticker symbols and network connectivity (see src/tests/test_yahoo.py)"
            )
        if failed:
            logger.warning("dropping %d failed tickers: %s", len(failed), ", ".join(failed))
            self.tickers = [t for t in self.tickers if t not in failed]

        self.price_data = pd.concat(prices, axis=1) if prices else pd.DataFrame()
        self.div_data = pd.concat(dividends, axis=1) if dividends else pd.DataFrame()
        logger.info("loaded %d tickers in %.1fs", len(prices), time.time() - started)

    # -------------------------------------------------------------- returns

    def calculate_returns(self) -> pd.DataFrame:
        """Estimate annualized mean returns and covariance from total returns.

        Total return = price return + dividend yield per period. Synthetic
        assets (CASH, TBILLS) get deterministic risk-free returns and are
        decorrelated from every other asset.

        Returns:
            Per-asset summary (AnnReturn %, AnnVolatility %, Sharpe).

        Raises:
            DataValidationError: if no aligned price data survives quality checks.
        """
        prices, divs = self._aligned_data()

        price_returns = prices.pct_change().iloc[1:]
        div_returns = (divs / prices.shift(1)).iloc[1:].fillna(0.0)
        total_returns = price_returns.add(div_returns)

        for ticker in (t for t in self.tickers if t in SYNTHETIC_TICKERS):
            rate = self.risk_free_rate * (_TBILLS_SPREAD if ticker == "TBILLS" else 1.0)
            total_returns[ticker] = (1.0 + rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

        self.total_returns = total_returns
        self.mean_returns = total_returns.mean() * TRADING_DAYS_PER_YEAR
        self.cov_matrix = total_returns.cov() * TRADING_DAYS_PER_YEAR
        self._decorrelate_synthetic_assets()

        vols = np.sqrt(np.diag(self.cov_matrix))
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(
                vols > 0, (self.mean_returns.values - self.risk_free_rate) / vols, np.nan
            )
        self.returns_summary = pd.DataFrame(
            {
                "AnnReturn": self.mean_returns.values * 100,
                "AnnVolatility": vols * 100,
                "Sharpe": sharpe,
            },
            index=self.mean_returns.index,
        )
        return self.returns_summary

    def _aligned_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align price/dividend history onto one index and enforce data quality."""
        if self.price_data is None:
            raise DataValidationError("no price data loaded; call fetch_data() first")

        real = [t for t in self.tickers if t not in SYNTHETIC_TICKERS]
        synthetic = [t for t in self.tickers if t in SYNTHETIC_TICKERS]
        missing_columns = [t for t in real if t not in self.price_data.columns]
        if missing_columns:
            logger.warning("tickers without price columns dropped: %s", ", ".join(missing_columns))
            real = [t for t in real if t in self.price_data.columns]
        if not real and not synthetic:
            raise DataValidationError("no tickers with price data to align")

        if real:
            prices = self.price_data[real].copy()
            prices.index = pd.to_datetime(prices.index)
            today = pd.Timestamp.now().normalize()
            future_rows = int((prices.index > today).sum())
            if future_rows:
                logger.warning("dropping %d future-dated rows from price history", future_rows)
                prices = prices.loc[prices.index <= today]
            prices, real = self._enforce_quality(prices)
        else:
            # Synthetic-only universe: build a business-day index for the window.
            end = pd.Timestamp.now().normalize()
            index = pd.bdate_range(end=end, periods=self.years * TRADING_DAYS_PER_YEAR)
            prices = pd.DataFrame(index=index)

        divs = pd.DataFrame(0.0, index=prices.index, columns=list(prices.columns))
        if self.div_data is not None:
            for ticker in prices.columns:
                if ticker in self.div_data:
                    shared = prices.index.intersection(self.div_data[ticker].index)
                    divs.loc[shared, ticker] = self.div_data[ticker].loc[shared]

        prices = prices.ffill().bfill()
        for ticker in synthetic:
            prices[ticker] = self._synthetic_prices(ticker, prices.index)
            divs[ticker] = 0.0

        self.tickers = list(prices.columns)
        logger.info(
            "aligned %d tickers over %d dates (%s to %s)",
            len(self.tickers),
            len(prices),
            prices.index[0].date(),
            prices.index[-1].date(),
        )
        return prices, divs

    def _enforce_quality(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Drop tickers with too much missing data, salvaging a common window first."""
        missing = prices.isna().mean() * 100
        if (missing > MAX_MISSING_PCT).all():
            # Systematic gaps usually mean misaligned listing windows, not bad
            # tickers: retry on the overlapping window before dropping anything.
            first_valid = prices.apply(lambda s: s.first_valid_index())
            last_valid = prices.apply(lambda s: s.last_valid_index())
            if first_valid.notna().all() and last_valid.notna().all():
                start, end = first_valid.max(), last_valid.min()
                if start <= end:
                    logger.warning(
                        "all tickers exceed %.0f%% missing data; retrying on common window %s to %s",
                        MAX_MISSING_PCT,
                        start.date(),
                        end.date(),
                    )
                    prices = prices.loc[start:end]
                    missing = prices.isna().mean() * 100

        keep = missing[missing <= MAX_MISSING_PCT].index.tolist()
        dropped = [t for t in prices.columns if t not in keep]
        if dropped:
            logger.warning(
                "dropping %d tickers over the %.0f%% missing-data threshold: %s",
                len(dropped),
                MAX_MISSING_PCT,
                ", ".join(dropped),
            )
        if not keep:
            raise DataValidationError(
                "every ticker failed the missing-data quality check; "
                "the cache may be stale (try --clear-cache) or the tickers unlisted"
            )
        return prices[keep], keep

    def _synthetic_prices(self, ticker: str, index: pd.Index) -> pd.Series:
        """Deterministic compounding price path for CASH/TBILLS.

        Deterministic on purpose: these are rate proxies, and random noise here
        made runs non-reproducible without informing the optimization.
        """
        rate = self.risk_free_rate * (_TBILLS_SPREAD if ticker == "TBILLS" else 1.0)
        daily = (1.0 + rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        return pd.Series((1.0 + daily) ** np.arange(len(index)), index=index)

    def _decorrelate_synthetic_assets(self) -> None:
        """Zero covariance between synthetic assets and everything else."""
        assert self.cov_matrix is not None
        for ticker in (t for t in self.tickers if t in SYNTHETIC_TICKERS):
            variance = 1e-8 if ticker == "CASH" else 1e-5
            self.cov_matrix.loc[ticker, :] = 0.0
            self.cov_matrix.loc[:, ticker] = 0.0
            self.cov_matrix.loc[ticker, ticker] = variance

    # ------------------------------------------------------------- optimize

    def build_optimizer(
        self, config: OptimizerConfig, exclude_cash: bool = False
    ) -> MeanVarianceOptimizer:
        """Create the optimization engine over the estimated inputs.

        Args:
            config: regime and constraint settings.
            exclude_cash: drop CASH/TBILLS from the optimized universe.

        Raises:
            DataValidationError: if returns have not been calculated, or the
                exclusion leaves no assets.
        """
        if self.mean_returns is None or self.cov_matrix is None:
            raise DataValidationError("returns not calculated; call calculate_returns() first")
        universe = list(self.tickers)
        if exclude_cash:
            universe = [t for t in universe if t not in SYNTHETIC_TICKERS]
            if not universe:
                raise DataValidationError("excluding cash left no assets to optimize")
        return MeanVarianceOptimizer(
            universe,
            self.mean_returns[universe].to_numpy(),
            self.cov_matrix.loc[universe, universe].to_numpy(),
            config,
        )


if __name__ == "__main__":
    # Backward-compatible entry point; the CLI moved to src/cli.py.
    from src.cli import main

    raise SystemExit(main())
