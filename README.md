# Modern Portfolio Optimizer

Modern Portfolio Optimizer is an experimental and educational Modern Portfolio Theory playground for exploring Markowitz mean-variance optimization, short-enabled portfolios, market-neutral portfolios, and Kelly-style leverage suggestions. It is intended for learning and experimentation only; it is not investment advice, a trading system, or a claim of market edge.

## Feature Highlights

- Three optimization modes:
  - `long-only`: classic Markowitz constraints with `w_i >= 0` and `sum(w) = 1`.
  - `long-short`: shorts allowed with full use of proceeds, `sum(w) = 1`, and a gross-exposure cap `sum(abs(w)) <= gross_limit`. A gross limit of `1.6` corresponds to a 130/30 fund, while `2.0` corresponds to Reg-T-style 2x gross exposure.
  - `market-neutral`: dollar-neutral portfolio with net exposure `0` and normalized gross exposure `1`.
- Two allocation methods:
  - `mean-variance` (default): numerical max-Sharpe with verified constraints.
  - `hrp`: Hierarchical Risk Parity (Lopez de Prado, 2016) — clusters assets by correlation distance and splits capital by inverse cluster variance. Uses no return forecasts and no covariance inversion, so it stays stable where mean-variance estimates are noisy and even works on singular covariance matrices. Long-only by construction.
- Short borrow costs through `--borrow-rate`, charged against short notional.
- Kelly leverage suggestion using expected return, volatility, risk-free rate, and margin borrow cost.
- Typed core errors: `ConfigurationError`, `DataValidationError`, and `OptimizationError`.
- Self-contained HTML report with embedded charts.
- Shorts-aware charts, including signed allocation bars and long/short exposure metrics.
- CSV market-data cache with date standardization and maintenance tools.
- Batch fetching with configurable batch size and parallel workers.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For development, also install the lint and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

The default ticker file is `src/data/tickers.csv`. It must contain a `ticker` column.

## Quick Start

Run the CLI directly:

```bash
python3 -m src.cli
```

Or use the helper script, which runs a standard 5-year long-only optimization with cash and treasuries excluded:

```bash
./scripts/run_optimizer.sh
```

Long-short example:

```bash
python3 -m src.cli \
  --mode long-short \
  --gross-limit 1.6 \
  --max-short 0.30 \
  --borrow-rate 0.02
```

Market-neutral example:

```bash
python3 -m src.cli \
  --mode market-neutral \
  --exclude-cash \
  --max-short 0.30 \
  --borrow-rate 0.02
```

## CLI Reference

Global options:

| Flag | Default | Description |
|---|---:|---|
| `-h`, `--help` | n/a | Show help and exit. |
| `--version` | n/a | Show the CLI version and exit. |

### Data

| Flag | Default | Description |
|---|---:|---|
| `--tickers-file PATH` | `src/data/tickers.csv` | CSV file with a `ticker` column. |
| `--years INT` | `5` | Years of history to use. |
| `--no-cache` | `False` | Bypass the local data cache. |
| `--clear-cache` | `False` | Clear the cache before running. |
| `--cache-dir PATH` | `data_cache` | Directory for cached market data. |
| `--batch-size INT` | `50` | Tickers per fetch batch. |
| `--workers INT` | `3` | Parallel fetch workers. |
| `--no-standardize` | `False` | Skip the cache date-standardization pass. |

### Strategy

| Flag | Default | Description |
|---|---:|---|
| `--optimizer {mean-variance,hrp}` | `mean-variance` | Allocation method: numerical max-Sharpe, or Hierarchical Risk Parity (long-only, forecast-free). |
| `--mode {long-only,long-short,market-neutral}` | `long-only` | Portfolio regime. |
| `--risk-free RATE` | `0.04` | Annual risk-free rate. |
| `--margin-cost RATE` | `0.065` | Annual margin borrow rate used in Kelly leverage. |
| `--max-weight W` | `1.0` | Per-asset cap on long weight. |
| `--max-short W` | `0.3` | Per-asset cap on short weight, as a positive fraction. |
| `--gross-limit G` | `1.6` | Gross-exposure cap for long-short portfolios. |
| `--borrow-rate RATE` | `0.0` | Annual borrow fee charged on short notional. |
| `--exclude-cash` | `False` | Exclude synthetic `CASH` and `TBILLS` from optimization. |

### Output

| Flag | Default | Description |
|---|---:|---|
| `--output-dir PATH` | `portfolio_analysis` | Directory for results, charts, and the report. |
| `--skip-plots` | `False` | Skip chart generation. |
| `--no-html-report` | `False` | Skip the self-contained HTML report. |
| `--frontier-points INT` | `30` | Resolution of the efficient frontier. |
| `--price-charts` | `False` | Also write one price-history chart per ticker. |
| `--fast` | `False` | Fastest run; implies `--skip-plots` and `--no-html-report`. |
| `-q`, `--quiet` | `False` | Print errors only. Mutually exclusive with `--debug`. |
| `--debug` | `False` | Print verbose diagnostics. Mutually exclusive with `--quiet`. |
| `--no-color` | `False` | Disable ANSI colors. |

## Project Structure

```text
ModernPortfolio/
|-- src/
|   |-- cli.py
|   |-- core/
|   |   |-- ModernPortfolio.py
|   |   |-- optimization.py
|   |   `-- exceptions.py
|   |-- reporting/
|   |   |-- html_report.py
|   |   `-- plots.py
|   |-- fetching/
|   |   `-- data_fetcher.py
|   |-- cache/
|   |   |-- cache_maintenance.py
|   |   |-- cache_standardize.py
|   |   `-- csv_cache_manager.py
|   |-- utils/
|   |   |-- cache_tools.py
|   |   `-- utils.py
|   |-- data/
|   |   `-- tickers.csv
|   `-- tests/
|       `-- test_yahoo.py
|-- tests/
|   |-- test_cli.py
|   |-- test_optimization.py
|   |-- test_pipeline.py
|   `-- test_reporting.py
|-- scripts/
|   |-- run_optimizer.sh
|   `-- test_modern_portfolio.sh
|-- data_cache/
|   |-- prices/
|   |-- dividends/
|   |-- metadata/
|   `-- info/
|-- portfolio_analysis/
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
|-- CONTRIBUTING.md
|-- LICENSE
`-- README.md
```

## Why account for your trades?

Accounting for your trades is the FIRST step to improving market performance because it creates a ledger of accountability. Without a record of what was intended, what was executed, what it cost, and what happened afterward, performance remains an impression rather than evidence.

Successful quantitative trading is hard because many variables sit between a backtest and realized returns:

- Transaction costs: commissions, exchange fees, broker fees, and other explicit charges.
- Bid-ask spread: crossing the spread can erase small modeled advantages.
- Slippage: fills differ from modeled prices, especially during volatility or thin liquidity.
- Market impact: larger orders can move the market against the strategy.
- Order execution mechanics: order types, fill quality, partial fills, queue position, routing, and latency all matter.
- Short borrow fees and availability: a shortable security can become expensive, unavailable, or recalled.
- Margin interest: borrowed capital has a financing cost that changes over time.
- Opportunity cost of capital: capital tied to one strategy cannot be used elsewhere.
- Taxes: realized gains, losses, dividends, withholding taxes, and account structure affect net returns.
- Dividends and corporate actions: splits, special dividends, spin-offs, mergers, and delistings alter realized performance.
- FX conversion for international listings: currency moves and conversion spreads can dominate local-asset returns.
- Data issues: survivorship bias, look-ahead bias, stale prices, adjusted-price errors, and missing history can distort a backtest.
- Estimation error: expected returns and covariances are noisy, unstable inputs.
- Regime change and non-stationarity: relationships estimated in one period may not persist.
- Capacity and liquidity limits: a strategy can stop working when scaled beyond what the market can absorb.
- Rebalancing drift and timing: real portfolios move between rebalance dates, and execution timing changes realized exposure.

This project can help explore portfolio construction assumptions, but it does not solve these execution, accounting, and market-structure problems.

## Landscape of Trading Products

Markets contain many product types: indexes, index futures, options, turbos/leveraged certificates, ETFs, tokens, perpetual futures, spot markets, bonds, and stocks. They do not all behave like simple historical return streams.

Some products are yield-bearing, such as dividend-paying stocks, coupon-paying bonds, staking tokens, or cash-like instruments. Some are cost-bearing, such as perpetual futures with funding rates, options with theta decay, borrowed shorts with borrow fees, futures with roll costs, leveraged products with financing drag, and FX-linked products with conversion costs.

Professional participants arbitrage within and between these products across venues, clearing systems, brokers, market makers, exchanges, and counterparties. That practical reality is one reason a naive optimizer over historical returns is best treated as a learning tool, not as an edge by itself.

## Data Management Tools

The optimizer stores fetched Yahoo Finance data in a CSV cache under `data_cache/`.

Useful tools:

| Tool | Purpose |
|---|---|
| `src/utils/cache_tools.py` | Lightweight cache listing, inspection, validation, and basic NaN fixes. |
| `src/cache/cache_maintenance.py` | Advanced cache inspection, ticker validation, repair, and batch maintenance. |
| `src/cache/cache_standardize.py` | Date-format standardization for cached price and dividend CSV files. |
| `src/tests/test_yahoo.py` | Manual Yahoo Finance connectivity check. |

The main CLI also exposes cache controls: `--no-cache`, `--clear-cache`, `--cache-dir`, and `--no-standardize`.

## Output and Analysis

By default, outputs are written to `portfolio_analysis/`:

- `portfolio_analysis/report.html`: self-contained HTML report.
- `portfolio_analysis/optimal_weights.json`: optimized weights in percent.
- `portfolio_analysis/returns_summary.csv`: per-asset annualized return, volatility, and Sharpe.
- `portfolio_analysis/correlations.csv`: return correlation matrix.
- `portfolio_analysis/portfolio_allocation.png`: signed long/short allocation chart.
- `portfolio_analysis/efficient_frontier.png`: efficient frontier when available.
- `portfolio_analysis/risk_return_profile.png`: per-asset risk/return chart.
- `portfolio_analysis/price_charts/`: optional per-ticker price charts when `--price-charts` is used.

Use `--output-dir` to write results somewhere else.

## Testing

Run the deterministic project gate:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

The repository configuration for ruff and pytest lives in `pyproject.toml`.

For manual end-to-end smoke tests across common CLI flag combinations:

```bash
./scripts/test_modern_portfolio.sh
```

The smoke script may hit the network on a cold cache; the pytest suite is the deterministic gate.

## Roadmap

The main roadmap item is a backtesting system for efficient portfolios:

- Simulate portfolios over historical periods with monthly, quarterly, or custom rebalancing.
- Track realized returns, volatility, drawdowns, turnover, transaction costs, and financing costs.
- Compare long-only, long-short, and market-neutral configurations under the same assumptions.
- Keep outputs reproducible so optimizer changes can be evaluated against realized backtest behavior.
- Consider a faster execution layer if simulation speed becomes the bottleneck.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, quality gates, optimizer discipline, and pull request expectations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
