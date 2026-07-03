"""CLI contract tests: argument parsing, exit codes, error presentation."""

import pytest
from src.cli import EXIT_ERROR, EXIT_USAGE, build_parser, main
from src.core.optimization import OptimizationMode


class TestParser:
    def test_defaults_are_long_only_and_safe(self):
        args = build_parser().parse_args([])
        assert args.mode is OptimizationMode.LONG_ONLY
        assert args.years == 5
        assert not args.no_cache

    def test_mode_flag_accepts_all_regimes(self):
        for value in ("long-only", "long-short", "market-neutral"):
            args = build_parser().parse_args(["--mode", value])
            assert args.mode.value == value

    def test_rejects_unknown_mode(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--mode", "yolo"])
        assert excinfo.value.code == EXIT_USAGE

    def test_short_selling_flags_parse(self):
        args = build_parser().parse_args(
            [
                "--mode",
                "long-short",
                "--max-short",
                "0.5",
                "--gross-limit",
                "2.0",
                "--borrow-rate",
                "0.01",
            ]
        )
        assert args.max_short == 0.5
        assert args.gross_limit == 2.0
        assert args.borrow_rate == 0.01

    def test_quiet_and_debug_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--quiet", "--debug"])

    def test_optimizer_flag_parses(self):
        assert build_parser().parse_args([]).optimizer == "mean-variance"
        assert build_parser().parse_args(["--optimizer", "hrp"]).optimizer == "hrp"

    def test_rejects_unknown_optimizer(self):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--optimizer", "genetic"])
        assert excinfo.value.code == EXIT_USAGE


class TestMain:
    def test_missing_tickers_file_exits_with_error_not_traceback(self, tmp_path, capsys):
        code = main(["--tickers-file", str(tmp_path / "nope.csv"), "--quiet"])
        assert code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "error:" in err
        assert "Traceback" not in err

    def test_invalid_config_is_a_clean_error(self, tmp_path, capsys):
        tickers = tmp_path / "tickers.csv"
        tickers.write_text("ticker\nAAA\n")
        # gross-limit < 1 is inconsistent with long-short's budget constraint.
        code = main(
            [
                "--tickers-file",
                str(tickers),
                "--mode",
                "long-short",
                "--gross-limit",
                "0.5",
                "--quiet",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == EXIT_ERROR
        assert "gross_limit" in capsys.readouterr().err

    def test_hrp_with_shorts_mode_is_a_clean_error_before_any_fetch(self, tmp_path, capsys):
        tickers = tmp_path / "tickers.csv"
        tickers.write_text("ticker\nAAA\n")
        code = main(
            [
                "--tickers-file",
                str(tickers),
                "--optimizer",
                "hrp",
                "--mode",
                "long-short",
                "--quiet",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "long-only" in err
        assert "Traceback" not in err

    def test_hrp_rejects_max_weight_instead_of_ignoring_it(self, tmp_path, capsys):
        tickers = tmp_path / "tickers.csv"
        tickers.write_text("ticker\nAAA\n")
        code = main(
            [
                "--tickers-file",
                str(tickers),
                "--optimizer",
                "hrp",
                "--max-weight",
                "0.6",
                "--quiet",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == EXIT_ERROR
        assert "max-weight" in capsys.readouterr().err

    def test_hrp_end_to_end_through_the_cli(self, tmp_path):
        # Full path: cached prices → pipeline → HRP → charts → HTML report,
        # via main() with zero network access (cache pre-seeded).
        import json

        import numpy as np
        import pandas as pd
        from src.cache.csv_cache_manager import CSVDataCache

        cache = CSVDataCache(str(tmp_path / "cache"))
        index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=504)
        rng = np.random.default_rng(11)
        for ticker, drift in (("AAA", 0.0008), ("BBB", 0.0002), ("CCC", 0.0005)):
            steps = rng.normal(drift, 0.01, size=504)
            prices = pd.Series(100 * np.exp(np.cumsum(steps)), index=index)
            assert cache.save_price_data(ticker, prices)
        tickers = tmp_path / "tickers.csv"
        tickers.write_text("ticker\nAAA\nBBB\nCCC\n")

        out = tmp_path / "out"
        code = main(
            [
                "--tickers-file",
                str(tickers),
                "--optimizer",
                "hrp",
                "--years",
                "2",
                "--no-standardize",
                "--quiet",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output-dir",
                str(out),
            ]
        )
        assert code == 0
        report = (out / "report.html").read_text()
        assert "hrp" in report
        weights = json.loads((out / "optimal_weights.json").read_text())
        assert set(weights) == {"AAA", "BBB", "CCC"}
        assert all(w >= 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(100.0, abs=1e-6)
        assert (out / "portfolio_allocation.png").stat().st_size > 0
