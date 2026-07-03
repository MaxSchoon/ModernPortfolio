"""Typed exceptions for the portfolio core.

Library code raises these instead of printing and returning ``None``; only the
CLI boundary catches them and maps them to exit codes and user-facing messages.
"""


class PortfolioError(Exception):
    """Base class for all portfolio errors."""


class ConfigurationError(PortfolioError):
    """A configuration value is invalid or internally inconsistent."""


class DataValidationError(PortfolioError):
    """Input data (prices, returns, covariance) is unusable for optimization."""


class OptimizationError(PortfolioError):
    """The optimizer failed to produce a feasible, converged solution."""
