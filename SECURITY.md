# Security Policy

This repository is an experimental Python portfolio optimizer. It fetches
market data from Yahoo Finance, caches data locally, and produces portfolio
analysis output. It executes no untrusted input.

The realistic risk surface is:

1. **Supply chain** - Python dependencies and GitHub Actions workflows.
2. **Data handling** - unexpected behavior when Yahoo Finance returns malformed,
   missing, stale, or rate-limited data.
3. **Local file handling** - cache and report output paths should stay within
   the repository's expected data and output directories.
4. **Optimizer correctness** - a defect that produces misleading weights,
   constraints, or risk metrics. Treat correctness reports seriously, but use
   the public optimizer correctness issue template unless the report also
   exposes a security vulnerability.

## Reporting a vulnerability

Please do not open a public issue for a security vulnerability. Use GitHub
private vulnerability reporting instead:

https://github.com/MaxSchoon/ModernPortfolio/security/advisories/new

Include what is wrong, how to reproduce it, the affected file or command, and
the expected impact.

## What to expect

- Acknowledgement within a few working days.
- An assessment of scope and severity.
- A fix on a private branch for confirmed vulnerabilities, merged once
  verified.
- Credit in release notes if you would like it.

## Scope notes

This project is not financial, legal, tax, or investment advice. Reports that a
specific optimizer path produces wrong, misleading, or unreproducible output are
welcome, but they are usually handled as correctness bugs rather than security
vulnerabilities.
