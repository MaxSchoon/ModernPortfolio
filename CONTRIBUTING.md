# Contributing to Modern Portfolio Optimizer

Thanks for considering a contribution. This project is an experimental,
educational Modern Portfolio Theory optimizer. It is a place to test portfolio
optimization ideas, not a source of investment advice.

## What this project is

Modern Portfolio Optimizer is a Python project for exploring portfolio
construction with historical market data, including long-only, short-enabled,
and market-neutral configurations. The code fetches data from Yahoo Finance,
caches it locally, runs optimization routines, and produces analysis output.

Keep changes focused on making the optimizer easier to reason about, test, and
extend. Do not add claims about investment performance unless the claim is
directly reproducible from the code and data in the pull request.

## Local setup

Use a virtual environment and install both runtime and development
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Quality gate

Run the full local gate before opening a pull request:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

The combined command is:

```bash
ruff check src tests && ruff format --check src tests && pytest -q
```

Do not weaken tests, mark failing tests as expected failures, hardcode fixture
answers, or loosen lint configuration to rescue a red gate. Fix the code or fix
the test with a clear reason.

## Optimizer discipline

Optimizer changes need tests or examples that can fail if the behavior is
wrong. When a change touches weights, objectives, constraints, risk metrics,
or reporting derived from optimizer output, exercise all relevant regimes:

- Long-only
- Short-enabled
- Market-neutral

For numerical or financial correctness changes, state the expected behavior in
plain terms: the input ticker set, the optimizer mode, the command run, and the
weights or metrics that would refute the change if they came out differently.

## How to contribute

1. Fork the repository on GitHub.
2. Create a focused branch for one logical change.
3. Make the smallest change that proves the hypothesis.
4. Run the quality gate above.
5. Open a pull request using the template.
6. Respond to review with code or evidence, not weakened checks.

## Pull request expectations

Every pull request should include:

- A hypothesis: what the change is expected to improve or fix.
- What would refute it: the output, test failure, or metric that would prove it
  wrong.
- What changed: a concise summary of the implementation.
- Tests run: exact commands and any important output.
- What was not tested and why.
- Risks and follow-ups.

## Reporting bugs and correctness issues

Use the issue template that best fits:

- Bug report: runtime failures, bad CLI behavior, data-fetch problems, cache
  issues, or report generation problems.
- Optimizer correctness: numerical, financial, or constraint behavior that
  appears wrong.
- Enhancement: new features, new optimizer modes, better reporting, or
  usability improvements.

## License

By contributing you agree your contribution is licensed under the repository's
MIT License.
