<!--
Keep PRs focused: one logical change each. This project is experimental and
educational, not investment advice. Do not weaken tests or lint rules to make a
red gate pass.
-->

## Hypothesis

<!-- What should be true after this change, and why? -->

## What would refute it

<!-- Name the output, test failure, metric, or invariant that would prove this wrong. -->

## What changed

<!-- Short implementation summary. Link related issues, for example "Closes #12". -->

## Tests run

<!-- Paste exact commands. The expected full gate is:
ruff check src tests && ruff format --check src tests && pytest -q
-->

- [ ] `ruff check src tests`
- [ ] `ruff format --check src tests`
- [ ] `pytest -q`

## What was not tested and why

<!-- Be explicit about any mode, ticker set, data source, cache state, or platform not covered. -->

## Risks and follow-ups

<!-- Note numerical risk, Yahoo Finance data dependence, cache behavior, and follow-up work. -->
