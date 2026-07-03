#!/bin/bash
# Smoke-tests the CLI across its main flag combinations, one log per case.
# These hit the network on first run; the pytest suite in tests/ is the
# deterministic gate — this script is a manual end-to-end check.
set -uo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="test_outputs"
mkdir -p "$OUTPUT_DIR"

declare -a TESTS=(
  "default|"
  "long_short|--mode long-short"
  "market_neutral|--mode market-neutral --exclude-cash"
  "no_cache|--no-cache"
  "debug|--debug"
  "exclude_cash|--exclude-cash"
  "fast|--fast"
  "custom_output|--output-dir $OUTPUT_DIR/custom"
)

failures=0
for TEST in "${TESTS[@]}"; do
  DESC="${TEST%%|*}"
  ARGS="${TEST#*|}"
  LOG_FILE="$OUTPUT_DIR/${DESC}.log"
  echo "Running test: $DESC ($ARGS)"
  # shellcheck disable=SC2086  # ARGS is intentionally word-split
  if ! python3 -m src.cli $ARGS >"$LOG_FILE" 2>&1; then
    echo "  ❌ FAILED — see $LOG_FILE"
    failures=$((failures + 1))
  else
    echo "  ✅ ok — output in $LOG_FILE"
  fi
done

echo ""
if [ "$failures" -gt 0 ]; then
  echo "❌ $failures test(s) failed. Check the $OUTPUT_DIR directory for logs."
  exit 1
fi
echo "✅ All smoke tests passed. Logs in $OUTPUT_DIR."
