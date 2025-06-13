#!/bin/bash

# Directory for test outputs
OUTPUT_DIR="test_outputs"
mkdir -p "$OUTPUT_DIR"

# Test cases: (description, arguments)
declare -a TESTS=(
  "default|"
  "no_cache|--no-cache"
  "clear_cache|--clear-cache"
  "debug|--debug"
  "exclude_cash|--exclude-cash"
  "skip_plots|--skip-plots"
  "custom_output|--output-dir $OUTPUT_DIR/custom"
)

for TEST in "${TESTS[@]}"; do
  DESC="${TEST%%|*}"
  ARGS="${TEST#*|}"
  LOG_FILE="$OUTPUT_DIR/${DESC}.log"
  echo "Running test: $DESC ($ARGS)"
  python3 -m src.core.ModernPortfolio $ARGS 2>&1 | tee "$LOG_FILE"
  echo "  Output saved to $LOG_FILE"
done

echo "\nAll tests completed. Check the $OUTPUT_DIR directory for logs and outputs." 