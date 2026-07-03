#!/bin/bash
# Modern Portfolio Optimizer — simple runner.
# Runs a standard 5-year, long-only optimization excluding cash/treasuries.
# Pass extra flags through, e.g.: ./scripts/run_optimizer.sh --mode long-short
set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}Modern Portfolio Optimizer${NC}"
echo -e "${BLUE}===========================================${NC}"

if python3 -m src.cli --years 5 --tickers-file src/data/tickers.csv --exclude-cash "$@"; then
    echo -e "\n${GREEN}✅ Optimization completed successfully${NC}"
else
    status=$?
    echo -e "\n${RED}❌ Optimization failed (exit code ${status})${NC}"
    exit "$status"
fi
