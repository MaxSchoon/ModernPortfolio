#!/bin/bash

# Modern Portfolio Optimizer Simple Runner
# This script runs a standard optimization with 5 years of data, excluding cash and treasuries

# Add some colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}Modern Portfolio Optimizer${NC}"
echo -e "${BLUE}===========================================${NC}"

echo -e "\n${GREEN}Running optimizer with 5 years of data, excluding cash and treasuries...${NC}"
python3 -m src.core.ModernPortfolio --years 5 --tickers-file src/data/tickers.csv --exclude-cash

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Optimization completed successfully${NC}"
else
    echo -e "\n${RED}❌ Error during optimization${NC}"
fi
