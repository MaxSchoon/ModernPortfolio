#!/bin/bash

# Modern Portfolio Optimizer Runner
# This script helps run the optimizer with different configurations

# Add some colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}Modern Portfolio Optimizer${NC}"
echo -e "${BLUE}===========================================${NC}"

# Function to run the batch fetcher separately
function run_batch_fetch() {
    echo -e "\n${YELLOW}Running batch fetcher to prepare data...${NC}"
    echo -e "${YELLOW}This will fetch data in small batches with delays to avoid rate limits${NC}\n"
    
    python3 batch_fetcher.py --file tickers.csv --batch-size 3 --delay-min 2 --delay-max 5 --retry 3
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✅ Data fetching completed successfully${NC}"
        return 0
    else
        echo -e "\n${RED}❌ Error during data fetching${NC}"
        return 1
    fi
}

# Function to standardize the cache
function standardize_cache() {
    echo -e "\n${YELLOW}Standardizing cache data formats...${NC}"
    echo -e "${YELLOW}This will fix date formats and remove any future dates in the cache${NC}\n"
    
    python3 cache_standardize.py
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✅ Cache standardization completed successfully${NC}"
        return 0
    else
        echo -e "\n${RED}❌ Error standardizing cache${NC}"
        return 1
    fi
}

# Check which mode to run in
if [ "$1" == "--fetch-only" ]; then
    run_batch_fetch
    exit $?
fi

if [ "$1" == "--standardize-cache" ]; then
    standardize_cache
    exit $?
fi

if [ "$1" == "--clear-cache" ]; then
    echo -e "\n${YELLOW}Clearing cache before running optimizer...${NC}"
    CLEAR_OPT="--clear-cache"
else
    CLEAR_OPT=""
fi

if [ "$1" == "--no-cache" ]; then
    echo -e "\n${YELLOW}Running without cache...${NC}"
    CACHE_OPT="--no-cache"
else
    CACHE_OPT=""
fi

if [ "$1" == "--no-batch" ]; then
    echo -e "\n${YELLOW}Running without batch fetcher...${NC}"
    BATCH_OPT="--no-batch"
else
    BATCH_OPT=""
fi

# Set years of data
YEARS=${2:-5}

echo -e "\n${GREEN}Running optimizer with ${YEARS} years of data...${NC}"
python3 ModernPortfolio.py --years ${YEARS} ${CACHE_OPT} ${BATCH_OPT} ${CLEAR_OPT}

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Optimization completed successfully${NC}"
else
    echo -e "\n${RED}❌ Error during optimization${NC}"
fi
