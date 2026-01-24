#!/bin/bash
# College Basketball Power Rating - Full Pipeline
# ================================================
# This script runs both the R data extraction and Python rating calculation.
#
# Prerequisites:
#   - R with cbbdata package installed
#   - Python with sportsdataverse package installed
#   - cbbdata API credentials configured in R

echo "============================================"
echo "College Basketball Power Rating Pipeline"
echo "============================================"
echo

SEASON=${1:-2025}

echo "Season: $SEASON"
echo

# Step 1: Run R data extraction
echo "[Step 1/2] Extracting data from cbbdata (R)..."
echo

if Rscript data_extract.R $SEASON; then
    NO_PRIORS=""
else
    echo
    echo "Warning: R data extraction failed or not available."
    echo "Continuing with Python-only mode (no priors)..."
    echo
    NO_PRIORS="--no-priors"
fi

# Step 2: Run Python power ratings
echo
echo "[Step 2/2] Calculating power ratings (Python)..."
echo

python power_rating.py --season $SEASON $NO_PRIORS --output ratings_${SEASON}.csv

echo
echo "============================================"
echo "Pipeline complete!"
echo "Results saved to: ratings_${SEASON}.csv"
echo "============================================"
