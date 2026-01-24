@echo off
REM College Basketball Power Rating - Full Pipeline
REM ================================================
REM This script runs both the R data extraction and Python rating calculation.
REM
REM Prerequisites:
REM   - R with cbbdata package installed
REM   - Python with sportsdataverse package installed
REM   - cbbdata API credentials configured in R

echo ============================================
echo College Basketball Power Rating Pipeline
echo ============================================
echo.

SET SEASON=%1
IF "%SEASON%"=="" SET SEASON=2025

echo Season: %SEASON%
echo.

REM Step 1: Run R data extraction
echo [Step 1/2] Extracting data from cbbdata (R)...
echo.
Rscript data_extract.R %SEASON%

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Warning: R data extraction failed or not available.
    echo Continuing with Python-only mode (no priors)...
    echo.
    SET NO_PRIORS=--no-priors
) ELSE (
    SET NO_PRIORS=
)

REM Step 2: Run Python power ratings
echo.
echo [Step 2/2] Calculating power ratings (Python)...
echo.
python power_rating.py --season %SEASON% %NO_PRIORS% --output ratings_%SEASON%.csv

echo.
echo ============================================
echo Pipeline complete!
echo Results saved to: ratings_%SEASON%.csv
echo ============================================
