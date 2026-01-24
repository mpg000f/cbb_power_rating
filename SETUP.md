# College Basketball Power Rating System - Setup Guide

## Overview

This system calculates college basketball power ratings using:
- **Opponent-adjusted efficiency** (offensive and defensive)
- **Stability metrics** (rebounding, turnovers, consistency)
- **Prior ratings** with transfer portal and returning production data

## Prerequisites

### Python
```bash
pip install sportsdataverse pandas numpy
```

### R (for prior data - transfers, recruiting, returning production)
```r
install.packages("devtools")
devtools::install_github("andreweatherman/cbbdata")
install.packages(c("dplyr", "readr"))
```

## First-Time R Setup

1. Create a free cbbdata account:
```r
library(cbbdata)
cbd_create_account(
  username = "your_username",
  email = "your@email.com",
  password = "your_password"
)
```

2. Check your email for the API key (check spam folder)

3. Set up persistent login by adding to your `.Renviron` file:
```
CBD_USER=your_username
CBD_PW=your_password
```

Or login per-session:
```r
cbd_login(username = "your_username", password = "your_password")
```

## Usage

### Full Pipeline (R + Python)

**Windows:**
```cmd
run_ratings.bat 2025
```

**Mac/Linux:**
```bash
./run_ratings.sh 2025
```

### Python Only (No Priors)

If you don't have R set up, you can run Python-only mode:
```bash
python power_rating.py --season 2025 --no-priors
```

This will calculate ratings based on current season data only, without transfer/returning production adjustments.

### Manual Step-by-Step

1. Extract data from R:
```bash
Rscript data_extract.R 2025
```

2. Run Python ratings:
```bash
python power_rating.py --season 2025 --output ratings.csv
```

## Output Files

### From R (`data/` directory):
- `combined_prior_data.csv` - Main prior data file
- `transfers.csv` - Transfer portal data
- `transfer_production.csv` - Aggregated transfer production by team
- `returning_production.csv` - Returning player production
- `team_ratings_prior.csv` - Prior season Barttorvik ratings
- `team_ratings_current.csv` - Current season ratings

### From Python:
- `ratings_2025.csv` - Final power ratings

## Configuration

Edit `power_rating.py` or create a custom config:

```python
from power_rating import RatingConfig, run_power_ratings

config = RatingConfig(
    # Adjust component weights (must sum to 1.0)
    weight_defensive_efficiency=0.25,
    weight_offensive_efficiency=0.20,
    weight_rebounding=0.20,
    weight_turnover_margin=0.12,
    weight_free_throw_rate=0.08,
    weight_consistency=0.15,

    # Prior decay (50% at game 8, 0% at game 20)
    prior_decay_games=20,
    prior_decay_power=0.756,

    # Regression toward mean
    prior_regression_factor=0.5,

    # Transfer scaling by tier
    transfer_scale_high_to_high=0.80,
    transfer_scale_mid_to_high=0.65,
    transfer_scale_high_to_mid=0.90,
)

results = run_power_ratings(current_season=2025, config=config)
```

## Rating Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Defensive Efficiency | 25% | Opponent-adjusted points allowed per 100 possessions |
| Offensive Efficiency | 20% | Opponent-adjusted points scored per 100 possessions |
| Rebounding | 20% | ORB% + DRB% composite |
| Turnover Margin | 12% | Opponent TO rate - Own TO rate |
| Free Throw Rate | 8% | FTA / FGA |
| Consistency | 15% | Inverse of game-to-game variance |

## Prior System

- Prior weight decays from ~91% at game 1 to 0% at game 20
- 50/50 crossover at game 8
- Priors based on last year's rating, adjusted for:
  - Returning production (% of minutes returning)
  - Transfer production (scaled by conference tier)
  - Partial regression toward D1 mean

## Troubleshooting

### "cbbdata API authentication failed"
- Verify CBD_USER and CBD_PW are set in .Renviron
- Try `cbd_login()` manually to test credentials

### "sportsdataverse import error"
- Install with: `pip install sportsdataverse`

### "No prior data found"
- Run `Rscript data_extract.R 2025` first
- Check that `data/combined_prior_data.csv` exists

### Missing teams in output
- Some smaller programs may not have complete data
- Check the sportsdataverse data coverage for your season
