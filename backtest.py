"""
Backtesting framework for power ratings.

Tests predictive accuracy by:
1. For each game, calculate ratings using ONLY prior games
2. Predict the spread
3. Compare to actual result
4. Report error metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import from main module
from power_rating import (
    load_team_boxscores, calculate_game_stats, merge_opponent_stats,
    calculate_kenpom_possessions, apply_home_away_adjustment,
    apply_margin_caps, apply_outlier_dampening, calculate_opponent_adjusted_efficiency,
    aggregate_team_season, apply_sos_regression, RatingConfig
)


def run_backtest(season: int, min_games_before_predict: int = 5, sample_every_n_days: int = 7):
    """
    Run backtest for a season.

    Args:
        season: Year to backtest (e.g., 2025)
        min_games_before_predict: Minimum games a team must have played before we predict their games
        sample_every_n_days: Only recalculate ratings every N days (for speed)
    """
    print(f"Loading {season} game data...")

    # Load raw boxscores
    boxscores = load_team_boxscores([season])

    # Parse dates
    boxscores['game_date'] = pd.to_datetime(boxscores['game_date'])

    # Get all unique game dates, sorted
    game_dates = sorted(boxscores['game_date'].unique())
    print(f"Found {len(game_dates)} game dates from {game_dates[0].date()} to {game_dates[-1].date()}")

    # Load prior year ratings for preseason component
    prior_ratings = None
    try:
        prior_ratings = pd.read_csv(f'historical_ratings/ratings_{season-1}.csv')
        print(f"Loaded {len(prior_ratings)} prior ratings from {season-1}")
    except:
        print(f"No prior ratings found for {season-1}")

    config = RatingConfig()

    # Track predictions
    predictions = []

    # Skip first N days to build up some data
    start_idx = 10  # Start predicting after ~10 days of games

    # Current ratings (updated every N days)
    current_ratings = None
    rating_lookup = {}
    tempo_lookup = {}
    games_lookup = {}
    last_rating_date = None

    for i, predict_date in enumerate(game_dates[start_idx:], start=start_idx):
        # Recalculate ratings every N days (or if we don't have any)
        should_recalc = (
            current_ratings is None or
            last_rating_date is None or
            (predict_date - last_rating_date).days >= sample_every_n_days
        )

        if should_recalc:
            # Get all games before this date
            prior_games = boxscores[boxscores['game_date'] < predict_date].copy()

            if len(prior_games) < 100:  # Need enough games for stable ratings
                continue

            # Calculate ratings using only prior games
            try:
                current_ratings = calculate_ratings_for_date(prior_games, prior_ratings, config, season)
            except Exception as e:
                print(f"Error calculating ratings for {predict_date.date()}: {e}")
                continue

            if current_ratings is None or len(current_ratings) == 0:
                continue

            # Update lookups
            rating_lookup = dict(zip(current_ratings['team_id'], current_ratings['power_rating']))
            tempo_lookup = dict(zip(current_ratings['team_id'], current_ratings['tempo']))
            games_lookup = dict(zip(current_ratings['team_id'], current_ratings['games_played']))
            last_rating_date = predict_date
            print(f"  Recalculated ratings as of {predict_date.date()}")

        # Get games on this date to predict
        todays_games = boxscores[boxscores['game_date'] == predict_date].copy()

        if len(todays_games) == 0:
            continue

        # Predict each game
        game_ids_seen = set()
        for _, game in todays_games.iterrows():
            game_id = game['game_id']

            # Skip if we've already processed this game (each game appears twice)
            if game_id in game_ids_seen:
                continue
            game_ids_seen.add(game_id)

            team_id = game['team_id']
            opp_id = game.get('opponent_team_id')

            if team_id not in rating_lookup or opp_id not in rating_lookup:
                continue

            # Check minimum games requirement
            if games_lookup.get(team_id, 0) < min_games_before_predict:
                continue
            if games_lookup.get(opp_id, 0) < min_games_before_predict:
                continue

            team_rating = rating_lookup[team_id]
            opp_rating = rating_lookup[opp_id]

            # Get tempos for pace adjustment
            team_tempo = tempo_lookup.get(team_id, 68)
            opp_tempo = tempo_lookup.get(opp_id, 68)
            expected_tempo = (team_tempo + opp_tempo) / 2
            tempo_factor = expected_tempo / 100

            # Predict spread (team perspective)
            raw_spread = (team_rating - opp_rating) * tempo_factor

            # Home court adjustment
            home_away = game.get('home_away', '')
            if home_away == 'home':
                predicted_spread = raw_spread + config.home_court_efficiency_boost * tempo_factor
            elif home_away == 'away':
                predicted_spread = raw_spread - config.home_court_efficiency_boost * tempo_factor
            else:
                predicted_spread = raw_spread  # Neutral

            # Actual result
            team_score = game.get('team_score', 0)
            opp_score = game.get('opponent_team_score', 0)
            actual_margin = team_score - opp_score

            predictions.append({
                'game_id': game_id,
                'date': predict_date,
                'team_id': team_id,
                'opp_id': opp_id,
                'team_name': game.get('team_display_name', ''),
                'opp_name': game.get('opponent_team_display_name', ''),
                'team_rating': team_rating,
                'opp_rating': opp_rating,
                'predicted_spread': predicted_spread,
                'actual_margin': actual_margin,
                'team_score': team_score,
                'opp_score': opp_score,
                'home_away': home_away,
                'error': predicted_spread - actual_margin
            })

        # Progress update
        if i % 10 == 0:
            print(f"  Processed through {predict_date.date()}, {len(predictions)} predictions so far")

    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)

    if len(pred_df) == 0:
        print("No predictions generated!")
        return None

    # Calculate metrics
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {season} Season")
    print(f"{'='*60}")
    print(f"Total games predicted: {len(pred_df)}")

    # Error metrics
    mae = pred_df['error'].abs().mean()
    rmse = np.sqrt((pred_df['error'] ** 2).mean())
    print(f"\nPrediction Error:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"  Root Mean Square Error (RMSE): {rmse:.2f} points")

    # Straight up record (did we pick the winner?)
    pred_df['predicted_winner'] = pred_df['predicted_spread'] > 0
    pred_df['actual_winner'] = pred_df['actual_margin'] > 0
    pred_df['correct_winner'] = pred_df['predicted_winner'] == pred_df['actual_winner']

    su_correct = pred_df['correct_winner'].sum()
    su_total = len(pred_df)
    su_pct = su_correct / su_total * 100
    print(f"\nStraight Up Record:")
    print(f"  {su_correct}-{su_total - su_correct} ({su_pct:.1f}%)")

    # Against the spread (assuming we're picking the favorite)
    # A prediction "covers" if the actual margin beats our spread
    # Since we don't have Vegas lines, we'll measure calibration instead

    # Calibration by predicted spread buckets
    print(f"\nCalibration by Predicted Spread:")
    pred_df['spread_bucket'] = pd.cut(pred_df['predicted_spread'].abs(),
                                       bins=[0, 3, 6, 10, 15, 100],
                                       labels=['0-3', '3-6', '6-10', '10-15', '15+'])

    for bucket in ['0-3', '3-6', '6-10', '10-15', '15+']:
        bucket_games = pred_df[pred_df['spread_bucket'] == bucket]
        if len(bucket_games) > 0:
            bucket_correct = bucket_games['correct_winner'].sum()
            bucket_pct = bucket_correct / len(bucket_games) * 100
            avg_error = bucket_games['error'].abs().mean()
            print(f"  {bucket} pts: {bucket_correct}/{len(bucket_games)} ({bucket_pct:.1f}% correct), MAE={avg_error:.1f}")

    # Save predictions
    output_file = f'backtest_results_{season}.csv'
    pred_df.to_csv(output_file, index=False)
    print(f"\nDetailed predictions saved to {output_file}")

    return pred_df


def calculate_ratings_for_date(games_df, prior_ratings, config, season):
    """
    Calculate ratings using only the provided games.
    Simplified version of the main pipeline for backtesting.
    """
    # Run through the pipeline
    game_stats = calculate_game_stats(games_df.copy())
    with_opponents = merge_opponent_stats(game_stats)
    with_opponents = calculate_kenpom_possessions(with_opponents)

    # Filter to current season
    current_games = with_opponents[with_opponents['season'] == season].copy()

    # Filter D1 teams (need at least some games)
    team_games = current_games.groupby('team_id').size()
    d1_teams = team_games[team_games >= 3].index  # Lower threshold for early season
    current_games = current_games[current_games['team_id'].isin(d1_teams)]
    current_games = current_games[current_games['opp_team_id'].isin(d1_teams)]

    if len(current_games) < 50:
        return None

    # Apply adjustments
    current_games = apply_home_away_adjustment(current_games, config)
    current_games = apply_margin_caps(current_games, config)

    # Opponent adjustment - returns team_id, adj_off_eff, adj_def_eff
    adj_efficiency = calculate_opponent_adjusted_efficiency(current_games, config)

    # Aggregate tempo and game count from current_games (adj_efficiency doesn't have these)
    game_agg = current_games.groupby('team_id').agg(
        games_played=('game_id', 'count'),
        tempo=('possessions', 'mean'),
    ).reset_index()

    # Merge adj_efficiency with game aggregates
    team_season = adj_efficiency.merge(game_agg, on='team_id', how='left')

    team_season['season'] = season

    # Add team_display_name
    name_lookup = games_df.groupby('team_id')['team_display_name'].first().to_dict()
    team_season['team_display_name'] = team_season['team_id'].map(name_lookup)

    # Simple power rating (skip SOS regression for speed)
    team_season['power_rating'] = team_season['adj_off_eff'] - team_season['adj_def_eff']

    return team_season


if __name__ == '__main__':
    import sys

    season = 2025
    sample_days = 5  # Recalculate ratings every 5 days for speed

    if len(sys.argv) > 1:
        season = int(sys.argv[1])
    if len(sys.argv) > 2:
        sample_days = int(sys.argv[2])

    results = run_backtest(season, sample_every_n_days=sample_days)
