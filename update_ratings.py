#!/usr/bin/env python3
"""
Daily Power Ratings Update Script
==================================
Updates the current season's power ratings and saves to the historical_ratings folder.

Usage:
    python update_ratings.py                    # Update current season (2026)
    python update_ratings.py --season 2025     # Update specific season
    python update_ratings.py --all             # Update 2025 and 2026

Schedule this script to run daily:
    Linux/WSL (cron): crontab -e
        0 6 * * * cd /mnt/c/Users/mpgra/Documents/cbb_power_rating && python update_ratings.py >> logs/update.log 2>&1

    Windows (Task Scheduler):
        Create task to run: python C:\\Users\\mpgra\\Documents\\cbb_power_rating\\update_ratings.py
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from power_rating import run_power_ratings, RatingConfig


def update_season(season: int, config: RatingConfig = None) -> bool:
    """
    Update ratings for a single season and save to CSV.

    Returns True on success, False on failure.
    """
    if config is None:
        config = RatingConfig()

    output_dir = project_root / "historical_ratings"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"ratings_{season}.csv"
    log_prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

    print(f"{log_prefix} Starting ratings update for season {season}...")

    try:
        # Run the rating pipeline
        results = run_power_ratings(
            current_season=season,
            config=config,
            use_r_data=True
        )

        # Save results
        results.to_csv(output_path, index=False)

        # Print summary
        top_team = results.iloc[0]
        print(f"{log_prefix} Update complete for season {season}")
        print(f"{log_prefix}   Teams rated: {len(results)}")
        print(f"{log_prefix}   #1 team: {top_team['team_display_name']} ({top_team['power_rating']:.2f})")
        print(f"{log_prefix}   Saved to: {output_path}")

        return True

    except Exception as e:
        print(f"{log_prefix} ERROR updating season {season}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update power ratings and save to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--season', type=int, default=2026,
        help='Season to update (default: 2026)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Update both 2025 and 2026 seasons'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress detailed output (only show errors)'
    )

    args = parser.parse_args()

    config = RatingConfig()

    if args.all:
        seasons = [2025, 2026]
    else:
        seasons = [args.season]

    success_count = 0
    for season in seasons:
        if update_season(season, config):
            success_count += 1

    if success_count == len(seasons):
        print(f"\nAll {len(seasons)} season(s) updated successfully.")
        sys.exit(0)
    else:
        print(f"\nWARNING: {len(seasons) - success_count} season(s) failed to update.")
        sys.exit(1)


if __name__ == "__main__":
    main()
