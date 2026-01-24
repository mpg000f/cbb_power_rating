"""
Generate historical power ratings for past seasons.
Saves each year's ratings to the historical_ratings folder.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from power_rating import run_power_ratings, RatingConfig, display_ratings

def generate_historical_ratings(start_year: int = 2020, end_year: int = 2024):
    """Generate ratings for multiple past seasons."""

    output_dir = Path(__file__).parent / "historical_ratings"
    output_dir.mkdir(exist_ok=True)

    config = RatingConfig()

    for season in range(start_year, end_year + 1):
        print("\n" + "=" * 80)
        print(f"GENERATING RATINGS FOR {season} SEASON")
        print("=" * 80 + "\n")

        try:
            # Run without priors for historical data (we don't have prior exports for past years)
            results = run_power_ratings(
                current_season=season,
                config=config,
                use_r_data=False  # No prior data for historical seasons
            )

            # Save to CSV
            output_file = output_dir / f"ratings_{season}.csv"
            results.to_csv(output_file, index=False)
            print(f"\nSaved {len(results)} team ratings to {output_file}")

            # Display top 25
            display_ratings(results, top_n=25)

        except Exception as e:
            print(f"ERROR generating ratings for {season}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("HISTORICAL RATINGS GENERATION COMPLETE")
    print(f"Files saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate historical power ratings")
    parser.add_argument('--start', type=int, default=2020, help='Start season (default: 2020)')
    parser.add_argument('--end', type=int, default=2024, help='End season (default: 2024)')

    args = parser.parse_args()

    generate_historical_ratings(start_year=args.start, end_year=args.end)
