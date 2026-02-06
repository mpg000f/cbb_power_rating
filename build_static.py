#!/usr/bin/env python3
"""
Build static JSON files from CSV ratings for GitHub Pages.

This script generates:
- docs/data/ratings_{season}.json for each CBB season
- docs/data/cfb/ratings_{season}.json for each CFB season
- docs/data/seasons.json with list of available seasons
- Copies index.html to docs/

Run this after updating ratings to regenerate the static site.
"""

import json
import pandas as pd
from pathlib import Path
import shutil

# Paths
PROJECT_ROOT = Path(__file__).parent
RATINGS_DIR = PROJECT_ROOT / "historical_ratings"
CFB_RATINGS_DIR = PROJECT_ROOT.parent / "cfb_power_rating" / "historical_ratings"
NFL_RATINGS_DIR = PROJECT_ROOT.parent / "nfl_power_rating" / "historical_ratings"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
CFB_DATA_DIR = DATA_DIR / "cfb"
NFL_DATA_DIR = DATA_DIR / "nfl"


def csv_to_json(csv_path: Path, sport: str = "cbb") -> list:
    """Convert a ratings CSV to a list of dicts for JSON."""
    df = pd.read_csv(csv_path)

    # Select and rename columns for the frontend (different for each sport)
    if sport == "cfb":
        columns = {
            'rank': 'rank',
            'team': 'team',
            'power_rating': 'rating',
            'record': 'record',
            'off_rating': 'adjO',
            'def_rating': 'adjD',
            'games': 'games',
        }
    elif sport == "nfl":
        columns = {
            'rank': 'rank',
            'team': 'team',
            'power_rating': 'rating',
            'record': 'record',
            # Adjusted stats
            'adj_off_epa': 'adjOffEpa',
            'adj_def_epa': 'adjDefEpa',
            'adj_off_success': 'adjOffSuccess',
            'adj_def_success': 'adjDefSuccess',
            # Raw stats
            'raw_off_epa': 'rawOffEpa',
            'raw_def_epa': 'rawDefEpa',
            'raw_off_success': 'rawOffSuccess',
            'raw_def_success': 'rawDefSuccess',
            'games_played': 'games',
        }
    else:  # cbb
        columns = {
            'rank': 'rank',
            'team_display_name': 'team',
            'power_rating': 'rating',
            'record': 'record',
            'adj_off_eff': 'adjO',
            'adj_def_eff': 'adjD',
            'tempo': 'tempo',
            'games_played': 'games',
            'sos': 'sos',
        }

    available = [c for c in columns.keys() if c in df.columns]
    result = df[available].rename(columns=columns)

    # Round numeric columns
    for col in ['rating', 'adjO', 'adjD', 'tempo', 'sos',
                'adjOffEpa', 'adjDefEpa', 'adjOffSuccess', 'adjDefSuccess',
                'rawOffEpa', 'rawDefEpa', 'rawOffSuccess', 'rawDefSuccess']:
        if col in result.columns:
            result[col] = result[col].round(3)

    # Replace NaN with None for valid JSON (NaN is not valid JSON)
    result = result.where(pd.notna(result), None)

    return result.to_dict(orient='records')


def build_static_site():
    """Build the complete static site."""
    print("Building static site...")

    # Create directories
    DOCS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    CFB_DATA_DIR.mkdir(exist_ok=True)

    # ===== CBB Ratings =====
    print("\n  College Basketball:")
    seasons = []
    for csv_file in sorted(RATINGS_DIR.glob("ratings_*.csv")):
        try:
            season = int(csv_file.stem.split("_")[1])
            seasons.append(season)

            # Convert to JSON
            ratings = csv_to_json(csv_file, sport="cbb")
            json_path = DATA_DIR / f"ratings_{season}.json"

            with open(json_path, 'w') as f:
                json.dump({
                    'season': season,
                    'count': len(ratings),
                    'ratings': ratings
                }, f)

            print(f"    Generated {json_path.name} ({len(ratings)} teams)")

        except Exception as e:
            print(f"    Error processing {csv_file.name}: {e}")

    # Generate seasons list
    seasons_path = DATA_DIR / "seasons.json"
    with open(seasons_path, 'w') as f:
        json.dump({'seasons': sorted(seasons, reverse=True)}, f)
    print(f"    Generated seasons.json ({len(seasons)} seasons)")

    # ===== CFB Ratings =====
    if CFB_RATINGS_DIR.exists():
        print("\n  College Football:")
        cfb_seasons = []
        for csv_file in sorted(CFB_RATINGS_DIR.glob("ratings_*.csv")):
            try:
                season = int(csv_file.stem.split("_")[1])
                cfb_seasons.append(season)

                # Convert to JSON
                ratings = csv_to_json(csv_file, sport="cfb")
                json_path = CFB_DATA_DIR / f"ratings_{season}.json"

                with open(json_path, 'w') as f:
                    json.dump({
                        'season': season,
                        'count': len(ratings),
                        'ratings': ratings
                    }, f)

                print(f"    Generated cfb/{json_path.name} ({len(ratings)} teams)")

            except Exception as e:
                print(f"    Error processing {csv_file.name}: {e}")

        # Generate CFB seasons list
        cfb_seasons_path = CFB_DATA_DIR / "seasons.json"
        with open(cfb_seasons_path, 'w') as f:
            json.dump({'seasons': sorted(cfb_seasons, reverse=True)}, f)
        print(f"    Generated cfb/seasons.json ({len(cfb_seasons)} seasons)")
    else:
        print(f"\n  CFB ratings not found at {CFB_RATINGS_DIR}")

    # ===== NFL Ratings =====
    NFL_DATA_DIR.mkdir(exist_ok=True)
    if NFL_RATINGS_DIR.exists():
        print("\n  NFL:")
        nfl_seasons = []
        for csv_file in sorted(NFL_RATINGS_DIR.glob("ratings_*.csv")):
            try:
                season = int(csv_file.stem.split("_")[1])
                nfl_seasons.append(season)

                # Convert to JSON
                ratings = csv_to_json(csv_file, sport="nfl")
                json_path = NFL_DATA_DIR / f"ratings_{season}.json"

                with open(json_path, 'w') as f:
                    json.dump({
                        'season': season,
                        'count': len(ratings),
                        'ratings': ratings
                    }, f)

                print(f"    Generated nfl/{json_path.name} ({len(ratings)} teams)")

            except Exception as e:
                print(f"    Error processing {csv_file.name}: {e}")

        # Generate NFL seasons list
        nfl_seasons_path = NFL_DATA_DIR / "seasons.json"
        with open(nfl_seasons_path, 'w') as f:
            json.dump({'seasons': sorted(nfl_seasons, reverse=True)}, f)
        print(f"    Generated nfl/seasons.json ({len(nfl_seasons)} seasons)")
    else:
        print(f"\n  NFL ratings not found at {NFL_RATINGS_DIR}")

    # Copy static HTML
    src_html = PROJECT_ROOT / "website" / "static" / "index.html"
    if src_html.exists():
        # Read and modify HTML for static hosting
        html_content = src_html.read_text()

        # Update API paths to use static JSON files
        # CBB paths
        html_content = html_content.replace(
            "'/api/seasons/cbb'",
            "'data/seasons.json'"
        )
        html_content = html_content.replace(
            "'/api/ratings/cbb'",
            "'data'"
        )
        # CFB paths
        html_content = html_content.replace(
            "'/api/seasons/cfb'",
            "'data/cfb/seasons.json'"
        )
        html_content = html_content.replace(
            "'/api/ratings/cfb'",
            "'data/cfb'"
        )
        # NFL paths
        html_content = html_content.replace(
            "'/api/seasons/nfl'",
            "'data/nfl/seasons.json'"
        )
        html_content = html_content.replace(
            "'/api/ratings/nfl'",
            "'data/nfl'"
        )
        # Fix the ratings fetch pattern
        html_content = html_content.replace(
            "${basePath}/${season}",
            "${basePath}/ratings_${season}.json"
        )

        # Write modified HTML
        dest_html = DOCS_DIR / "index.html"
        dest_html.write_text(html_content)
        print(f"  Generated index.html")

    print(f"\nStatic site built in {DOCS_DIR}/")
    print(f"To test locally: cd docs && python -m http.server 8000")


if __name__ == "__main__":
    build_static_site()
