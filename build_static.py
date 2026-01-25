#!/usr/bin/env python3
"""
Build static JSON files from CSV ratings for GitHub Pages.

This script generates:
- docs/data/ratings_{season}.json for each season
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
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"


def csv_to_json(csv_path: Path) -> list:
    """Convert a ratings CSV to a list of dicts for JSON."""
    df = pd.read_csv(csv_path)

    # Select and rename columns for the frontend
    columns = {
        'rank': 'rank',
        'team_display_name': 'team',
        'power_rating': 'rating',
        'adj_off_eff': 'adjO',
        'adj_def_eff': 'adjD',
        'tempo': 'tempo',
        'games_played': 'games',
        'sos': 'sos',
    }

    available = [c for c in columns.keys() if c in df.columns]
    result = df[available].rename(columns=columns)

    # Round numeric columns
    for col in ['rating', 'adjO', 'adjD', 'tempo', 'sos']:
        if col in result.columns:
            result[col] = result[col].round(2)

    return result.to_dict(orient='records')


def build_static_site():
    """Build the complete static site."""
    print("Building static site...")

    # Create directories
    DOCS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # Find all rating files and convert to JSON
    seasons = []
    for csv_file in sorted(RATINGS_DIR.glob("ratings_*.csv")):
        try:
            season = int(csv_file.stem.split("_")[1])
            seasons.append(season)

            # Convert to JSON
            ratings = csv_to_json(csv_file)
            json_path = DATA_DIR / f"ratings_{season}.json"

            with open(json_path, 'w') as f:
                json.dump({
                    'season': season,
                    'count': len(ratings),
                    'ratings': ratings
                }, f)

            print(f"  Generated {json_path.name} ({len(ratings)} teams)")

        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

    # Generate seasons list
    seasons_path = DATA_DIR / "seasons.json"
    with open(seasons_path, 'w') as f:
        json.dump({'seasons': sorted(seasons, reverse=True)}, f)
    print(f"  Generated seasons.json ({len(seasons)} seasons)")

    # Copy static HTML
    src_html = PROJECT_ROOT / "website" / "static" / "index.html"
    if src_html.exists():
        # Read and modify HTML for static hosting
        html_content = src_html.read_text()

        # Update API paths to use static JSON files
        html_content = html_content.replace(
            "fetch('/api/seasons')",
            "fetch('data/seasons.json')"
        )
        html_content = html_content.replace(
            "fetch(`/api/ratings/${season}`)",
            "fetch(`data/ratings_${season}.json`)"
        )

        # Write modified HTML
        dest_html = DOCS_DIR / "index.html"
        dest_html.write_text(html_content)
        print(f"  Generated index.html")

    print(f"\nStatic site built in {DOCS_DIR}/")
    print(f"To test locally: cd docs && python -m http.server 8000")


if __name__ == "__main__":
    build_static_site()
