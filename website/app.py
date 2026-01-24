"""
Power Ratings API
Simple FastAPI backend to serve college basketball power ratings.
"""

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

app = FastAPI(title="Power Ratings API", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
RATINGS_DIR = BASE_DIR / "historical_ratings"


def load_ratings(season: int) -> pd.DataFrame:
    """Load ratings for a given season."""
    path = RATINGS_DIR / f"ratings_{season}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.sort_values('power_rating', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


@app.get("/")
async def home():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/seasons")
async def get_seasons():
    """Get list of available seasons."""
    seasons = []
    for f in RATINGS_DIR.glob("ratings_*.csv"):
        try:
            year = int(f.stem.split("_")[1])
            seasons.append(year)
        except:
            pass
    return {"seasons": sorted(seasons, reverse=True)}


@app.get("/api/ratings/{season}")
async def get_ratings(
    season: int,
    limit: Optional[int] = Query(None, description="Limit number of results"),
    search: Optional[str] = Query(None, description="Search team name")
):
    """Get power ratings for a season."""
    df = load_ratings(season)

    if df.empty:
        return {"error": f"No ratings found for {season}", "ratings": []}

    # Search filter
    if search:
        mask = df['team_display_name'].str.lower().str.contains(search.lower(), na=False)
        df = df[mask]

    # Limit results
    if limit:
        df = df.head(limit)

    # Select and rename columns for API response
    columns = {
        'rank': 'rank',
        'team_display_name': 'team',
        'power_rating': 'rating',
        'adj_off_eff': 'adjO',
        'adj_def_eff': 'adjD',
        'tempo': 'tempo',
        'games_played': 'games',
        'sos': 'sos',
        'consistency_score': 'consistency'
    }

    available_cols = [c for c in columns.keys() if c in df.columns]
    result = df[available_cols].rename(columns=columns)

    # Round numeric columns
    for col in ['rating', 'adjO', 'adjD', 'tempo', 'sos', 'consistency']:
        if col in result.columns:
            result[col] = result[col].round(2)

    return {
        "season": season,
        "count": len(result),
        "updated": datetime.now().isoformat(),
        "ratings": result.to_dict(orient='records')
    }


@app.get("/api/team/{team_name}")
async def get_team_history(team_name: str):
    """Get historical ratings for a team across all seasons."""
    history = []

    for f in sorted(RATINGS_DIR.glob("ratings_*.csv")):
        try:
            season = int(f.stem.split("_")[1])
            df = load_ratings(season)

            # Find team (case-insensitive partial match)
            mask = df['team_display_name'].str.lower().str.contains(team_name.lower(), na=False)
            team_data = df[mask]

            if len(team_data) > 0:
                row = team_data.iloc[0]
                history.append({
                    "season": season,
                    "rank": int(row['rank']),
                    "rating": round(row['power_rating'], 2),
                    "adjO": round(row['adj_off_eff'], 2),
                    "adjD": round(row['adj_def_eff'], 2),
                    "games": int(row['games_played'])
                })
        except:
            pass

    return {
        "team": team_name,
        "history": sorted(history, key=lambda x: x['season'], reverse=True)
    }


@app.get("/api/spread")
async def calculate_spread(
    team_a: str,
    team_b: str,
    season: int = 2026,
    home: Optional[str] = Query(None, description="Which team is home (a or b)")
):
    """Calculate predicted spread between two teams."""
    df = load_ratings(season)

    if df.empty:
        return {"error": f"No ratings found for {season}"}

    # Find teams
    mask_a = df['team_display_name'].str.lower().str.contains(team_a.lower(), na=False)
    mask_b = df['team_display_name'].str.lower().str.contains(team_b.lower(), na=False)

    if not mask_a.any():
        return {"error": f"Team not found: {team_a}"}
    if not mask_b.any():
        return {"error": f"Team not found: {team_b}"}

    row_a = df[mask_a].iloc[0]
    row_b = df[mask_b].iloc[0]

    # Calculate spread
    rating_diff = row_a['power_rating'] - row_b['power_rating']
    avg_tempo = (row_a['tempo'] + row_b['tempo']) / 2
    tempo_factor = avg_tempo / 100

    raw_spread = rating_diff * tempo_factor

    # Home court adjustment (~3.5 points)
    home_advantage = 3.5 * tempo_factor
    if home == 'a':
        spread = raw_spread + home_advantage
    elif home == 'b':
        spread = raw_spread - home_advantage
    else:
        spread = raw_spread  # Neutral

    return {
        "team_a": row_a['team_display_name'],
        "team_b": row_b['team_display_name'],
        "team_a_rating": round(row_a['power_rating'], 2),
        "team_b_rating": round(row_b['power_rating'], 2),
        "predicted_spread": round(spread, 1),
        "home": home or "neutral",
        "expected_tempo": round(avg_tempo, 1)
    }


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
