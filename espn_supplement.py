"""
ESPN API supplement for recent CBB game data.

Sportsdataverse boxscore data lags 1-2 days. This module fetches missing
recent games from ESPN's public API to fill the gap. Only activates for
the current 2026 season.
"""

import time
import warnings
from datetime import datetime, timedelta

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary"
)

REQUEST_DELAY_SECONDS = 0.3

# ESPN summary stat names we need, mapped to our split keys
ESPN_SPLIT_STATS = {
    'fieldGoalsMade-fieldGoalsAttempted': ('field_goals_made', 'field_goals_attempted'),
    'threePointFieldGoalsMade-threePointFieldGoalsAttempted': (
        'three_point_field_goals_made', 'three_point_field_goals_attempted'
    ),
    'freeThrowsMade-freeThrowsAttempted': ('free_throws_made', 'free_throws_attempted'),
}

ESPN_DIRECT_STATS = {
    'offensiveRebounds': 'offensive_rebounds',
    'defensiveRebounds': 'defensive_rebounds',
    'totalRebounds': 'total_rebounds',
    'turnovers': 'turnovers',
}


def _fetch_scoreboard(date):
    """Fetch completed D1 games from ESPN scoreboard for a given date."""
    date_str = date.strftime('%Y%m%d')
    resp = requests.get(
        SCOREBOARD_URL,
        params={'dates': date_str, 'groups': '50', 'limit': '500'},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    completed = []
    for event in data.get('events', []):
        comp = event.get('competitions', [{}])[0]
        status = comp.get('status', {}).get('type', {})
        if status.get('completed', False):
            completed.append(event)
    return completed


def _extract_schedule_rows(events):
    """Extract schedule-format rows from scoreboard events."""
    rows = []
    for event in events:
        game_id = int(event['id'])
        comp = event['competitions'][0]
        competitors = comp.get('competitors', [])

        home, away = None, None
        for c in competitors:
            if c.get('homeAway') == 'home':
                home = c
            elif c.get('homeAway') == 'away':
                away = c

        if home is None or away is None:
            continue

        rows.append({
            'game_id': game_id,
            'home_id': int(home['team']['id']),
            'away_id': int(away['team']['id']),
            'home_score': int(home.get('score', 0)),
            'away_score': int(away.get('score', 0)),
        })
    return rows


def _parse_summary_stats(summary_data, game_id, game_date):
    """Parse ESPN summary response into sportsdataverse-format boxscore rows."""
    boxscore = summary_data.get('boxscore', {})
    header = summary_data.get('header', {})

    # Get scores and home/away from header competitors
    header_comp = header.get('competitions', [{}])[0]
    header_competitors = header_comp.get('competitors', [])

    score_map = {}  # team_id -> {score, homeAway, winner}
    for c in header_competitors:
        tid = int(c.get('id', 0))
        score_map[tid] = {
            'score': int(c.get('score', 0)),
            'homeAway': c.get('homeAway', ''),
            'winner': c.get('winner', False),
        }

    # Figure out opponent mapping
    team_ids = list(score_map.keys())
    if len(team_ids) != 2:
        return []
    opp_map = {team_ids[0]: team_ids[1], team_ids[1]: team_ids[0]}

    rows = []
    for team_data in boxscore.get('teams', []):
        tid = int(team_data.get('team', {}).get('id', 0))
        if tid not in score_map:
            continue

        info = score_map[tid]
        opp_id = opp_map[tid]
        opp_info = score_map[opp_id]

        # Parse stats from summary
        stats = {}
        for stat_entry in team_data.get('statistics', []):
            name = stat_entry.get('name', '')
            val = stat_entry.get('displayValue', '')

            if name in ESPN_SPLIT_STATS:
                made_col, att_col = ESPN_SPLIT_STATS[name]
                parts = val.split('-')
                if len(parts) == 2:
                    stats[made_col] = int(parts[0])
                    stats[att_col] = int(parts[1])

            elif name in ESPN_DIRECT_STATS:
                col = ESPN_DIRECT_STATS[name]
                try:
                    stats[col] = int(val)
                except (ValueError, TypeError):
                    stats[col] = 0

        row = {
            'game_id': game_id,
            'season': 2026,
            'game_date': pd.Timestamp(game_date),
            'team_id': tid,
            'opponent_team_id': opp_id,
            'team_home_away': info['homeAway'],
            'team_score': info['score'],
            'opponent_team_score': opp_info['score'],
            'team_winner': info['winner'],
            'team_display_name': team_data.get('team', {}).get('displayName', ''),
        }
        row.update(stats)
        rows.append(row)

    return rows


def fetch_espn_supplement(boxscores_df, current_season):
    """Fetch recent games from ESPN that are missing from sportsdataverse.

    Returns (boxscore_supplement_df, schedule_supplement_df). Both may be
    empty DataFrames if there's nothing to supplement.
    """
    if current_season != 2026:
        return pd.DataFrame(), pd.DataFrame()

    if requests is None:
        warnings.warn("requests package not installed, ESPN supplement disabled")
        return pd.DataFrame(), pd.DataFrame()

    # Find the latest date in sportsdataverse data
    if 'game_date' not in boxscores_df.columns or boxscores_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = pd.to_datetime(boxscores_df['game_date'], errors='coerce')
    max_date = dates.max()
    if pd.isna(max_date):
        return pd.DataFrame(), pd.DataFrame()

    max_date = max_date.normalize()  # strip time component
    today = pd.Timestamp(datetime.now().date())

    # Nothing to supplement if sportsdataverse is up to date
    if max_date >= today:
        return pd.DataFrame(), pd.DataFrame()

    existing_game_ids = set(boxscores_df['game_id'].dropna().astype(int).unique())

    all_schedule_rows = []
    new_game_ids = []
    game_date_map = {}  # game_id -> date

    # Fetch scoreboard for each missing date
    current_date = max_date + timedelta(days=1)
    while current_date <= today:
        try:
            events = _fetch_scoreboard(current_date)
            schedule_rows = _extract_schedule_rows(events)
            all_schedule_rows.extend(schedule_rows)

            for row in schedule_rows:
                gid = row['game_id']
                if gid not in existing_game_ids:
                    new_game_ids.append(gid)
                    game_date_map[gid] = current_date

        except Exception as e:
            warnings.warn(f"ESPN scoreboard fetch failed for {current_date.date()}: {e}")

        current_date += timedelta(days=1)

    # Build schedule supplement
    schedule_supplement = pd.DataFrame(all_schedule_rows) if all_schedule_rows else pd.DataFrame()

    if not new_game_ids:
        print("  ESPN supplement: no new games to fetch")
        return pd.DataFrame(), schedule_supplement

    # Fetch boxscore details for each new game
    boxscore_rows = []
    fetched = 0
    failed = 0
    for gid in new_game_ids:
        try:
            resp = requests.get(
                SUMMARY_URL,
                params={'event': str(gid)},
                timeout=15,
            )
            resp.raise_for_status()
            summary = resp.json()
            rows = _parse_summary_stats(summary, gid, game_date_map[gid])
            boxscore_rows.extend(rows)
            fetched += 1
        except Exception as e:
            warnings.warn(f"ESPN summary fetch failed for game {gid}: {e}")
            failed += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    boxscore_supplement = pd.DataFrame(boxscore_rows) if boxscore_rows else pd.DataFrame()

    added = len(boxscore_supplement)
    print(f"  ESPN supplement: added {added} team-game rows "
          f"({fetched} games fetched, {failed} failed)")

    return boxscore_supplement, schedule_supplement
