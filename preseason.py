"""Build a frozen preseason baseline from prior ratings and preseason ESPN rosters.

Season is the END year: --season 2027 means 2026-27. No current-season
boxscores are used. Roster and player-minute snapshots are retained for audit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
import pandas as pd
from power_rating import RatingConfig, calculate_preseason_projection, get_conference_tier, get_transfer_scale

ROOT = Path(__file__).resolve().parent


def read_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def roster(team_id, season):
    url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/roster?season={season}'
    data = read_json(url)
    if data.get('season', {}).get('year') != season:
        raise ValueError(f'{team_id}: roster season does not match {season}')
    athletes = data.get('athletes', [])
    ids = [int(a['id']) for a in athletes]
    if len(ids) < 5 or len(ids) != len(set(ids)):
        raise ValueError(f'{team_id}: incomplete or duplicate roster')
    return {'team_id': team_id, 'athletes': [{'id': int(a['id']), 'name': a.get('displayName')} for a in athletes]}


def build(season, refresh=False):
    directory = ROOT / 'preseason_inputs' / str(season)
    directory.mkdir(parents=True, exist_ok=True)
    config = RatingConfig()
    prior = pd.read_csv(ROOT / 'historical_ratings' / f'ratings_{season-1}.csv')
    snapshots = directory / 'rosters.json'
    if snapshots.exists() and not refresh:
        payload = json.loads(snapshots.read_text())
        if payload['season'] != season:
            raise ValueError('Wrong season in cached rosters')
    else:
        teams = read_json('https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=1000')['sports'][0]['leagues'][0]['teams']
        # ESPN's team directory omits some active D1 teams; include prior D1 IDs.
        ids = set(prior.team_id.astype(int)) | {int(t['team']['id']) for t in teams}
        ids.discard(3229)  # Virginia-Lynchburg is not D1.
        records, errors = [], []
        with ThreadPoolExecutor(max_workers=12) as pool:
            pending = {pool.submit(roster, tid, season): tid for tid in sorted(ids)}
            for task in as_completed(pending):
                try:
                    records.append(task.result())
                except Exception as exc:
                    errors.append(f'{pending[task]}: {exc}')
        if errors:
            raise ValueError('Roster fetch failed; baseline not written: ' + '; '.join(errors))
        payload = {'season': season, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'source': 'ESPN season-specific roster API', 'teams': sorted(records, key=lambda r: r['team_id'])}
        snapshots.write_text(json.dumps(payload, indent=2) + '\n')
    minutes_path = directory / 'prior_player_minutes.csv'
    if not minutes_path.exists():
        url = f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_player_boxscores/player_box_{season-1}.parquet'
        temp = directory / 'players.parquet'
        urllib.request.urlretrieve(url, temp)
        players = pd.read_parquet(temp)
        if set(players.season.unique()) != {season-1}:
            raise ValueError('Wrong season in prior player data')
        players.groupby(['team_id', 'athlete_id'], as_index=False)['minutes'].sum().to_csv(minutes_path, index=False)
        temp.unlink()
    minutes = pd.read_csv(minutes_path)
    if minutes.duplicated(['team_id', 'athlete_id']).any() or (minutes.minutes < 0).any():
        raise ValueError('Invalid prior minutes')
    totals = minutes.groupby('team_id').minutes.sum()
    by_player = {int(pid): g for pid, g in minutes.groupby('athlete_id')}
    all_roster_ids = [a['id'] for t in payload['teams'] for a in t['athletes']]
    if len(all_roster_ids) != len(set(all_roster_ids)):
        raise ValueError('A player appears on multiple current rosters; resolve before publishing')
    # Saint Francis (PA) left Division I for 2026-27. West Florida joins;
    # without D1 history it starts at the median of last year's bottom quartile.
    expansion = {2697: 'West Florida Argonauts'} if season == 2027 else {}
    departed = {2598} if season == 2027 else set()
    expansion_rating = prior.loc[prior.power_rating <= prior.power_rating.quantile(.25), 'power_rating'].median()
    for tid, name in expansion.items():
        if tid not in set(prior.team_id):
            prior = pd.concat([prior, pd.DataFrame([{'team_id': tid, 'team_display_name': name, 'power_rating': expansion_rating, 'tempo': prior.tempo.mean()}])], ignore_index=True)
    records = []
    for team in payload['teams']:
        if team['team_id'] in departed:
            continue
        tid = team['team_id']
        denom = totals.get(tid, 0)
        if denom <= 0 and tid in expansion:
            records.append({'team_id': tid, 'returning_pct': None, 'transfer_in_pct_raw': None, 'transfer_in_pct': None, 'effective_returning': config.preseason_returning_avg})
            continue
        if denom <= 0:
            raise ValueError(f'{tid}: missing prior team minutes; explicit expansion-team prior required')
        retained, incoming, scaled = 0., 0., 0.
        for athlete in team['athletes']:
            for row in by_player.get(athlete['id'], pd.DataFrame()).itertuples():
                if row.team_id == tid:
                    retained += row.minutes
                else:
                    incoming += row.minutes
                    scaled += row.minutes * get_transfer_scale(get_conference_tier(row.team_id), get_conference_tier(tid), config)
        records.append({'team_id': tid, 'returning_pct': 100*retained/denom, 'transfer_in_pct_raw': 100*incoming/denom, 'transfer_in_pct': 100*scaled/denom, 'effective_returning': min(100., 100*(retained+scaled)/denom)})
    production = pd.DataFrame(records)
    ids = set(production.team_id)
    missing = ids - set(prior.team_id)
    if missing:
        raise ValueError(f'Teams missing prior ratings: {sorted(missing)}')
    prior = prior[prior.team_id.isin(ids)].copy()
    projection = calculate_preseason_projection(prior, production, config)
    projection.loc[projection.team_id.isin(expansion), 'preseason_rating'] = expansion_rating
    projection['preseason_rating'] -= projection.preseason_rating.mean()
    result = projection.merge(production.drop(columns='effective_returning'), on='team_id', validate='one_to_one')
    result = result.merge(prior[['team_id', 'tempo']], on='team_id', validate='one_to_one')
    result['season'] = season
    result['power_rating'] = result.preseason_rating
    result['prior_weight'] = 1.
    result['games_played'] = result['wins'] = result['losses'] = 0
    result['record'] = '0-0'
    # Offensive/defensive splits are intentionally absent: roster continuity
    # alone cannot identify which side of the ball changed.
    result['rating_phase'] = 'preseason'
    result['prior_source'] = np.where(result.team_id.isin(expansion), 'expansion: median of prior bottom quartile', 'prior final rating + roster continuity')
    result = result.sort_values(['power_rating', 'team_id'], ascending=[False, True]).reset_index(drop=True)
    result['rank'] = np.arange(1, len(result)+1)
    current = ROOT / 'historical_ratings' / f'ratings_{season}.csv'
    if current.exists() and pd.read_csv(current).games_played.max() > 0:
        raise ValueError('Refusing to overwrite in-season ratings with preseason ratings')
    result.to_csv(ROOT / 'historical_ratings' / f'ratings_{season}_preseason.csv', index=False)
    result.to_csv(current, index=False)
    print(result[['rank', 'team_display_name', 'power_rating', 'effective_returning']].head(25).to_string(index=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2027)
    parser.add_argument('--refresh-inputs', action='store_true')
    args = parser.parse_args()
    build(args.season, args.refresh_inputs)
