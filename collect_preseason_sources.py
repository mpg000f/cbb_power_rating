"""Download the public sources used by the 2027 production-model candidate.

Responses are cached individually for resumable collection. This explicitly
collects current roster membership and prior team/coach/player data, not target-
season results. Refreshing inputs requires removing the relevant cache files.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
import urllib.request
import pandas as pd

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'research'/'espn'


def fetch_json(url,path):
    if path.exists():return json.loads(path.read_text())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=20) as response:result=json.load(response)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result))
            return result
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)


def collect(season=2027):
    if season!=2027:raise ValueError('Recruiting source and reviewed policies currently cover 2027 only')
    CACHE.mkdir(parents=True,exist_ok=True)
    jobs=[('roster',season,int(t)) for t in pd.read_csv(ROOT/'historical_ratings'/f'ratings_{season}.csv').team_id]
    for year in range(season-7,season):
        jobs.extend(('coach',year,int(t)) for t in pd.read_csv(ROOT/'historical_ratings'/f'ratings_{year}.csv').team_id)
    def work(job):
        kind,year,tid=job
        if kind=='roster':url=f'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{tid}/roster?season={year}'
        else:url=f'https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/{year}/teams/{tid}/coaches?limit=10'
        data=fetch_json(url,CACHE/f'{kind}_{year}_{tid}.json')
        if kind=='roster':
            if data.get('season',{}).get('year')!=year:raise ValueError(f'{tid}: wrong roster season')
            return None
        ids=[re.search(r'/coaches/(\d+)',x['$ref']).group(1) for x in data.get('items',[])]
        return {'season':year,'team_id':tid,'coach_id':ids[0] if len(ids)==1 else None,'coach_count':len(ids)}
    records=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(work,job) for job in jobs]
        for i,future in enumerate(as_completed(futures)):
            result=future.result()
            if result is not None:records.append(result)
            if (i+1)%100==0:print(f'{i+1}/{len(jobs)} responses cached',flush=True)
    pd.DataFrame(records).sort_values(['season','team_id']).to_csv(ROOT/'research'/'coach_history_raw.csv',index=False)
    raw=ROOT/'research'/'raw';raw.mkdir(exist_ok=True)
    values=raw/f'value_{season-1}.parquet'
    if not values.exists():urllib.request.urlretrieve(f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/mbb_player_value/mbb_player_value_{season-1}.parquet',values)
    # RSCI publishes its final rankings through a public sheet embedded here.
    recruiting_path=ROOT/'research'/'rsci2026.csv'
    if not recruiting_path.exists():
        url='https://sites.google.com/site/rscihoops/home/rankings/2026-final'
        with urllib.request.urlopen(url,timeout=30) as response:html=response.read().decode()
        doc=re.search(r'data-embed-doc-id="([^"]+)"',html)
        if not doc:raise ValueError('RSCI final page has no public spreadsheet')
        urllib.request.urlretrieve(f'https://docs.google.com/spreadsheets/d/{doc.group(1)}/export?format=csv',recruiting_path)
    print('Source collection complete')


if __name__=='__main__':collect()
