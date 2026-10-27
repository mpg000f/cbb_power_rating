"""Prepare compact, reviewable snapshots from downloaded ESPN/RSCI/SDV inputs.

Run collect_preseason_sources.py first. Source files
are cached to avoid unbounded repeated collection. No secret credentials needed.
"""
from pathlib import Path
from datetime import datetime, timezone
import json,re
import pandas as pd
from preseason_model import normalized_name

ROOT=Path(__file__).resolve().parent


def prepare(season=2027):
    if season!=2027:
        raise ValueError('Reviewed membership and availability policies currently cover 2027 only')
    out=ROOT/'preseason_inputs'/str(season)/'v2';out.mkdir(parents=True,exist_ok=True)
    ids=set(pd.read_csv(ROOT/'historical_ratings'/f'ratings_{season}.csv').team_id.astype(int))
    roster=[];coaches=[];timestamps=[]
    for tid in sorted(ids):
        path=ROOT/'research'/'espn'/f'roster_{season}_{tid}.json'
        d=json.loads(path.read_text())
        if d.get('season',{}).get('year')!=season:raise ValueError(f'{tid}: wrong roster season')
        if len(d['athletes'])<5:raise ValueError(f'{tid}: incomplete roster')
        timestamps.append(d.get('timestamp'))
        for a in d['athletes']:
            roster.append({'team_id':tid,'team_display_name':d['team']['displayName'],'college':d['team']['location'],'athlete_id':int(a['id']),'player':a['displayName'],'status':a.get('status',{}).get('name'),'experience':a.get('experience',{}).get('displayValue'),'source_url':f'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{tid}/roster?season={season}'})
        c=d.get('coach') or []
        if len(c)>1:raise ValueError(f'{tid}: multiple current head coaches')
        coaches.append({'team_id':tid,'coach_id':c[0]['id'] if c else None,'coach':' '.join([c[0].get('firstName',''),c[0].get('lastName','')]) if c else None,'source_url':f'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{tid}/roster?season={season}'})
    roster=pd.DataFrame(roster)
    if roster.athlete_id.duplicated().any():raise ValueError('Duplicate roster athlete IDs')
    roster.drop(columns='source_url').to_csv(out/'roster.csv',index=False)
    history=pd.read_csv(ROOT/'research'/'coach_history_raw.csv')
    if history.get('error',pd.Series(dtype=str)).notna().any():
        raise ValueError('Historical coach requests failed; retry before preparing sources')
    history=history[['season','team_id','coach_id','coach_count']]
    history.to_csv(out/'coach_history.csv',index=False)
    coaches=pd.DataFrame(coaches)
    # ESPN lacks this current coach field. Official Northeastern roster identifies
    # Bill Coen; use the same identity as his historical ESPN record.
    old=history[(history.season==2026)&(history.team_id==2447)&(history.coach_count==1)]
    coen_id=old.iloc[0].coach_id if len(old)==1 else None
    coaches.loc[coaches.team_id==2447,['coach_id','coach','source_url']]=[coen_id,'Bill Coen','https://nuhuskies.com/sports/mens-basketball/roster']
    coaches.to_csv(out/'coaches.csv',index=False)
    players=pd.read_parquet(ROOT/'research'/'raw'/f'value_{season-1}.parquet')
    prior_ids=set(pd.read_csv(ROOT/'historical_ratings'/f'ratings_{season-1}.csv').team_id.astype(int))
    players=players[players.team_id.astype(int).isin(prior_ids)]
    players.to_csv(out/'player_production.csv',index=False)
    # Retain the original RSCI table, including unresolved/unrostered prospects.
    recruits=pd.read_csv(ROOT/'research'/'rsci2026.csv')
    recruits.to_csv(out/'recruits.csv',index=False)
    aliases={normalized_name('Cam Williams'):5210407,normalized_name('Obina Ekezie Jr.'):5210950,normalized_name('Anthony Brown'):5195075,normalized_name('Aiden Chronister'):5210405,normalized_name('Trent Perry'):5142733}
    for name,aid in aliases.items():
        if aid not in set(roster.athlete_id):raise ValueError(f'Recruit alias not on current roster: {name}')
    (out/'recruit_aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
    pd.DataFrame([
        {'athlete_id':5142728,'player':'Brandon McCoy Jr.','team_id':130,'disposition':'exclude','reason':'Confirmed out for the full 2026-27 season (ACL)','verified_date':'2026-10-27','source_url':'https://mgoblue.com/news/2026/9/1/mens-basketball-mccoy-jr-out-for-2026-27-season-with-acl-injury'},
        {'athlete_id':5458626,'player':'Josh Manchester','team_id':275,'disposition':'exclude','reason':'Confirmed out for the full 2026-27 season (shoulder surgery); UW Stats & Info announcement quoted by WMTV','verified_date':'2026-10-27','source_url':'https://www.wmtv15news.com/2026/10/07/badger-mens-basketball-manchester-out-season/'}
    ]).to_csv(out/'availability.csv',index=False)
    manifest={'season':season,'retrieved_at':datetime.now(timezone.utc).isoformat(),'roster_source_timestamps':{'earliest':min(x for x in timestamps if x),'latest':max(x for x in timestamps if x)},'player_production':f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/mbb_player_value/mbb_player_value_{season-1}.parquet','player_metric_definition':'SportsDataverse team/opponent-adjusted box BPM. Not Sports-Reference BPM and not a causal measure.','recruiting':'https://sites.google.com/site/rscihoops/home/rankings/2026-final','recruiting_csv':'https://docs.google.com/spreadsheets/d/1neBv9U0fBIbMbWLh0v3oZ_7XX9BGGj5d3jp7VtI7DSk/export?format=csv','coach_history_template':'https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/{season}/teams/{team_id}/coaches?limit=10','coach_history_seasons':[2020,2021,2022,2023,2024,2025,2026],'availability_policy':'Confirmed full-season exclusions only. No complete injury/eligibility census is claimed. Unresolved eligibility follows roster inclusion and is an explicit remaining limitation.','recruit_alias_policy':'Five reviewed athlete-ID matches; no automatic fuzzy matching. College commitment is not used to override current roster membership.','player_development':'disabled: no age/class improvement term','roster_url_template':'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/roster?season=2027'}
    (out/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared {len(roster)} roster players and {len(coaches)} coaches')


if __name__=='__main__':prepare()
