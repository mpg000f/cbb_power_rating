"""Retrospective preseason-only margin forecasts for early D1 games.

For each 2022-2026 target, tune regression on the previous five seasons only.
No target-year scores, tempo, final ratings or roster participation enter a
forecast. All families use the same games and the site's existing pace/HCA rule.
"""
from pathlib import Path
import urllib.request
import json
import numpy as np
import pandas as pd
from preseason_model import load_history
from evaluate_preseason_baselines import FAMILIES

ROOT=Path(__file__).resolve().parent


def previous_baseline(history,year,weights):
    ids=history[year-1].index
    numerator=pd.Series(0.,index=ids);denominator=numerator.copy()
    for lag,w in enumerate(weights,1):
        prior=history[year-lag].power_rating
        values=(prior-prior.mean()).reindex(ids)
        numerator+=values.fillna(0)*w;denominator+=values.notna()*w
    return numerator/denominator.replace(0,np.nan)


def early_games(schedule,eligible,limit=5):
    x=schedule.copy()
    x=x[(x.season_type==2)&x.status_type_completed.fillna(False)&(x.home_score>0)&(x.away_score>0)]
    x=x[x.home_id.isin(eligible)&x.away_id.isin(eligible)]
    if x.game_id.duplicated().any():raise ValueError('Duplicate game IDs')
    x=x.sort_values(['date','game_id'])
    appearances=pd.concat([x[['game_id','date','home_id']].rename(columns={'home_id':'team_id'}),x[['game_id','date','away_id']].rename(columns={'away_id':'team_id'})]).sort_values(['date','game_id'])
    appearances['number']=appearances.groupby('team_id').cumcount()+1
    counts=appearances.groupby('game_id').number.max()
    return x[x.game_id.map(counts)<=limit]


def evaluate():
    history=load_history(ROOT/'historical_ratings')
    cache=ROOT/'research/schedules';cache.mkdir(parents=True,exist_ok=True)
    schedules={}
    for year in range(2017,2027):
        p=cache/f'{year}.parquet'
        if not p.exists():urllib.request.urlretrieve(f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{year}.parquet',p)
        schedules[year]=pd.read_parquet(p)
        if set(schedules[year].season.unique())!={year}:raise ValueError('Wrong schedule season')
    evidence={};rows=[];predictions=[]
    for year,sched in schedules.items():
        prior=history[year-1];games=early_games(sched,set(prior.index));tempo=prior.tempo
        if games.neutral_site.isna().any():raise ValueError('Missing neutral site flag')
        poss=games.home_id.map(tempo)*games.away_id.map(tempo)/tempo.mean()
        if poss.isna().any():raise ValueError('Missing prior tempo')
        home=(~games.neutral_site.astype(bool)).astype(int)*3.5
        actual=games.home_score-games.away_score
        for name,w in FAMILIES.items():
            baseline=previous_baseline(history,year,w)
            differential=(games.home_id.map(baseline)-games.away_id.map(baseline))*poss/100
            evidence[name,year]={'d':differential.to_numpy(),'home':home.to_numpy(),'actual':actual.to_numpy(),'games':games}
    def loss(name,years,shrink):
        # Equal season weights in model selection, matching baseline review.
        return float(np.mean([np.sqrt(np.mean((evidence[name,y]['d']*shrink+evidence[name,y]['home']-evidence[name,y]['actual'])**2)) for y in years]))
    for year in range(2022,2027):
        training=list(range(year-5,year))
        fitted={name:min((loss(name,training,s),float(s)) for s in np.round(np.arange(.50,1.251,.01),2))[1] for name in FAMILIES}
        for name in FAMILIES:
            protocols=[('game_fitted',fitted[name])]
            if name=='three_year_70_20_10':
                protocols += [('v3_baseline_085',.85),('unshrunk',1.),('game_fitted_stretched_20pct',1.2*fitted[name])]
            e=evidence[name,year]
            for protocol,shrink in protocols:
                pred=e['d']*shrink+e['home'];err=pred-e['actual']
                rows.append({'family':name,'protocol':protocol,'season':year,'train_start':min(training),'train_end':max(training),'shrink':shrink,'games':len(pred),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(np.abs(err))),'winner_accuracy':float(np.mean((pred>0)==(e['actual']>0))),'mean_predicted_absolute_margin':float(np.mean(np.abs(pred)))})
                if protocol=='game_fitted':
                    for i,game in enumerate(e['games'].itertuples()):
                        predictions.append({'family':name,'season':year,'game_id':game.game_id,'date':str(game.date),'home_id':game.home_id,'away_id':game.away_id,'neutral_site':game.neutral_site,'predicted_home_margin':float(pred[i]),'actual_home_margin':float(e['actual'][i]),'regression':shrink,'training_end':max(training)})
    yearly=pd.DataFrame(rows)
    summary=yearly.groupby(['family','protocol'],as_index=False).agg(games=('games','sum'),mean_rmse=('rmse','mean'),mean_mae=('mae','mean'),mean_winner_accuracy=('winner_accuracy','mean'))
    out=ROOT/'preseason_candidates/2027/baseline_review'
    yearly.to_csv(out/'early_game_yearly.csv',index=False);summary.to_csv(out/'early_game_summary.csv',index=False);pd.DataFrame(predictions).to_csv(out/'early_game_predictions.csv',index=False)
    details={'target_seasons':[2022,2023,2024,2025,2026],'scope':'regular-season games where both teams are within their first five completed D1 games against prior-year-rated opponents; unranked/new D1 opponents omitted consistently','regression_selection':'minimum mean season margin RMSE over previous five seasons only, grid .50 to 1.25 by .01','pace':'prior-season home tempo * prior-season away tempo / prior-season national mean tempo','home_advantage':3.5,'unchanged_preseason_decay':'20 games, power 1.32; not applied here because the test isolates the preseason prior','limitations':['Retrospective comparison; recent historical years already inspected during project development','Tests history-only forecasts, not full roster/coaching/recruiting candidates','Home advantage and pace rule held fixed across models, not fitted to target outcomes','No odds/ATS or betting profitability test','Coach metadata and historical preseason rosters are not needed or used']}
    (out/'early_game_method.json').write_text(json.dumps(details,indent=2)+'\n')
    print(summary.to_string(index=False));print('\nYEARLY\n'+yearly.to_string(index=False))
    return yearly,summary


if __name__=='__main__':evaluate()
