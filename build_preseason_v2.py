"""Build a reviewable v2 candidate; does not change published ratings.

Sources must first be prepared with prepare_preseason_v2.py. The incumbent
baseline and football site files are never modified by this command.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from preseason_model import (load_history, calibrate_baseline, team_baseline,
                            production_components, recruiting_component,
                            coaching_component)

ROOT=Path(__file__).resolve().parent


def build(season=2027):
    inputs=ROOT/'preseason_inputs'/str(season)/'v2'
    out=ROOT/'preseason_candidates'/str(season)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'sources.json').read_text())
    if manifest['season']!=season:raise ValueError('Source season mismatch')
    roster=pd.read_csv(inputs/'roster.csv')
    coaches=pd.read_csv(inputs/'coaches.csv')
    history_coaches=pd.read_csv(inputs/'coach_history.csv')
    values=pd.read_csv(inputs/'player_production.csv')
    recruits=pd.read_csv(inputs/'recruits.csv')
    recruits.attrs['athlete_aliases']=json.loads((inputs/'recruit_aliases.json').read_text())
    availability=pd.read_csv(inputs/'availability.csv')
    excluded=set(availability.loc[availability.disposition=='exclude','athlete_id'].astype(int))
    if availability.athlete_id.duplicated().any():raise ValueError('Duplicate availability override')
    if not set(availability.athlete_id).issubset(set(roster.athlete_id)):raise ValueError('Availability override has no roster match')
    if availability.source_url.isna().any():raise ValueError('Availability decision requires a source')
    ids=sorted(roster.team_id.unique())
    if len(ids)!=len(coaches) or set(ids)!=set(coaches.team_id):raise ValueError('Coach roster universe mismatch')
    history=load_history(ROOT/'historical_ratings')
    if max(history)>=season:raise ValueError('Target season is in historical data; not a preseason run')
    config,validation=calibrate_baseline(history)
    production,audit=production_components(roster,values,history,season,config,excluded)
    recruiting,recruit_audit=recruiting_component(roster,recruits,config,excluded)
    coaching,coach_obs=coaching_component(history,history_coaches,coaches,season,config)
    components=team_baseline(history,season,ids,config).merge(production,on='team_id',validate='one_to_one').merge(coaching,on='team_id',validate='one_to_one')
    components['recruit_adjustment']=components.team_id.map(recruiting).fillna(0.)
    terms=['team_baseline','returning_adjustment','transfer_adjustment','recruit_adjustment','coach_adjustment']
    components['rating_before_centering']=components[terms].sum(axis=1)
    components['centering_adjustment']=-components.rating_before_centering.mean()
    components['power_rating']=components.rating_before_centering+components.centering_adjustment
    names=roster.groupby('team_id').team_display_name.first()
    components['team_display_name']=components.team_id.map(names)
    prior=history[season-1]
    components['tempo']=components.team_id.map(prior.tempo).fillna(prior.tempo.mean())
    components['season']=season
    components['preseason_rating']=components.power_rating
    components['rating_phase']='preseason'
    components['prior_weight']=1.
    components['games_played']=components['wins']=components['losses']=0
    components['record']='0-0'
    components['model_version']=config.model_version
    incumbent=pd.read_csv(ROOT/'historical_ratings'/f'ratings_{season}_preseason.csv')
    components['incumbent_rating']=components.team_id.map(incumbent.set_index('team_id').power_rating)
    components['rating_change']=components.power_rating-components.incumbent_rating
    components=components.sort_values(['power_rating','team_id'],ascending=[False,True]).reset_index(drop=True)
    components['rank']=np.arange(1,len(components)+1)
    if not np.isfinite(components[terms+['power_rating']].to_numpy()).all():raise ValueError('Nonfinite rating component')
    if components.team_id.duplicated().any() or abs(components.power_rating.mean())>1e-10:raise ValueError('Invalid output universe or centering')
    validation['player_coach_recruit_weights']='Provisional capped assumptions; not fitted or claimed validated by the baseline holdout'
    validation['historical_roster_limitation']='No verified preseason-date historical roster snapshots are available. No full-model accuracy claim or coefficient tuning uses retrospectively observed player participation.'
    validation['coaching_limitation']='Historical program-baseline residual proxy; not a causal effect controlling historical roster quality.'
    validation['source_coverage']={'teams':len(ids),'roster_players':len(roster),'prior_production_player_stints':len(values),'coaches':len(coaches),'missing_current_coach_ids':int(coaches.coach_id.isna().sum()),'missing_previous_coach':int(coaching.coach_source_missing.sum()),'matched_recruits':int((recruit_audit.status=='matched').sum()),'unmatched_recruits':int((recruit_audit.status=='not_on_roster').sum()),'excluded_recruits':int((recruit_audit.status=='excluded').sum()),'confirmed_unavailable_players':len(excluded)}
    validation['phaseout']={'games_until_zero':20,'power':1.32,'unchanged':True}
    for name,frame in [('ratings',components),('player_audit',audit),('recruit_audit',recruit_audit),('coach_observations',coach_obs)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    print(components[['rank','team_display_name','power_rating','team_baseline','returning_adjustment','transfer_adjustment','recruit_adjustment','coach_adjustment']].head(25).to_string(index=False))
    print(json.dumps(validation['source_coverage'],indent=2))
    return components,validation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',type=int,default=2027)
    args=parser.parse_args();build(args.season)
