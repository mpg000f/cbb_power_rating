"""Game-calibrated baseline candidate; no discretionary spread multiplier.

Keep the three-year family supported by retrospective comparisons. Fit the
baseline scale using 2022-2026 early games, matching the rolling previous-five-
season protocol used for 2022-2026 evaluation. Roster terms remain provisional.
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from preseason_model import PreseasonConfig, load_history, team_baseline, coaching_component
from build_preseason_v3 import revise

ROOT=Path(__file__).resolve().parent


def fit_game_scale(predictions, target_season):
    p=predictions[(predictions.family=='three_year_70_20_10')&predictions.season.between(target_season-5,target_season-1)].copy()
    if set(p.season)!={target_season-i for i in range(1,6)}:raise ValueError('Five complete prior evaluation seasons required')
    numeric=p[['predicted_home_margin','actual_home_margin','regression']].to_numpy()
    if not np.isfinite(numeric).all() or (p.regression<=0).any() or p.neutral_site.isna().any():raise ValueError('Invalid prediction evidence')
    if p.duplicated(['season','game_id']).any():raise ValueError('Duplicate forecast games')
    if p.training_end.ge(p.season).any():raise ValueError('Forecast contains current/future training data')
    p['home_advantage']=np.where(p.neutral_site.astype(bool),0.,3.5)
    p['unscaled_margin']=(p.predicted_home_margin-p.home_advantage)/p.regression
    scores=[]
    for scale in np.round(np.arange(.50,1.251,.01),2):
        error=p.unscaled_margin*scale+p.home_advantage-p.actual_home_margin
        yearly=pd.DataFrame({'season':p.season,'error':error}).groupby('season').error.apply(lambda x:np.sqrt(np.mean(x*x)))
        scores.append({'scale':float(scale),'mean_season_rmse':float(yearly.mean())})
    best=min(scores,key=lambda r:r['mean_season_rmse'])
    return best,scores


def build():
    season=2027
    inputs=ROOT/'preseason_inputs/2027/v2';review=ROOT/'preseason_candidates/2027/baseline_review'
    predictions=pd.read_csv(review/'early_game_predictions.csv')
    best,grid=fit_game_scale(predictions,season)
    config=replace(PreseasonConfig(),baseline_shrink=best['scale'],model_version='game-calibrated-v4-candidate')
    history=load_history(ROOT/'historical_ratings')
    candidate=pd.read_csv(ROOT/'preseason_candidates/2027/ratings.csv')
    base=team_baseline(history,season,candidate.team_id,config).set_index('team_id')
    for col in base.columns:candidate[col]=candidate.team_id.map(base[col])
    coaches=pd.read_csv(inputs/'coaches.csv');coach_history=pd.read_csv(inputs/'coach_history.csv')
    coaching,observations=coaching_component(history,coach_history,coaches,season,config)
    coaching=coaching.set_index('team_id')
    for col in coaching.columns:candidate[col]=candidate.team_id.map(coaching[col])
    result=revise(candidate,pd.read_csv(inputs/'player_production.csv'),config)
    result['model_version']=config.model_version
    old=pd.read_csv(ROOT/'preseason_candidates/2027/v3/ratings.csv').set_index('team_id')
    result['v3_rating']=result.team_id.map(old.power_rating)
    result['v3_rank']=result.team_id.map(old['rank'])
    result['change_from_v3']=result.power_rating-result.v3_rating
    out=ROOT/'preseason_candidates/2027/v4';out.mkdir(exist_ok=True)
    result.to_csv(out/'ratings.csv',index=False);observations.to_csv(out/'coach_observations.csv',index=False)
    pd.DataFrame(grid).to_csv(out/'game_scale_selection.csv',index=False)
    details={'version':config.model_version,'baseline_weights':[.7,.2,.1],'baseline_scale':best['scale'],'selection_training_years':[2022,2023,2024,2025,2026],'selection_rule':'minimum average season margin RMSE, grid .50-1.25 by .01','training_fit_rmse_not_out_of_sample':best['mean_season_rmse'],'out_of_time_evaluation':'see ../baseline_review/early_game_yearly.csv: each 2022-2026 forecast uses the preceding five seasons only','ratings_sd':float(result.power_rating.std()),'v3_ratings_sd':float(old.power_rating.std()),'net_replacement_scale':.30,'combined_adjustment_cap':3.,'player_development':False,'unchanged_coach_bonus':0.,'phaseout_games':20,'phaseout_power':1.32,'full_model_validated':False,'limitations':['Recent evaluation periods already inspected; these are retrospective comparisons, not a fresh untouched holdout','History-only calibration does not validate provisional player, recruiting or coaching terms','Calibration is specific to the existing tempo and 3.5-point home-advantage rule','No betting/ATS performance claim; no target-season games are used','Coaching-change residuals are recomputed against the recalibrated baseline, while unchanged coaches still get zero bonus']}
    (out/'validation.json').write_text(json.dumps(details,indent=2)+'\n')
    assert result.team_id.nunique()==365 and abs(result.power_rating.mean())<1e-10
    assert (result.combined_adjustment.abs()<=3.00000001).all()
    print(result[['rank','team_display_name','power_rating','combined_adjustment','v3_rank']].head(25).to_string(index=False));print(json.dumps(details,indent=2))
    return result


if __name__=='__main__':build()
