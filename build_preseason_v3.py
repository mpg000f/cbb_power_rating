"""Build a replacement-based preseason candidate without stacked strength bonuses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_preseason_v2 import build as build_v2
from preseason_model import PreseasonConfig

ROOT=Path(__file__).resolve().parent


def centered_capped_adjustment(raw, cap):
    """Center inside the cap so both the population mean and bound hold."""
    raw=np.asarray(raw,dtype=float)
    lo,hi=float(raw.min()-cap),float(raw.max()+cap)
    for _ in range(100):
        midpoint=(lo+hi)/2
        if np.clip(raw-midpoint,-cap,cap).mean()>0:lo=midpoint
        else:hi=midpoint
    return np.clip(raw-(lo+hi)/2,-cap,cap)


def revise(candidate, player_values, config, roster_scale=.30, combined_cap=3.):
    x=candidate.copy()
    p=player_values.copy()
    p['team_id']=p.team_id.astype(int)
    p['value']=(p.box_bpm.clip(-8,12)*p['min']/(p['min']+config.player_sample_minutes)-config.replacement_bpm).clip(0,12)
    p['weighted']=p['min']*p.value
    prior_value=5*p.groupby('team_id').weighted.sum()/p.groupby('team_id')['min'].sum()
    x['prior_roster_value']=x.team_id.map(prior_value)
    # An expansion team has no measured departing D1 roster. Its provisional
    # baseline is retained, with no invented departure charge or roster upgrade.
    x['departed_value']=(x.prior_roster_value-x.retained_value).clip(lower=0)
    x['net_transfer_replacement']=x.transfer_value-x.departed_value
    x['net_roster_adjustment']=(roster_scale*x.net_transfer_replacement).fillna(0)
    x['coach_change_adjustment']=np.where(x.coach_change,x.coach_adjustment,0.)
    x['raw_combined_adjustment']=x.net_roster_adjustment+x.recruit_adjustment+x.coach_change_adjustment
    x['combined_adjustment']=centered_capped_adjustment(x.raw_combined_adjustment,combined_cap)
    x['centered_baseline']=x.team_baseline-x.team_baseline.mean()
    x['v2_rating']=x.power_rating
    x['power_rating']=x.centered_baseline+x.combined_adjustment
    x['preseason_rating']=x.power_rating
    x['model_version']='replacement-v3-candidate'
    # Prior v2 bonuses remain in the audit but are clearly marked as superseded.
    x=x.rename(columns={'returning_adjustment':'superseded_v2_returning_adjustment','transfer_adjustment':'superseded_v2_transfer_adjustment','coach_adjustment':'superseded_v2_coach_adjustment','rating_before_centering':'superseded_v2_rating_before_centering','centering_adjustment':'superseded_v2_centering_adjustment'})
    x=x.sort_values(['power_rating','team_id'],ascending=[False,True]).reset_index(drop=True)
    x['rank']=np.arange(1,len(x)+1)
    return x


def build():
    candidate,validation=build_v2(2027)
    inputs=ROOT/'preseason_inputs/2027/v2'
    config=PreseasonConfig(**{**json.loads((ROOT/'preseason_candidates/2027/config.json').read_text()),'history_weights':(.7,.2,.1)})
    players=pd.read_csv(inputs/'player_production.csv')
    result=revise(candidate,players,config)
    out=ROOT/'preseason_candidates/2027/v3';out.mkdir(exist_ok=True)
    result.to_csv(out/'ratings.csv',index=False)
    sensitivity=[]
    reference=result.set_index('team_id')['rank']
    for scale in [.15,.30,.45]:
        for cap in [2.,3.,4.]:
            scenario=revise(candidate,players,config,scale,cap)
            movement=scenario.set_index('team_id')['rank']-reference
            sensitivity.append({'roster_scale':scale,'combined_cap':cap,'top25_overlap':len(set(scenario.head(25).team_id)&set(result.head(25).team_id)),'median_absolute_rank_change':float(movement.abs().median()),'maximum_absolute_rank_change':int(movement.abs().max()),'teams_at_cap':int(np.isclose(scenario.combined_adjustment.abs(),cap).sum())})
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    validation.update(model_version='replacement-v3-candidate',roster_scale=.30,combined_cap=3.,unchanged_coach_bonus=0.,full_model_validated=False,formula='centered multiyear baseline + centered capped(net replacement production * 0.30 + recruiting + coaching change)',unknown_player_policy='No measured D1 production credit; not evidence of zero ability. Recruiting supplies a conservative bonus where available.')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    assert result.team_id.nunique()==365 and abs(result.power_rating.mean())<1e-10
    assert (result.combined_adjustment.abs()<=3.00000001).all()
    assert (result.loc[~result.coach_change,'coach_change_adjustment']==0).all()
    print('\nREVISED TOP 25\n'+result[['rank','team_display_name','power_rating','combined_adjustment']].head(25).to_string(index=False))
    print('\nSENSITIVITY\n'+pd.DataFrame(sensitivity).to_string(index=False))
    return result


if __name__=='__main__':build()
