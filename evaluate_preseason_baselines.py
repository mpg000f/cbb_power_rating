"""Compare preseason team-history models using only earlier seasons for tuning.

Previously inspected 2022-2026 results are evaluation data, not a new untouched
holdout. No roster coefficients are fitted from retrospective participation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from preseason_model import load_history

ROOT=Path(__file__).resolve().parent
FAMILIES={'one_year':(1.,0.,0.),'two_year_85_15':(.85,.15,0.),'three_year_70_20_10':(.7,.2,.1)}
SHRINKS=np.round(np.arange(.60,1.101,.01),2)


def raw_baseline(history,season,weights):
    ids=history[season].index if season in history else history[season-1].index
    num=pd.Series(0.,index=ids);den=num.copy()
    for lag,w in enumerate(weights,1):
        s=history[season-lag].power_rating
        s=(s-s.mean()).reindex(ids)
        num+=s.fillna(0)*w;den+=s.notna()*w
    p=num/den.replace(0,np.nan)
    valid=history[season-1].power_rating.reindex(ids).notna()
    return p[valid]


def evaluate():
    history=load_history(ROOT/'historical_ratings')
    years=[y for y in sorted(history) if y>=2012]
    raw={(name,y):raw_baseline(history,y,w) for name,w in FAMILIES.items() for y in years}
    def metrics(name,year,shrink):
        p=raw[name,year]*shrink;a=history[year].power_rating.reindex(p.index);e=p-a
        top=a.rank(ascending=False)<=50
        return {'n':len(e),'rmse':float(np.sqrt((e**2).mean())),'mae':float(e.abs().mean()),'top50_rmse':float(np.sqrt((e[top]**2).mean())),'prediction_sd':float(p.std()),'actual_sd':float(a.std()),'rating_correlation':float(p.corr(a))}
    def tune(name,trainyears):
        scores=[(float(np.mean([metrics(name,y,s)['rmse'] for y in trainyears])),float(s)) for s in SHRINKS]
        return min(scores)[1]
    rows=[]
    for name in FAMILIES:
        fixed=tune(name,[y for y in years if y<=2021])
        for year in [2022,2023,2024,2025,2026]:
            for protocol,train in [('fixed_pre2022',[y for y in years if y<=2021]),('rolling_recent5',list(range(year-5,year))),('expanding',[y for y in years if y<year])]:
                shrink=fixed if protocol=='fixed_pre2022' else tune(name,train)
                rows.append({'family':name,'protocol':protocol,'season':year,'shrink':shrink,'train_start':min(train),'train_end':max(train),**metrics(name,year,shrink)})
    yearly=pd.DataFrame(rows)
    summary=yearly.groupby(['protocol','family'],as_index=False).agg(mean_rmse=('rmse','mean'),mean_mae=('mae','mean'),mean_top50_rmse=('top50_rmse','mean'),mean_sd=('prediction_sd','mean'),mean_actual_sd=('actual_sd','mean'),mean_correlation=('rating_correlation','mean'))
    # Paired season differences are inspectable; five target years are too few
    # for a confident causal claim that one family is universally superior.
    paired=yearly.pivot(index=['protocol','season'],columns='family',values='rmse').reset_index()
    paired['one_minus_three']=paired.one_year-paired.three_year_70_20_10
    paired['two_minus_three']=paired.two_year_85_15-paired.three_year_70_20_10
    current=[]
    for name,w in FAMILIES.items():
        for train_name,train in [('all_prior',[y for y in years if y<=2026]),('recent5',list(range(2022,2027)))]:
            shrink=tune(name,train)
            pred=raw_baseline(history,2027,w)*shrink
            current.append({'family':name,'training':train_name,'shrink':shrink,'sd':float(pred.std()),'highest_baseline':float(pred.max())})
    out=ROOT/'preseason_candidates/2027/baseline_review'
    yearly.to_csv(out/'yearly_errors.csv',index=False);summary.to_csv(out/'summary.csv',index=False);paired.to_csv(out/'paired_errors.csv',index=False)
    pd.DataFrame(current).to_csv(out/'current_baselines.csv',index=False)
    # Audit why v3 is narrower than its own baseline; correlation does not
    # establish that widening or removing the term improves prediction.
    x=pd.read_csv(ROOT/'preseason_candidates/2027/v3/ratings.csv')
    audit={'baseline_sd':float(x.centered_baseline.std()),'v3_sd':float(x.power_rating.std()),'baseline_adjustment_correlation':float(x.centered_baseline.corr(x.combined_adjustment)),'baseline_raw_net_replacement_correlation':float(x.centered_baseline.corr(x.net_transfer_replacement)),'top25_mean_adjustment':float(x.head(25).combined_adjustment.mean()),'teams_at_cap':int(np.isclose(x.combined_adjustment.abs(),3).sum()),'scope':'Team final-rating prediction only; not full-model/game validation. All tuning uses seasons earlier than the forecast year. 2022-2026 evaluation results were previously inspected.'}
    (out/'compression_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(summary.to_string(index=False));print('\nPAIRED\n'+paired.to_string(index=False));print('\n2027\n'+pd.DataFrame(current).to_string(index=False));print('\nCOMPRESSION\n'+json.dumps(audit,indent=2))
    return yearly,summary


if __name__=='__main__':evaluate()
