"""Auditable preseason components. No player-development / age progression bonus.

Every forecast uses only team seasons strictly before its target season.
Roster terms are additive; preserving minutes never multiplies a negative rating.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import unicodedata
import re
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreseasonConfig:
    history_weights: tuple = (.70, .20, .10)
    baseline_shrink: float = .85
    player_sample_minutes: float = 250.
    replacement_bpm: float = -2.
    returning_scale: float = .35
    returning_cap: float = 3.
    transfer_scale: float = .30
    transfer_cap: float = 3.
    recruit_max_player: float = 1.25
    recruit_cap: float = 3.
    coach_scale: float = .35
    coach_cap: float = 2.
    coach_prior_seasons: float = 4.
    transition_penalty: float = .50
    model_version: str = 'production-v2-candidate'


def normalized_name(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    words = re.sub(r'[^a-z0-9 ]', '', value).split()
    return ' '.join(w for w in words if w not in {'jr', 'sr', 'ii', 'iii', 'iv'})


def load_history(directory):
    result = {}
    for path in sorted(Path(directory).glob('ratings_*.csv')):
        if re.fullmatch(r'ratings_\d{4}', path.stem):
            year = int(path.stem.split('_')[1])
            df = pd.read_csv(path)
            if df.games_played.max() > 0:
                if df.team_id.duplicated().any():
                    raise ValueError(f'Duplicate teams in {path}')
                result[year] = df.set_index('team_id')
    return result


def team_baseline(history, season, team_ids, config):
    result = pd.DataFrame(index=pd.Index(team_ids, name='team_id'))
    values, availability = [], []
    for lag, weight in enumerate(config.history_weights, 1):
        if season-lag not in history:
            raise ValueError(f'Missing historical season {season-lag}')
        series = history[season-lag].power_rating
        series = series - series.mean()
        result[f'prior_{lag}'] = series.reindex(result.index)
        values.append(result[f'prior_{lag}'].fillna(0) * weight)
        availability.append(result[f'prior_{lag}'].notna() * weight)
    denominator = sum(availability)
    result['history_coverage'] = denominator
    result['team_baseline'] = sum(values).div(denominator.replace(0, np.nan)) * config.baseline_shrink
    result['prior_rating'] = result.prior_1
    result['program_rating'] = (result.prior_2.fillna(result.prior_1) + result.prior_3.fillna(result.prior_1)) / 2
    # Explicit lower-end prior for a new D1 member; never silently average strength.
    expansion = history[season-1].power_rating
    expansion = expansion[expansion <= expansion.quantile(.25)].median()
    result['team_baseline'] = result.team_baseline.fillna(expansion)
    result['is_expansion'] = denominator == 0
    return result.reset_index()


def calibrate_baseline(history, training_end=2021, holdout_start=2022):
    """Select on 2012-2021 only; 2022+ is untouched temporal holdout.

    Later final seasons cannot affect earlier predictors. New D1 teams are
    excluded from validation because their provisional priors are a separate policy.
    """
    weights = [(1.,0.,0.),(.8,.15,.05),(.7,.2,.1),(.6,.25,.15),(.55,.3,.15)]
    shrinks = [.65,.70,.75,.80,.85,.90,.95,1.]
    seasons = [s for s in sorted(history) if s >= 2012 and all(s-l in history for l in [1,2,3])]
    rows = []
    for w in weights:
        for shrink in shrinks:
            cfg = PreseasonConfig(history_weights=w, baseline_shrink=shrink)
            yearly=[]
            for season in seasons:
                actual=history[season].power_rating
                predicted=team_baseline(history,season,actual.index,cfg).set_index('team_id')
                valid=predicted.prior_1.notna()
                error=predicted.loc[valid,'team_baseline']-actual.loc[valid]
                yearly.append({'season':season,'rmse':float(np.sqrt(np.mean(error**2))), 'n':int(valid.sum())})
            # Equal weight for each season, so membership size does not tune the model.
            loss=float(np.mean([r['rmse'] for r in yearly if r['season']<=training_end]))
            rows.append({'weights':w,'shrink':shrink,'training_rmse':loss,'years':yearly})
    best=min(rows,key=lambda r:r['training_rmse'])
    cfg=PreseasonConfig(history_weights=best['weights'],baseline_shrink=best['shrink'])
    last_year=next(r for r in rows if r['weights']==(1.,0.,0.) and r['shrink']==1.)
    report={'training_end':training_end,'holdout_start':holdout_start,'selection':'minimum mean season RMSE on training seasons only','selected':{'weights':best['weights'],'shrink':best['shrink']},'training_rmse':best['training_rmse'], 'holdout':[]}
    for a,b in zip(best['years'],last_year['years']):
        if a['season']>=holdout_start:
            report['holdout'].append({**a,'last_year_only_rmse':b['rmse']})
    report['holdout_mean_rmse']=float(np.mean([x['rmse'] for x in report['holdout']]))
    report['last_year_only_holdout_mean_rmse']=float(np.mean([x['last_year_only_rmse'] for x in report['holdout']]))
    return cfg,report


def production_components(roster, player_values, history, season, config, excluded_ids=()):
    """Opponent-adjusted box BPM * evidence-weighted playing-time share.

    BPM already adjusts for team/opponent strength. No second conference-tier
    discount is applied. Incoming minutes are capped to the destination's vacant
    rotation capacity. Missing production is unknown, not zero-quality evidence.
    """
    if roster.duplicated('athlete_id').any():
        raise ValueError('Player assigned to multiple current rosters')
    p=player_values.copy()
    if not (p.season == season-1).all():
        raise ValueError('Player-production season must be target minus one')
    for col in ['player_id','team_id']:
        p[col]=pd.to_numeric(p[col],errors='raise').astype(int)
    if p.duplicated(['player_id','team_id']).any():
        raise ValueError('Duplicate player/team production')
    if not np.isfinite(p[['min','box_bpm']].to_numpy()).all() or (p['min']<0).any():
        raise ValueError('Invalid player production')
    p=p[p.team_id.isin(history[season-1].index)].copy()
    team_minutes=p.groupby('team_id')['min'].sum()
    # Multiple prior teams: retain the minutes-weighted production of every stint.
    p['evidence_bpm']=p.box_bpm.clip(-8,12) * p['min']/(p['min']+config.player_sample_minutes)
    p['value']= (p.evidence_bpm-config.replacement_bpm).clip(0,12)
    p['weighted_value']=p['min']*p.value
    # Index a player's stints once rather than repeated whole-table scans.
    groups={int(k):v for k,v in p.groupby('player_id')}
    audit=[]; summary=[]
    for tid,team in roster.groupby('team_id'):
        denom=team_minutes.get(tid, float(team_minutes.median()))
        if denom<=0:raise ValueError(f'{tid}: nonpositive team minute denominator')
        retained_min=incoming_min=retained_value=incoming_value=0.
        unknown=excluded=0
        for row in team.itertuples():
            aid=int(row.athlete_id)
            if aid in excluded_ids:
                excluded+=1
                audit.append({'team_id':tid,'athlete_id':aid,'player':row.player,'role':'excluded','source_team_id':None,'prior_minutes':None,'prior_bpm':None,'evidence_bpm':None,'production_value':None})
                continue
            previous=groups.get(aid)
            if previous is None:
                unknown+=1
                audit.append({'team_id':tid,'athlete_id':aid,'player':row.player,'role':'no_prior_d1_production','source_team_id':None,'prior_minutes':None,'prior_bpm':None,'evidence_bpm':None,'production_value':None})
                continue
            for old in previous.itertuples():
                role='returner' if old.team_id==tid else 'transfer'
                if role=='returner':retained_min+=old.min;retained_value+=old.weighted_value
                else:incoming_min+=old.min;incoming_value+=old.weighted_value
                audit.append({'team_id':tid,'athlete_id':aid,'player':row.player,'role':role,'source_team_id':old.team_id,'prior_minutes':old.min,'prior_bpm':old.box_bpm,'evidence_bpm':old.evidence_bpm,'production_value':old.value})
        retained_share=min(retained_min/denom,1.)
        incoming_share=incoming_min/denom
        capacity_factor=min(1., max(0.,1.-retained_share)/incoming_share) if incoming_share>0 else 1.
        summary.append({'team_id':tid,'returning_pct':100*retained_share,'transfer_in_pct':100*incoming_share*capacity_factor,'transfer_capacity_factor':capacity_factor,'retained_value':5*retained_value/denom,'transfer_value':5*incoming_value/denom*capacity_factor,'unknown_production_players':unknown,'excluded_players':excluded})
    result=pd.DataFrame(summary)
    result['effective_returning']=(result.returning_pct+result.transfer_in_pct).clip(0,100)
    for key,scale,cap,name in [('retained_value',config.returning_scale,config.returning_cap,'returning_adjustment'),('transfer_value',config.transfer_scale,config.transfer_cap,'transfer_adjustment')]:
        result[name]=((result[key]-result[key].mean())*scale).clip(-cap,cap)
    return result,pd.DataFrame(audit)


def recruiting_component(roster, recruiting, config, excluded_ids=()):
    """Conservative top-100 RSCI bonus, matched to current roster, not commitment.

    No rank extrapolation for an unranked freshman. No age/class development effect.
    """
    r=roster[~roster.athlete_id.isin(excluded_ids)].copy()
    r['name_key']=r.player.map(normalized_name)
    recruits=recruiting[['RSCI','Player','College']].copy()
    recruits['rank']=pd.to_numeric(recruits.RSCI,errors='raise')
    recruits['name_key']=recruits.Player.map(normalized_name)
    if recruits.name_key.duplicated().any():raise ValueError('Ambiguous recruiting names')
    # Explicit reviewed spelling/identity aliases live with the source snapshot.
    aliases = recruiting.attrs.get('athlete_aliases', {})
    matches=[]
    for recruit in recruits.itertuples():
        key=normalized_name(recruit.Player)
        candidates=r[r.name_key==key]
        if key in aliases:
            candidates=r[r.athlete_id==int(aliases[key])]
        excluded = bool(roster.loc[roster.athlete_id.isin(excluded_ids), 'player'].map(normalized_name).eq(key).any())
        if len(candidates)>1:
            raise ValueError(f'Ambiguous recruit {recruit.Player}: provide a reviewed athlete-ID alias')
        row={'rank':recruit.rank,'recruit':recruit.Player,'committed_college':recruit.College,'team_id':None,'athlete_id':None,'matched_player':None,'status':'excluded' if excluded else 'not_on_roster','bonus':0.}
        if len(candidates)==1:
            player=candidates.iloc[0]
            row.update(team_id=int(player.team_id),athlete_id=int(player.athlete_id),matched_player=player.player,status='matched',bonus=config.recruit_max_player*max(0,(101-recruit.rank)/100)**2)
        matches.append(row)
    matches=pd.DataFrame(matches)
    result=matches.dropna(subset='team_id').groupby('team_id').bonus.sum().clip(upper=config.recruit_cap)
    return result,matches


def coaching_component(history, coach_history, current_coaches, season, config):
    """Shrunken persistent residual above the regressed team baseline.

    This is a program-residual proxy, not an isolated causal coaching effect:
    historical roster talent is not controlled. Changes use new-minus-old coach
    residuals so the baseline's previous coach is not credited twice.
    """
    h=coach_history.copy()
    h=h[(h.season<season)&h.coach_id.notna()&(h.coach_count==1)].copy()
    if h.duplicated(['season','team_id']).any():raise ValueError('Duplicate coach/team/season')
    observations=[]
    for year in sorted(h.season.unique()):
        if not all(year-l in history for l in [0,1,2,3]):continue
        coaches=h[h.season==year]
        base=team_baseline(history,int(year),history[year].index,config).set_index('team_id')
        actual=history[year].power_rating
        for row in coaches.itertuples():
            if row.team_id not in base.index or pd.isna(base.loc[row.team_id,'prior_1']):continue
            observations.append({'coach_id':str(int(row.coach_id)),'season':year,'residual':actual.loc[row.team_id]-base.loc[row.team_id,'team_baseline']})
    obs=pd.DataFrame(observations,columns=['coach_id','season','residual'])
    estimates={}
    for cid,g in obs.groupby('coach_id'):
        w=np.power(.8,season-1-g.season)
        # Confidence depends on observed seasons; wins don't create confidence.
        estimates[cid]=(float(np.average(g.residual.clip(-15,15),weights=w))*len(g)/(len(g)+config.coach_prior_seasons),len(g))
    previous=h[h.season==season-1].set_index('team_id').coach_id.to_dict()
    rows=[]
    for c in current_coaches.itertuples():
        cid=str(int(c.coach_id)) if pd.notna(c.coach_id) else None
        old=str(int(previous[c.team_id])) if c.team_id in previous else None
        value,n=estimates.get(cid,(0.,0));previous_value,_=estimates.get(old,(0.,0))
        changed=cid is not None and old is not None and cid!=old
        raw=config.coach_scale*(value-previous_value if changed else value)
        if changed:raw-=config.transition_penalty
        rows.append({'team_id':c.team_id,'coach_id':cid,'coach':c.coach,'previous_coach_id':old,'coach_change':changed,'coach_history_seasons':n,'coach_residual':value,'previous_coach_residual':previous_value,'coach_adjustment':float(np.clip(raw,-config.coach_cap,config.coach_cap)),'coach_source_missing':cid is None or old is None})
    return pd.DataFrame(rows),obs
