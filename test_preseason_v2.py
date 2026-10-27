import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from preseason_model import (PreseasonConfig,team_baseline,production_components,
                             recruiting_component,coaching_component)


def history_fixture():
    return {y:pd.DataFrame({'team_id':[1,2,3],'power_rating':[-10.,0.,10.],'tempo':[68.,68.,68.]}).set_index('team_id') for y in range(2016,2027)}


def players_fixture():
    return pd.DataFrame({'player_id':[11,12,21,31],'player':['Good','Bad','Average','Strong'],'season':[2026]*4,'team_id':[1,1,2,3],'min':[500.,500.,1000.,1000.],'box_bpm':[4.,-4.,0.,6.]})


class ProductionModelTests(unittest.TestCase):
    def setUp(self):self.h=history_fixture();self.cfg=PreseasonConfig()

    def test_returning_productive_player_helps_below_average_team(self):
        # Other rosters identical; add a productive returner to a -10 team.
        r=pd.DataFrame({'team_id':[1,2,3],'athlete_id':[99,21,31],'player':['New','Average','Strong']})
        a,_=production_components(r,players_fixture(),self.h,2027,self.cfg)
        r=pd.concat([r,pd.DataFrame([{'team_id':1,'athlete_id':11,'player':'Good'}])],ignore_index=True)
        b,_=production_components(r,players_fixture(),self.h,2027,self.cfg)
        self.assertGreater(b.set_index('team_id').loc[1,'returning_adjustment'],a.set_index('team_id').loc[1,'returning_adjustment'])

    def test_better_transfer_gets_more_credit_at_same_minutes(self):
        r=pd.DataFrame({'team_id':[1,2,3],'athlete_id':[31,11,99],'player':['Strong','Good','New']})
        p=players_fixture();p.loc[p.player_id.isin([11,31]),'min']=500.
        a,_=production_components(r,p,self.h,2027,self.cfg)
        p.loc[p.player_id==31,'box_bpm']=-4.
        b,_=production_components(r,p,self.h,2027,self.cfg)
        self.assertGreater(a.set_index('team_id').loc[1,'transfer_adjustment'],b.set_index('team_id').loc[1,'transfer_adjustment'])

    def test_tiny_samples_do_not_dominate(self):
        p=players_fixture();p.loc[p.player_id==31,['min','box_bpm']]=[1.,100.]
        r=pd.DataFrame({'team_id':[1,2,3],'athlete_id':[31,21,99],'player':['Strong','Average','New']})
        _,audit=production_components(r,p,self.h,2027,self.cfg)
        self.assertLess(audit.loc[audit.athlete_id==31,'evidence_bpm'].iloc[0],.05)

    def test_transfer_minutes_fit_rotation_capacity(self):
        r=pd.DataFrame({'team_id':[1,1,1,2,3],'athlete_id':[11,12,31,21,99],'player':['Good','Bad','Strong','Average','New']})
        a,_=production_components(r,players_fixture(),self.h,2027,self.cfg)
        self.assertTrue((a.effective_returning<=100).all())
        self.assertEqual(a.set_index('team_id').loc[1,'transfer_in_pct'],0.)

    def test_reject_current_season_production(self):
        p=players_fixture();p['season']=2027
        r=pd.DataFrame({'team_id':[1],'athlete_id':[11],'player':['Good']})
        with self.assertRaises(ValueError):production_components(r,p,self.h,2027,self.cfg)

    def test_reject_duplicate_roster_assignment(self):
        r=pd.DataFrame({'team_id':[1,2],'athlete_id':[11,11],'player':['Good','Good']})
        with self.assertRaises(ValueError):production_components(r,players_fixture(),self.h,2027,self.cfg)

    def test_recruit_current_roster_and_availability_override_commitment(self):
        r=pd.DataFrame({'team_id':[1,2],'athlete_id':[11,21],'player':['Top Recruit Jr.','Other']})
        rec=pd.DataFrame({'RSCI':[1,2],'Player':['Top Recruit','Unrostered'],'College':['Old School','Old School']})
        bonus,audit=recruiting_component(r,rec,self.cfg)
        self.assertEqual(bonus.loc[1],self.cfg.recruit_max_player)
        self.assertEqual(audit.iloc[1].status,'not_on_roster')
        bonus,audit=recruiting_component(r,rec,self.cfg,{11})
        self.assertEqual(bonus.sum(),0.)
        self.assertEqual(audit.iloc[0].status,'excluded')

    def test_ambiguous_recruit_requires_explicit_identity(self):
        r=pd.DataFrame({'team_id':[1,2],'athlete_id':[11,21],'player':['Trent Perry','Trent Perry']})
        rec=pd.DataFrame({'RSCI':[1],'Player':['Trent Perry'],'College':['Kansas']})
        with self.assertRaises(ValueError):recruiting_component(r,rec,self.cfg)

    def test_future_coaching_results_do_not_change_forecast(self):
        hist=pd.DataFrame({'season':[2025,2026,2027],'team_id':[1,1,1],'coach_id':[5,5,6],'coach_count':[1,1,1]})
        current=pd.DataFrame({'team_id':[1],'coach_id':[5],'coach':['Coach']})
        a,_=coaching_component(self.h,hist,current,2027,self.cfg)
        b,_=coaching_component(self.h,hist[hist.season<2027],current,2027,self.cfg)
        pd.testing.assert_frame_equal(a,b)

    def test_departed_coach_is_not_assigned_to_replacement(self):
        hist=pd.DataFrame({'season':[2024,2025,2026],'team_id':[3,3,3],'coach_id':[5,5,5],'coach_count':[1,1,1]})
        keep=pd.DataFrame({'team_id':[3],'coach_id':[5],'coach':['Established']})
        change=pd.DataFrame({'team_id':[3],'coach_id':[6],'coach':['New']})
        a,_=coaching_component(self.h,hist,keep,2027,self.cfg)
        b,_=coaching_component(self.h,hist,change,2027,self.cfg)
        self.assertLess(b.coach_adjustment.iloc[0],a.coach_adjustment.iloc[0])
        self.assertTrue(b.coach_change.iloc[0])
        self.assertEqual(b.coach_history_seasons.iloc[0],0)

    def test_baseline_never_reads_target_season(self):
        a=team_baseline(self.h,2027,[1,2,3],self.cfg)
        h=dict(self.h);h[2027]=h[2026].assign(power_rating=[100.,200.,300.])
        b=team_baseline(h,2027,[1,2,3],self.cfg)
        pd.testing.assert_frame_equal(a,b)

    def test_development_is_not_a_model_parameter(self):
        self.assertFalse(any('develop' in k or 'age' in k for k in self.cfg.__dict__))

if __name__=='__main__':unittest.main()
