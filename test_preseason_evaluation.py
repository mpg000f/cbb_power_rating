import unittest
import numpy as np
import pandas as pd
from build_preseason_v4 import fit_game_scale
from evaluate_preseason_games import previous_baseline,early_games


class EvaluationTests(unittest.TestCase):
    def test_target_ratings_cannot_change_forecast(self):
        history={y:pd.DataFrame({'team_id':[1,2],'power_rating':[-5.,5.],'tempo':[70.,70.]}).set_index('team_id') for y in [2024,2025,2026]}
        a=previous_baseline(history,2027,(.7,.2,.1))
        history[2027]=history[2026].assign(power_rating=[-999.,999.])
        b=previous_baseline(history,2027,(.7,.2,.1))
        pd.testing.assert_series_equal(a,b)

    def test_only_earlier_seasons_select_scale(self):
        p=pd.DataFrame([{'family':'three_year_70_20_10','season':y,'game_id':y,'neutral_site':True,'predicted_home_margin':10.,'actual_home_margin':11.,'regression':1.,'training_end':y-1} for y in range(2022,2027)])
        a,_=fit_game_scale(p,2027)
        # Even spectacular target-season outcomes cannot alter this forecast.
        p=pd.concat([p,p.iloc[[0]].assign(season=2027,game_id=999,actual_home_margin=-999.,training_end=2026)],ignore_index=True)
        b,_=fit_game_scale(p,2027)
        self.assertEqual(a,b)
        self.assertEqual(a['scale'],1.1)

    def test_evidence_training_leak_is_rejected(self):
        p=pd.DataFrame([{'family':'three_year_70_20_10','season':y,'game_id':y,'neutral_site':True,'predicted_home_margin':10.,'actual_home_margin':11.,'regression':1.,'training_end':y} for y in range(2022,2027)])
        with self.assertRaises(ValueError):fit_game_scale(p,2027)

    def test_first_five_game_scope_and_same_eligible_teams(self):
        sched=pd.DataFrame([{'game_id':i,'date':f'2025-11-{i:02d}','season_type':2,'status_type_completed':True,'home_score':80,'away_score':60,'home_id':1,'away_id':2} for i in range(1,8)])
        sched=pd.concat([sched,sched.iloc[[0]].assign(game_id=20,away_id=999)],ignore_index=True)
        result=early_games(sched,{1,2})
        self.assertEqual(result.game_id.tolist(),[1,2,3,4,5])

if __name__=='__main__':unittest.main()
