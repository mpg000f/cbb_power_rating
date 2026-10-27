import unittest
import numpy as np
import pandas as pd
from build_preseason_v3 import centered_capped_adjustment,revise
from preseason_model import PreseasonConfig


class ReplacementTests(unittest.TestCase):
    def test_centering_preserves_combined_cap(self):
        result=centered_capped_adjustment([-40,-5,1,2,30],3)
        self.assertLess(abs(result.mean()),1e-10)
        self.assertTrue((np.abs(result)<=3).all())

    def test_identical_roster_has_zero_replacement_charge(self):
        cfg=PreseasonConfig()
        p=pd.DataFrame({'team_id':[1,2],'box_bpm':[2.,2.],'min':[1000.,1000.]})
        value=5*(2*1000/1250+2)
        x=pd.DataFrame({'team_id':[1,2],'retained_value':[value,value],'transfer_value':[0.,0.],'coach_change':[False,False],'coach_adjustment':[2.,1.],'recruit_adjustment':[0.,0.],'team_baseline':[10.,-10.],'power_rating':[15.,-8.],'preseason_rating':[15.,-8.],'returning_adjustment':[3.,2.],'transfer_adjustment':[1.,1.],'rating_before_centering':[15.,-8.],'centering_adjustment':[0.,0.]})
        r=revise(x,p,cfg)
        self.assertTrue(np.allclose(r.departed_value,0))
        self.assertTrue(np.allclose(r.coach_change_adjustment,0))
        self.assertTrue(np.allclose(r.net_roster_adjustment,0))
        self.assertTrue(np.allclose(r.power_rating,r.centered_baseline))

if __name__=='__main__':unittest.main()
