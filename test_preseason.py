import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import power_rating as model
from build_static import csv_to_json


class PreseasonTests(unittest.TestCase):
    def test_decay_and_blend(self):
        cfg = model.RatingConfig()
        self.assertEqual(model.calculate_prior_weight_preseason(0, cfg), 1)
        self.assertAlmostEqual(model.calculate_prior_weight_preseason(10, cfg), .5**1.32)
        self.assertEqual(model.calculate_prior_weight_preseason(20, cfg), 0)
        stats = pd.DataFrame({'team_id': [1, 2], 'games_played': [0, 20], 'off_efficiency': [110., 90.], 'def_efficiency': [100., 100.]})
        prior = pd.DataFrame({'team_id': [1, 2], 'preseason_rating': [5., -5.], 'effective_returning': [60., 60.]})
        output = model.calculate_power_rating(stats, cfg, preseason_data=prior)
        self.assertEqual(list(output.power_rating), [5., -10.])

    def test_zero_games_keep_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = pd.DataFrame({'team_id': [1], 'power_rating': [9.], 'games_played': [0]})
            baseline.to_csv(Path(directory) / 'ratings_2027_preseason.csv', index=False)
            with patch.object(model, 'load_team_boxscores', return_value=pd.DataFrame()), patch.object(model, 'ESPN_AVAILABLE', False):
                result = model.run_power_ratings(2027, config=model.RatingConfig(prior_ratings_path=directory))
            pd.testing.assert_frame_equal(result, baseline)

    def test_export_has_no_fabricated_efficiency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ratings.csv'
            pd.DataFrame({'team_id': [1], 'power_rating': [9.], 'games_played': [0], 'rating_phase': ['preseason']}).to_csv(path, index=False)
            row = csv_to_json(path)[0]
            self.assertEqual(row['ratingPhase'], 'preseason')
            self.assertNotIn('adjO', row)

if __name__ == '__main__':
    unittest.main()
