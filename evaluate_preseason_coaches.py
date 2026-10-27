"""Retrospective coaching ablation; season coach metadata is NOT as-of audited.

Scores and ratings used to construct adjustments always precede the forecast.
This is a diagnostic, not validation of preseason roster/coaching availability.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from preseason_model import load_history, coaching_component, PreseasonConfig

ROOT = Path(__file__).resolve().parent


def evaluate():
    history = load_history(ROOT / 'historical_ratings')
    coaches = pd.read_csv(ROOT / 'preseason_inputs/2027/v2/coach_history.csv')
    games = pd.read_csv(ROOT / 'preseason_candidates/2027/baseline_review/early_game_predictions.csv')
    games = games[games.family == 'three_year_70_20_10'].copy()
    if games.duplicated(['season', 'game_id']).any():
        raise ValueError('Duplicate predictions')
    yearly, predictions, adjustments = [], [], []
    for year, g in games.groupby('season'):
        if (g.training_end >= year).any() or g.regression.nunique() != 1:
            raise ValueError('Invalid temporal calibration')
        current = coaches[(coaches.season == year) & (coaches.coach_count == 1)].copy()
        current['coach'] = current.coach_id.astype(str)
        config = PreseasonConfig(baseline_shrink=float(g.regression.iloc[0]))
        component, _ = coaching_component(history, coaches, current, int(year), config)
        component['season'] = year
        adjustments.append(component)
        component = component.set_index('team_id')
        prior = history[year - 1]
        poss = g.home_id.map(prior.tempo) * g.away_id.map(prior.tempo) / prior.tempo.mean()
        known = (~g.home_id.map(component.coach_source_missing).astype("boolean").fillna(True).astype(bool)
                 & ~g.away_id.map(component.coach_source_missing).astype("boolean").fillna(True).astype(bool))
        changed = (g.home_id.map(component.coach_change).astype("boolean").fillna(False).astype(bool)
                   | g.away_id.map(component.coach_change).astype("boolean").fillna(False).astype(bool))
        schemes = {
            'baseline_only': pd.Series(0., index=component.index),
            'change_only_current': component.coach_adjustment.where(component.coach_change, 0.),
            'change_only_half': .5 * component.coach_adjustment.where(component.coach_change, 0.),
            'all_coaches_old_bonus': component.coach_adjustment,
        }
        for name, adj in schemes.items():
            pred = g.predicted_home_margin + (g.home_id.map(adj).fillna(0) - g.away_id.map(adj).fillna(0)) * poss / 100
            for scope, mask in [('all_games', pd.Series(True, index=g.index)),
                                ('known_coach_change_games', known & changed)]:
                err = pred[mask] - g.actual_home_margin[mask]
                yearly.append(dict(season=year, model=name, scope=scope, games=int(mask.sum()),
                                   rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(abs(err)))))
            for idx in g.index:
                predictions.append(dict(season=year, game_id=g.loc[idx, 'game_id'], model=name,
                                        predicted_home_margin=float(pred.loc[idx]),
                                        actual_home_margin=float(g.loc[idx, 'actual_home_margin']),
                                        known_coach_change_game=bool((known & changed).loc[idx])))
    out = ROOT / 'preseason_candidates/2027/coaching_review'
    out.mkdir(exist_ok=True)
    yearly = pd.DataFrame(yearly)
    summary = yearly.groupby(['model', 'scope'], as_index=False).agg(
        games=('games', 'sum'), mean_yearly_rmse=('rmse', 'mean'), mean_yearly_mae=('mae', 'mean'))
    yearly.to_csv(out / 'yearly.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    pd.concat(adjustments).to_csv(out / 'adjustments.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    (out / 'method.json').write_text(json.dumps({
        'status': 'preliminary diagnostic; not as-of validated',
        'seasons': sorted(map(int, games.season.unique())),
        'baseline': 'previous-five-year game-calibrated 70/20/10; same frozen forecasts for all ablations',
        'coaching': 'existing .35 residual scale, four-season confidence prior, .5 transition penalty, +/-2 cap; no fitting to evaluated outcomes',
        'limitations': [
            'ESPN season coach associations were collected retrospectively and not audited against opening-day coaches; midseason changes can leak membership.',
            'Coach history begins in 2020, truncating experience especially in earlier test years.',
            'Recruiting and roster production adjustments are omitted; their combined +/-3 cap is not applied.',
            'Recent seasons were already inspected; scenarios are diagnostic, not an untouched model-selection holdout.',
            'Season-average RMSE differences are descriptive, not significance estimates.'
        ]}, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(yearly[yearly.scope == 'known_coach_change_games'].to_string(index=False))
    return yearly, summary


if __name__ == '__main__':
    evaluate()
