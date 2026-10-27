# Predictive baseline review and v4 candidate

## Decision

Keep 70/20/10 team-history weights. Regressed one-year and 85/15 two-year alternatives did not beat the three-year baseline in these retrospective tests. Use early-game margin prediction to calibrate the scale rather than matching KenPom/Torvik's rating distribution. Do not increase provisional roster weights without verified historical preseason roster evidence.

## Fair comparison against shorter history

`python evaluate_preseason_baselines.py` compares the three families, each with its own fitted regression strength. It includes parameters selected on 2012-2021, expanding historical training, and rolling previous-five-season training. All target-year predictors use earlier seasons only. The 2022-2026 periods have already been inspected in this project, so this is not a new pristine holdout.

Average yearly final-rating RMSE with expanding training:

| Family | RMSE, points per 100 possessions |
|---|---:|
| Regressed one-year | 6.9595 |
| 85/15 two-year | 6.7723 |
| 70/20/10 three-year | 6.6290 |

The three-year family wins in each of the five evaluation years. Its advantage is also present with recent-five-season fitting; these results do not justify removing older seasons simply because of the transfer portal. They do not establish that 70/20/10 is the optimal history specification conditional on a complete roster model.

## Test the actual prediction target: early games

`python evaluate_preseason_games.py` loads the public SportsDataverse season schedules for 2017-2026. For each 2022-2026 target, it selects a rating scale using only the prior five seasons' early-game margins. The common grid is 0.50-1.25 in increments of 0.01; each training season receives equal weight in average yearly RMSE. Home advantage is fixed at the site's 3.5 game points, and pace uses only prior-season team tempos under the site's multiplicative rule.

The sample contains 4,058 completed regular-season games where both teams are within their first five games against prior-year-rated D1 opponents. New/unrated opponents are omitted consistently across models. Actual target-year tempo, final ratings and roster participation never enter a prediction.

| Family / protocol | Average yearly game-margin RMSE | Average yearly MAE |
|---|---:|---:|
| One-year, game-fitted scale | 13.1129 | 10.3204 |
| 85/15 two-year, game-fitted scale | 12.9752 | 10.2089 |
| 70/20/10 three-year, game-fitted scale | 12.8694 | 10.1253 |
| Three-year, old fixed 0.85 scale | 13.1318 | 10.2797 |
| Three-year, no scale adjustment (1.00) | 12.9036 | 10.1345 |
| Three-year, stretch fitted scale another 20% | 12.9941 | 10.2627 |

The fitted three-year model beats both shorter-history models in every target season. Its error improvement over the one-year model and old fixed-scale baseline is modest (roughly 2%), not a dramatic predictive breakthrough. An extra 20% stretch worsens average RMSE: a wider distribution is not inherently better.

This is margin error in actual game points. It is not ATS, betting profitability, full-model evaluation, or a comparison of historical KenPom/Torvik accuracy. Schedule scores are outcomes only. Overlapping teams/games and only five target seasons limit confidence; no statistical-significance claim is made.

## What caused the narrow v3 distribution?

V3 baseline SD is 9.395; after its net roster adjustment the SD is 8.713. That adjustment correlates -0.562 with the baseline; top-25 teams receive -0.931 points on average. Only 13/365 teams hit the cap. The negative association may reflect real losses or additional mean reversion and is not proof that every departure adjustment is wrong. It does not justify lifting the cap on its own.

The old 0.85 scale was selected to predict final-season team ratings, not early-game score margins. Those targets favor different calibration. For 2027, the same previous-five-season game-fitting protocol (2022-2026) chooses **1.10**. This is a fitted calibration of the raw 70/20/10 baseline, not an added 10% multiplier on v3's finished ratings.

## Revised candidate

`python build_preseason_v4.py` produces `../v4/ratings.csv` from the frozen current roster inputs and that game-calibrated baseline. The combined roster/recruiting/coaching-change cap stays +/-3; there is no unchanged-coach bonus or player development. Coaching-change residuals are recomputed against the calibrated baseline. The 20-game / 1.32-power preseason phaseout remains unchanged.

V4 SD is **11.476**, versus v3's **8.713**. The current calibration's fit error is training fit, not a new out-of-sample result; the forward-season checks above support the protocol. The full roster/coaching model is still not historically validated. No live ratings change.

## Evidence

- `summary.csv`, `yearly_errors.csv`, `paired_errors.csv`: final-rating tests.
- `early_game_summary.csv`, `early_game_yearly.csv`: actual game-error tests.
- `early_game_predictions.csv`: all evaluated forecasts, scores, team IDs and training cutoffs.
- `early_game_method.json`: exact eligibility and prediction rules.
- `compression_audit.json`: descriptive v3 spread audit.
- `../v4/game_scale_selection.csv`: complete 2027 scale grid, including alternatives.

Checks: 21 unit tests, independent recomputation of a deterministic cross-season sample of game predictions, and reconciliation of all 365 final component sums. Source schedules: `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet`.
