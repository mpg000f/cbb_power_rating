# Preliminary historical coaching test

Run `python evaluate_preseason_coaches.py` to reproduce from committed inputs.

The test holds the existing walk-forward, game-calibrated three-year baseline fixed and adds coaching components using only prior-season ratings and outcomes. It covers 4,058 early games in seasons ending 2022–2026. Each team is within its first five completed games against prior-rated D1 opponents. Forecasts use prior-year tempo and the existing 3.5-point home advantage.

| Model | Average yearly margin RMSE |
| --- | ---: |
| Baseline alone | 12.869445 |
| Current change-only coaching term | 12.855188 |
| Half-size change-only term | 12.861297 |
| Previous bonus for unchanged coaches as well | 12.881167 |

Among 1,388 games with known coach metadata and at least one coaching change, the current term reduces average yearly RMSE from 12.934542 to 12.895696. It improves RMSE in all five years in that subgroup, but the magnitude is small. No significance claim is made. This does not support increasing coaching weights, restoring unchanged-coach bonuses, or declaring the whole model validated.

## Limits

ESPN season coach associations were collected retrospectively and have not been audited against opening-day head coaches. Midseason changes could introduce membership leakage even though no target-year outcome enters the adjustment. Records begin in 2020, which truncates coach histories, particularly for earlier forecasts. Unknown metadata receive zero adjustment and are excluded from the known-change subgroup. The all-games result includes missing metadata under that policy.

Historical preseason roster snapshots, recruiting and availability were not reconstructed. The current roster and combined adjustment cap are not part of this test. Archived end-of-season rosters or actual target-season minutes must not be presented as opening-day information. The full preseason model remains unvalidated. The target years were previously inspected during development, so this is a retrospective diagnostic rather than an untouched holdout.

`yearly.csv`, `summary.csv`, `adjustments.csv`, `predictions.csv`, and `method.json` preserve the calculations and limitations. No live ratings changed.
