# Replacement-based candidate, revision 3

Run `python build_preseason_v3.py`. This builds on the same frozen inputs and validated 70/20/10 history baseline with 0.85 shrinkage. No live site output changes.

Changes from v2:
- Retained production is no longer a separate positive bonus. Departed production equals prior roster value less retained value, using the same minutes-weighted, sample-shrunken box BPM calculation on both sides.
- Transfer production first offsets departed production. Net replacement adjustment is 0.30 times (capacity-constrained incoming value minus departed value). An unchanged full roster has zero replacement adjustment.
- The conservative RSCI freshman contribution enters the same combined adjustment, helping offset lost production rather than stacking outside its bound.
- An unchanged coach receives no extra bonus. Only the previously computed new-minus-old coaching-change adjustment remains.
- Center and clip the entire adjustment together within +/-3 points per 100 possessions. Solve centering inside the clip so the cap holds after centering as well. Center the team baseline independently, then add this bounded adjustment.

The 0.30 scale and +/-3 combined limit are provisional guardrails, not fitted coefficients. Unknown newcomers have no measured D1 production credit; this is not evidence of zero ability. Freshmen receive a modest recruiting contribution where matched. Expansion teams have no invented departing-D1 charge. No player-development effect is added. The existing 20-game phaseout is unchanged.

Sensitivity checks vary net replacement scale across 0.15, 0.30, 0.45 and the combined cap across 2, 3, 4. Top-25 membership overlap is 23-25 teams. At the default, 13 of 365 teams reach the cap. Rankings can still move materially outside the top 25; see sensitivity.csv. This does not establish predictive accuracy.

ratings.csv retains the old v2 bonus fields prefixed superseded_v2_* for audit. The active formula is centered_baseline + combined_adjustment. No standalone returning, transfer or unchanged-coach bonus is added.
