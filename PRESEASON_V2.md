# 2026–27 preseason model: production candidate

This builds a complete 365-team candidate with traceable components. It does **not** replace the live ratings. The multiyear baseline has a chronological holdout check; the player, recruiting and coaching weights remain conservative, provisional assumptions. No full-model accuracy advantage is claimed.

Run from the repository root:

```sh
# Reproduce the candidate from the committed compact source snapshots:
python build_preseason_v2.py --season 2027
python -m unittest test_preseason test_preseason_v2 -v

# Recollect public inputs if a deliberate refresh is needed:
python collect_preseason_sources.py
python prepare_preseason_v2.py
python build_preseason_v2.py --season 2027
```

The collector caches each request, retries failures, and resumes interrupted downloads. Use a new cache or explicitly remove a source cache file to refresh it. The preparation step contains reviewed recruiting name aliases and availability exclusions; review those policies before reusing it for another season. The current collector and preparation policy intentionally support only 2027.

## Formula and units

All components are in rating points per 100 possessions:

`rating = team baseline + returning production + transfer production + recruiting + coaching + common centering offset`

The final population is centered at zero. These are not literal game spreads; spread estimates still need possession scaling. Tempo is inherited from the previous season (league mean for expansion teams). No unsupported offensive/defensive split is fabricated.

### Team baseline

Use the last three final ratings, centered within each year, with weights **70% / 20% / 10%**, then multiply by **0.85**. Renormalize history weights for a team missing one of those seasons. New D1 members use the median rating of the prior year's bottom quartile.

Selection used the average season-level RMSE across 2012–2021 only, from five history-weight choices and eight shrinkage factors. The untouched 2022–2026 holdout had average season RMSE **6.629**, compared with **7.336** for unregressed last-year-only ratings, a **9.64% reduction**. This is forecast error against our final team ratings, not game spread error, betting performance, or a comparison with KenPom/Torvik/BPI. It does not validate the other new components.

The exact year-by-year results and selection rule are in `preseason_candidates/2027/validation.json`.

### Returning and transfer production

Use SportsDataverse's season-level, opponent/team-adjusted **box BPM**. This is that provider's metric, not Sports-Reference BPM, and defensive box estimates remain less informative than full lineup data.

For each prior player-team stint:

1. Clip BPM to [-8, 12] to contain extreme outliers.
2. Multiply by `minutes / (minutes + 250)` to reduce reliance on small samples. This is evidence shrinkage, **not player development**.
3. Compute nonnegative value above a provisional replacement level of -2: `clip(shrunken BPM + 2, 0, 12)`.
4. Multiply by the stint's minutes divided by the destination team's prior total minutes, then by 5 to put the five-player rotation contribution on a team scale.

Returning and incoming production are summed separately. Incoming experience is proportionally reduced if retained plus incoming minutes would exceed a full rotation. This is a capacity constraint, not a forecast of individual minutes. A non-D1 arrival or player with no prior D1 season remains explicitly unknown; it is not a measured zero-quality player.

Subtract each component's D1 mean. Scale returning value by **0.35** and transfer value by **0.30**, each capped at **±3** points. These weights/caps and the -2 replacement assumption are provisional. A productive returner adds nonnegative value regardless of whether last year's team was good or bad; roster continuity never multiplies last year's signed rating. A low-quality returner need not earn a positive bonus.

The provider's BPM already uses adjusted team efficiency, so the model **removes the old hardcoded conference-tier discounts**. Applying them again would duplicate a competition adjustment. Transfer role changes and fit are not separately predicted.

### Recruiting

Use RSCI's final class-of-2026 top-100 consensus table, matched to the **actual 2026–27 roster**, not a spring commitment. There are 101 rows because of ties at #100. A recruit's bonus is `1.25 * ((101 - rank) / 100)^2`, capped at **3 points per team**. It is deliberately top-heavy and modest.

Five reviewed athlete-ID aliases resolve spelling variants or a same-name collision. There is no automatic fuzzy matching. The build matches 98 recruits, excludes one confirmed season-long injury, and leaves two prospects unassigned because they are not on a current roster. These unresolved rows are retained in the audit; they are never credited to a former commitment.

International newcomers outside RSCI and unranked freshmen receive no separate talent bonus.

### Coaching

For every observed coach/team season from 2020–2026, calculate actual final rating minus that year's regressed multiyear baseline. Historical predictions use only earlier team seasons. Estimate a coach's persistent residual using 0.8 annual recency weights and shrink it by `observed seasons / (observed seasons + 4)`.

For a returning coach, the adjustment is **0.35 × estimated residual**. For a coaching change, use **0.35 × (new coach residual − previous coach residual) − 0.5**, capped at **±2** points. History follows the coach ID across schools. A coach without enough observed history is pulled toward neutral. This prevents a replacement from automatically inheriting the departed coach's residual record.

This is a **program-baseline residual proxy**, not a causal estimate of coaching skill: historical roster quality is not controlled. The scale, shrinkage and transition penalty remain provisional. No school or coach has a hand-assigned prestige bonus. Northeastern's current coach is identified from its official roster, but ESPN has no matching coach ID/history in these responses, so its coaching effect is neutral and flagged.

### Availability

Confirmed full-season absences are excluded from both production and recruiting. The snapshot includes Brandon McCoy Jr. (Michigan) and Josh Manchester (Wisconsin), with source links and verification dates. Temporary injuries are not treated as lost seasons. ESPN reports all current roster status flags as active, so those flags cannot serve as a complete availability audit.

The legal eligibility situation remains an explicit limitation. Unless a source-backed exclusion is entered, roster inclusion is assumed. **This is not a complete injury or eligibility census.** All exclusions and their provenance live in `preseason_inputs/2027/v2/availability.csv`.

## Deliberately unchanged or excluded

- No player-development, age, sophomore-jump or class-progression term.
- Preseason weight remains `(1 - games / 20)^1.32`, with zero after 20 games.
- No changes to live CBB outputs or football data.
- No use of target-season results in the candidate.

## Validation scope and next evidence

Fifteen tests cover phaseout compatibility, additive support for a below-average team, transfer quality at equal minutes, small-sample containment, rotation capacity, roster identity collisions, current-season leakage, availability exclusions, recruit matching and departed-coach handling.

The committed historical roster data were not captured on verified preseason dates. Inferring historical rosters from who eventually played would create selection bias. Therefore player/recruit/coach weights have **not** been tuned against those retrospectively observed rotations, and the baseline holdout is not advertised as a full-model backtest. Verified archived preseason rosters and availability snapshots are needed for that test.

`ratings.csv` contains every component, coach-change indicator and movement from the incumbent forecast. `player_audit.csv`, `recruit_audit.csv`, `coach_observations.csv`, `config.json` and `validation.json` preserve the evidence. Source URLs and timing are recorded in the input manifest.
