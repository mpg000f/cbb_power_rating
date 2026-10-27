# College basketball preseason ratings

`python preseason.py --season 2027` builds the 2026–27 forecast (basketball uses the ending year). `python build_static.py --skip-sports cfb,nfl` exports it to the existing site. Inputs are cached; `--refresh-inputs` explicitly fetches new rosters.

The baseline retains the existing model's formula: prior centered efficiency margin multiplied by `0.6 * clip(effective_returning / 60, 0, 1.5)`, then recentered across current D1 teams. Effective returning is prior minutes retained plus incoming transfer minutes scaled by the existing conference-tier factors, capped at 100%. This is a continuity-based projection; freshmen, coaching changes, injuries and individual player quality are not separately modeled. Transfer minutes are experience, not projected rotation minutes.

The input folder retains ESPN season-specific rosters (with athlete IDs and retrieval time) and aggregated prior-season sportsdataverse player minutes. No current-season results enter the preseason forecast. Missing or wrong-season rosters, duplicate roster assignments, and missing prior data stop the build. The 2027 membership adjustment removes Saint Francis (PA) and adds West Florida. West Florida's provisional prior is the median rating of the prior year's bottom quartile, with league-average tempo; it bypasses continuity regression because it lacks D1 history. This is a modeling assumption, not measured D1 strength.

Sources:
- ESPN rosters: `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{id}/roster?season=2027`
- Prior player minutes: `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_player_boxscores/player_box_2026.parquet`
- Prior ratings: this repository's `historical_ratings/ratings_2026.csv`.

The frozen `ratings_2027_preseason.csv` is loaded during the season. Preseason weight is `(1 - games/20)^1.32`, floored at zero after 20 games. Unplayed teams retain their baseline. Preseason offensive/defensive splits are deliberately absent, and tempo is explicitly inherited from the prior season for matchup spread estimates. The site labels preseason projections and hides the efficiency scatter until splits exist.

Checks: `python -m unittest test_preseason -v`.
