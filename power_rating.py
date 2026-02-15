"""
College Basketball Power Rating System
======================================
A stability-focused power rating that rewards defensive consistency,
rebounding, and low-variance performance.

Features:
- Opponent-adjusted efficiency metrics (iterative algorithm)
- Prior ratings with transfer/recruiting data from R export
- Configurable weights and decay curves

Data sources:
- Game data: sportsdataverse (ESPN)
- Priors/transfers/recruiting: cbbdata via R export
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import warnings

try:
    import sportsdataverse.mbb as mbb
except ImportError:
    raise ImportError("Install sportsdataverse: pip install sportsdataverse")

try:
    from espn_supplement import fetch_espn_supplement
    ESPN_AVAILABLE = True
except ImportError:
    ESPN_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RatingConfig:
    """Configurable weights and parameters for the power rating system."""

    # Power rating component weights
    # Pure efficiency model: 100% from opponent-adjusted efficiency margin
    # Offense and defense weighted equally (implicit in AdjO - AdjD)
    weight_defensive_efficiency: float = 0.50
    weight_offensive_efficiency: float = 0.50
    # Supporting metrics disabled (set to 0 for pure efficiency model)
    weight_rebounding: float = 0.0
    weight_turnover_margin: float = 0.0
    weight_free_throw_rate: float = 0.0
    weight_consistency: float = 0.0

    # Prior calculation parameters
    prior_regression_factor: float = 0.5  # How much to regress toward mean (0-1)
    prior_decay_games: int = 20  # Games until priors reach zero
    prior_decay_power: float = 1.32  # Decay curve shape (60% current at game 10, 84% at game 15)

    # Transfer production scaling by conference tier
    # Format: scale factor applied to transfer minutes based on origin -> destination
    # Note: Transferring DOWN often means player wasn't cutting it at higher level
    transfer_scale_high_to_high: float = 0.85   # Power to power (known quantity)
    transfer_scale_high_to_mid: float = 0.80    # Power to mid (why leave? probably not a star)
    transfer_scale_high_to_low: float = 0.75    # Power to low (likely couldn't hang)
    transfer_scale_mid_to_high: float = 0.65    # Mid to power (big step up)
    transfer_scale_mid_to_mid: float = 0.80     # Mid to mid (lateral)
    transfer_scale_mid_to_low: float = 0.70     # Mid to low (step down for a reason)
    transfer_scale_low_to_high: float = 0.50    # Low to power (huge step up)
    transfer_scale_low_to_mid: float = 0.65     # Low to mid (step up)
    transfer_scale_low_to_low: float = 0.75     # Low to low (lateral)
    transfer_scale_default: float = 0.70        # Unknown tier

    # Opponent adjustment parameters
    opp_adjust_iterations: int = 15  # Number of iterations for convergence
    opp_adjust_weight: float = 0.8   # How much to weight opponent strength
    opp_adjust_shrinkage: float = 0.0  # Shrinkage toward mean (0 = disabled)

    # SOS-based regression using WIN50 method (Sagarin)
    # WIN50 SOS = rating needed to go .500 against schedule (more robust than averaging)
    # Teams with weak SOS get efficiency regressed toward the mean
    sos_regression_factor: float = 0.04  # Regression per point of SOS below baseline
    sos_boost_factor: float = 0.01  # Small boost per point of SOS above baseline
    sos_max_regression: float = 0.25  # Maximum regression (25% toward mean)

    # Recency weighting - bucket approach
    # Most recent N games get full weight, earlier games fall off
    recency_full_weight_games: int = 10  # Number of recent games at full weight
    recency_weight_min: float = 0.5  # Minimum weight for earliest games
    recency_weight_max: float = 1.0  # Weight for recent games (within bucket)

    # D1 filtering - minimum games to be considered D1
    min_d1_games: int = 10  # Teams with fewer games are filtered out

    # Margin diminishing returns - cap blowouts
    margin_cap_stddev: float = 2.5  # Cap efficiency at X std devs from mean per game

    # Outlier dampening - winsorize extreme games
    outlier_percentile: float = 0.05  # Trim top/bottom X% of games per team

    # Home/away adjustment - adjust raw efficiency for game location
    home_court_efficiency_boost: float = 3.5  # Points per 100 possessions home advantage

    # Use adjusted opponent ratings for SOS (more accurate than raw margins)
    # Raw margins are flawed because weak-conference teams beat each other
    use_raw_margin_sos: bool = False  # Use adjusted opponent ratings for SOS

    # D1 average baseline (will be calculated from data)
    d1_average_rating: float = 50.0
    d1_average_off_eff: float = 100.0  # Average offensive efficiency
    d1_average_def_eff: float = 100.0  # Average defensive efficiency

    # Preseason projection parameters
    use_preseason: bool = True  # Enable preseason blending
    preseason_games_full_weight: int = 20  # Games until current season is 100% weighted
    preseason_returning_avg: float = 60.0  # Expected average effective returning %
    preseason_returning_regression: float = 0.6  # How much to regress low-returning teams

    # Data paths
    r_data_dir: str = "data"
    prior_ratings_path: str = "historical_ratings"  # Path to prior year ratings

    def validate(self):
        """Ensure weights sum to 1.0."""
        total = (
            self.weight_defensive_efficiency +
            self.weight_offensive_efficiency +
            self.weight_rebounding +
            self.weight_turnover_margin +
            self.weight_free_throw_rate +
            self.weight_consistency
        )
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_team_boxscores(seasons: list[int], as_pandas: bool = True) -> pd.DataFrame:
    """Load team box score data from sportsdataverse."""
    print(f"Loading team boxscores for seasons: {seasons}")
    df = mbb.load_mbb_team_boxscore(seasons=seasons, return_as_pandas=as_pandas)
    return df


def load_schedule(seasons: list[int], as_pandas: bool = True) -> pd.DataFrame:
    """Load schedule data from sportsdataverse."""
    print(f"Loading schedule for seasons: {seasons}")
    df = mbb.load_mbb_schedule(seasons=seasons, return_as_pandas=as_pandas)
    return df


def load_r_export_data(config: RatingConfig) -> dict:
    """
    Load data exported from R (cbbdata package).

    Returns dict with keys:
    - combined_prior: Main prior data with returning production
    - transfers: Transfer portal data
    - recruiting: Recruiting rankings
    """
    data_dir = Path(config.r_data_dir)
    result = {}

    # Combined prior data (main file)
    combined_path = data_dir / "combined_prior_data.csv"
    if combined_path.exists():
        result['combined_prior'] = pd.read_csv(combined_path)
        print(f"  Loaded {len(result['combined_prior'])} teams from combined_prior_data.csv")
    else:
        print(f"  Warning: {combined_path} not found. Run data_extract.R first.")
        result['combined_prior'] = None

    # Transfer production by team
    transfer_path = data_dir / "transfer_production.csv"
    if transfer_path.exists():
        result['transfers'] = pd.read_csv(transfer_path)
        print(f"  Loaded {len(result['transfers'])} team transfer records")
    else:
        result['transfers'] = None

    # Returning production
    returning_path = data_dir / "returning_production.csv"
    if returning_path.exists():
        result['returning'] = pd.read_csv(returning_path)
        print(f"  Loaded {len(result['returning'])} team returning production records")
    else:
        result['returning'] = None

    # Prior season ratings
    prior_path = data_dir / "team_ratings_prior.csv"
    if prior_path.exists():
        result['prior_ratings'] = pd.read_csv(prior_path)
        print(f"  Loaded {len(result['prior_ratings'])} prior season ratings")
    else:
        result['prior_ratings'] = None

    return result


def load_prior_season_ratings(current_season: int, config: RatingConfig) -> Optional[pd.DataFrame]:
    """
    Load final ratings from the prior season.

    Args:
        current_season: The current season year (e.g., 2026)
        config: RatingConfig with prior_ratings_path

    Returns:
        DataFrame with prior season ratings, or None if not found
    """
    prior_year = current_season - 1
    prior_path = Path(config.prior_ratings_path) / f"ratings_{prior_year}.csv"

    if prior_path.exists():
        df = pd.read_csv(prior_path)
        print(f"  Loaded {len(df)} prior ratings from {prior_path}")
        return df
    else:
        print(f"  Warning: Prior ratings not found at {prior_path}")
        return None


# =============================================================================
# CONFERENCE TIER LOOKUP (Hardcoded for 2024-25 season)
# =============================================================================

# Power conferences (High tier) - SEC, Big Ten, Big 12, ACC, Big East
HIGH_TIER_TEAMS = {
    # SEC
    333,   # Alabama
    8,     # Arkansas
    2,     # Auburn
    57,    # Florida
    61,    # Georgia
    96,    # Kentucky
    99,    # LSU
    344,   # Mississippi State
    145,   # Ole Miss
    142,   # Missouri
    201,   # Oklahoma
    2579,  # South Carolina
    2633,  # Tennessee
    251,   # Texas
    245,   # Texas A&M
    238,   # Vanderbilt

    # Big Ten
    356,   # Illinois
    84,    # Indiana
    2294,  # Iowa
    120,   # Maryland
    130,   # Michigan
    127,   # Michigan State
    135,   # Minnesota
    158,   # Nebraska
    77,    # Northwestern
    194,   # Ohio State
    2483,  # Oregon
    213,   # Penn State
    2509,  # Purdue
    164,   # Rutgers
    26,    # UCLA
    30,    # USC
    264,   # Washington
    275,   # Wisconsin

    # Big 12
    9,     # Arizona State
    12,    # Arizona
    239,   # Baylor
    252,   # BYU
    2132,  # Cincinnati
    38,    # Colorado
    248,   # Houston
    66,    # Iowa State
    2305,  # Kansas
    2306,  # Kansas State
    197,   # Oklahoma State
    2628,  # TCU
    2641,  # Texas Tech
    2116,  # UCF
    328,   # Utah
    277,   # West Virginia

    # ACC
    103,   # Boston College
    25,    # California
    228,   # Clemson
    150,   # Duke
    52,    # Florida State
    59,    # Georgia Tech
    97,    # Louisville
    2390,  # Miami
    153,   # North Carolina
    152,   # NC State
    87,    # Notre Dame
    221,   # Pittsburgh
    2567,  # SMU
    24,    # Stanford
    183,   # Syracuse
    258,   # Virginia
    259,   # Virginia Tech
    154,   # Wake Forest

    # Big East
    2086,  # Butler
    41,    # UConn
    156,   # Creighton
    305,   # DePaul
    46,    # Georgetown
    269,   # Marquette
    2507,  # Providence
    2550,  # Seton Hall
    2599,  # St. John's
    222,   # Villanova
    2752,  # Xavier
}

# Strong mid-majors (Mid tier) - MWC, WCC, A-10, MVC, AAC, etc.
MID_TIER_TEAMS = {
    # Mountain West
    68,    # Boise State
    278,   # Fresno State
    2439,  # UNLV
    2440,  # Nevada
    2428,  # New Mexico
    21,    # San Diego State
    23,    # San Jose State
    328,   # Utah State (if not Big 12)
    2751,  # Wyoming
    36,    # Colorado State
    2005,  # Air Force

    # WCC
    2250,  # Gonzaga
    2608,  # Saint Mary's
    2541,  # Santa Clara
    2539,  # San Francisco
    2492,  # Pepperdine
    301,   # San Diego
    2501,  # Portland
    2484,  # Pacific
    2857,  # Loyola Marymount

    # Atlantic 10
    2166,  # Davidson
    2168,  # Dayton
    2184,  # Duquesne
    2233,  # Fordham
    2244,  # George Mason
    45,    # George Washington
    2295,  # La Salle
    2352,  # Loyola Maryland (now A-10)
    139,   # Saint Louis
    179,   # St. Bonaventure
    2603,  # Saint Joseph's
    227,   # Rhode Island
    257,   # Richmond
    2670,  # VCU
    2382,  # UMass

    # MVC
    71,    # Bradley
    2181,  # Drake
    339,   # Evansville
    356,   # Illinois State
    282,   # Indiana State
    2623,  # Missouri State
    2460,  # Northern Iowa
    79,    # Southern Illinois
    2674,  # Valparaiso
    2057,  # Belmont

    # AAC
    2429,  # Charlotte
    151,   # East Carolina
    2226,  # FAU
    235,   # Memphis
    249,   # North Texas
    242,   # Rice
    58,    # South Florida
    218,   # Temple
    2655,  # Tulane
    202,   # Tulsa
    5,     # UAB
    2636,  # UTSA
    2724,  # Wichita State

    # Sun Belt
    2026,  # App State
    2032,  # Arkansas State
    324,   # Coastal Carolina
    290,   # Georgia Southern
    2247,  # Georgia State
    2304,  # James Madison
    2335,  # Louisiana
    2433,  # UL Monroe
    2372,  # Marshall
    2349,  # Old Dominion
    2571,  # South Dakota State
    233,   # South Dakota
    2579,  # Southern Miss
    326,   # Texas State
    2653,  # Troy
    6,     # South Alabama

    # MAC
    2006,  # Akron
    2050,  # Ball State
    2066,  # Binghamton (now America East)
    189,   # Bowling Green
    2084,  # Buffalo
    2117,  # Central Michigan
    2197,  # Eastern Michigan
    193,   # Miami (OH)
    94,    # Northern Illinois
    195,   # Ohio
    2649,  # Toledo
    2711,  # Western Michigan
    2310,  # Kent State

    # Ivy League
    43,    # Yale
    227,   # Brown
    171,   # Columbia
    172,   # Cornell
    159,   # Dartmouth
    108,   # Harvard
    219,   # Penn
    163,   # Princeton

    # CAA
    232,   # Charleston
    2169,  # Delaware
    2182,  # Drexel
    2207,  # Elon
    2271,  # Hofstra
    2415,  # Monmouth
    2447,  # Northeastern
    2619,  # Stony Brook
    119,   # Towson
    2729,  # William & Mary
    350,   # UNC Wilmington

    # Other notable mid-majors
    2057,  # Belmont
    91,    # Bellarmine
    2856,  # California Baptist
    2241,  # Gardner-Webb
    2283,  # High Point
    2331,  # Liberty
    2340,  # Lipscomb
    2348,  # Longwood
    2463,  # North Florida
    2617,  # Stephen F. Austin
    2527,  # Radford
    2737,  # Winthrop
    2547,  # Seattle
    2750,  # Wright State
    3084,  # Utah Valley
    2692,  # Weber State
    2534,  # Sam Houston
}


def get_conference_tier(team_id: int, conference_lookup: dict = None) -> str:
    """
    Get conference tier for a team using hardcoded lookup.

    Tiers:
    - 'high': Power conferences (SEC, Big Ten, Big 12, ACC, Big East)
    - 'mid': Strong mid-majors (MWC, WCC, A-10, MVC, AAC, Sun Belt, MAC, Ivy, CAA, etc.)
    - 'low': Low-majors and others

    Returns tier string: 'high', 'mid', or 'low'
    """
    if team_id in HIGH_TIER_TEAMS:
        return 'high'
    elif team_id in MID_TIER_TEAMS:
        return 'mid'
    else:
        return 'low'


def get_transfer_scale(from_tier: str, to_tier: str, config: RatingConfig) -> float:
    """
    Get the scaling factor for transfer production based on conference tiers.

    Transfers moving up in competition level get scaled down more heavily
    since their production may not translate.
    """
    scale_map = {
        ('high', 'high'): config.transfer_scale_high_to_high,
        ('high', 'mid'): config.transfer_scale_high_to_mid,
        ('high', 'low'): config.transfer_scale_high_to_low,
        ('mid', 'high'): config.transfer_scale_mid_to_high,
        ('mid', 'mid'): config.transfer_scale_mid_to_mid,
        ('mid', 'low'): config.transfer_scale_mid_to_low,
        ('low', 'high'): config.transfer_scale_low_to_high,
        ('low', 'mid'): config.transfer_scale_low_to_mid,
        ('low', 'low'): config.transfer_scale_low_to_low,
    }

    return scale_map.get((from_tier, to_tier), config.transfer_scale_default)


def calculate_returning_production(
    current_season: int,
    prior_d1_teams: set = None,
    config: RatingConfig = None
) -> pd.DataFrame:
    """
    Calculate returning production for each team by comparing player minutes
    from prior season to current season rosters.

    Transfer production is scaled based on conference tier transitions:
    - Transfers moving up (low->high, mid->high) are scaled down significantly
    - Transfers moving laterally are scaled moderately
    - Transfers moving down are scaled minimally

    Returns DataFrame with:
    - team_id
    - team_name
    - returning_pct: % of prior year minutes returning (unscaled)
    - transfer_in_pct: % of prior year minutes from incoming transfers (scaled by tier)
    - effective_returning: returning_pct + scaled transfer_in_pct
    """
    if config is None:
        config = RatingConfig()

    prior_year = current_season - 1

    print(f"  Loading {prior_year} player boxscores...")
    p_prior = mbb.load_mbb_player_boxscore(seasons=[prior_year]).to_pandas()

    print(f"  Loading {current_season} player boxscores...")
    p_current = mbb.load_mbb_player_boxscore(seasons=[current_season]).to_pandas()

    # Build conference lookup from team boxscores (need to load separately)
    print(f"  Building conference tier lookup...")
    try:
        team_info_prior = mbb.load_mbb_team_boxscore(seasons=[prior_year]).to_pandas()
        team_info_current = mbb.load_mbb_team_boxscore(seasons=[current_season]).to_pandas()

        # Extract team_id -> conference mapping
        # Try to get conference from the data
        conference_lookup = {}
        for df in [team_info_prior, team_info_current]:
            if 'team_conference' in df.columns:
                for _, row in df[['team_id', 'team_conference']].drop_duplicates().iterrows():
                    conference_lookup[row['team_id']] = row['team_conference']
            elif 'team_conference_id' in df.columns:
                # May need to map conference_id to name
                for _, row in df[['team_id', 'team_conference_id']].drop_duplicates().iterrows():
                    conference_lookup[row['team_id']] = str(row['team_conference_id'])
    except Exception as e:
        print(f"    Warning: Could not build conference lookup: {e}")
        conference_lookup = {}

    # Aggregate minutes per player per team per season
    prior_agg = p_prior.groupby(['team_id', 'team_display_name', 'athlete_id']).agg(
        total_minutes=('minutes', 'sum'),
        points=('points', 'sum')
    ).reset_index()

    current_agg = p_current.groupby(['team_id', 'team_display_name', 'athlete_id']).agg(
        total_minutes=('minutes', 'sum')
    ).reset_index()

    # Get current season players (who's on each team now)
    players_current = set(zip(current_agg['team_id'], current_agg['athlete_id']))

    # Create lookup for player's prior team
    player_prior_team = dict(zip(prior_agg['athlete_id'], prior_agg['team_id']))

    # For each team, calculate returning minutes from prior year
    team_returning = []
    transfer_scale_stats = []  # Track scaling for debugging

    for team_id in prior_agg['team_id'].unique():
        team_prior = prior_agg[prior_agg['team_id'] == team_id]
        team_name = team_prior['team_display_name'].iloc[0]

        total_prior_minutes = team_prior['total_minutes'].sum()

        if total_prior_minutes == 0:
            continue

        # Get destination team's conference tier
        dest_tier = get_conference_tier(team_id, conference_lookup)

        # Check which players are on this team in current season (returning players)
        returning_minutes = 0
        for _, player in team_prior.iterrows():
            if (team_id, player['athlete_id']) in players_current:
                returning_minutes += player['total_minutes']

        # Check for transfers IN (players on current team who were on different team prior)
        team_current = current_agg[current_agg['team_id'] == team_id]
        transfer_in_minutes_raw = 0
        transfer_in_minutes_scaled = 0

        for _, player in team_current.iterrows():
            athlete_id = player['athlete_id']
            prev_team = prior_agg[prior_agg['athlete_id'] == athlete_id]

            if len(prev_team) > 0:
                source_team_id = prev_team['team_id'].iloc[0]
                if source_team_id != team_id:
                    # This is a transfer
                    raw_minutes = prev_team['total_minutes'].iloc[0]
                    transfer_in_minutes_raw += raw_minutes

                    # Get source team's conference tier and apply scaling
                    source_tier = get_conference_tier(source_team_id, conference_lookup)
                    scale = get_transfer_scale(source_tier, dest_tier, config)
                    scaled_minutes = raw_minutes * scale

                    transfer_in_minutes_scaled += scaled_minutes
                    transfer_scale_stats.append({
                        'to_team': team_name,
                        'from_tier': source_tier,
                        'to_tier': dest_tier,
                        'scale': scale,
                        'raw_min': raw_minutes,
                        'scaled_min': scaled_minutes
                    })

        returning_pct = returning_minutes / total_prior_minutes * 100
        transfer_in_pct_raw = transfer_in_minutes_raw / total_prior_minutes * 100
        transfer_in_pct_scaled = transfer_in_minutes_scaled / total_prior_minutes * 100

        team_returning.append({
            'team_id': team_id,
            'team_name': team_name,
            'returning_pct': returning_pct,
            'transfer_in_pct_raw': transfer_in_pct_raw,
            'transfer_in_pct': transfer_in_pct_scaled,  # Scaled version
            'effective_returning': min(returning_pct + transfer_in_pct_scaled, 100)  # Cap at 100%
        })

    df = pd.DataFrame(team_returning)

    # Filter to D1 teams if provided
    if prior_d1_teams is not None:
        df = df[df['team_id'].isin(prior_d1_teams)]

    print(f"  Calculated returning production for {len(df)} teams")
    print(f"    Mean returning: {df['returning_pct'].mean():.1f}%")
    print(f"    Mean transfer-in (raw): {df['transfer_in_pct_raw'].mean():.1f}%")
    print(f"    Mean transfer-in (scaled): {df['transfer_in_pct'].mean():.1f}%")
    print(f"    Mean effective: {df['effective_returning'].mean():.1f}%")

    # Show transfer scaling impact
    if len(transfer_scale_stats) > 0:
        scale_df = pd.DataFrame(transfer_scale_stats)
        tier_summary = scale_df.groupby(['from_tier', 'to_tier']).agg(
            count=('scale', 'count'),
            avg_scale=('scale', 'mean')
        ).reset_index()
        print(f"    Transfer scaling summary:")
        for _, row in tier_summary.iterrows():
            print(f"      {row['from_tier']:>4} -> {row['to_tier']:<4}: {row['count']:3d} transfers, avg scale {row['avg_scale']:.2f}")

    return df


def calculate_preseason_projection(
    prior_ratings: pd.DataFrame,
    returning_production: pd.DataFrame,
    config: RatingConfig
) -> pd.DataFrame:
    """
    Calculate preseason rating projection for each team.

    Formula:
    - Start with prior year's final rating
    - Adjust based on returning production:
      - Teams with high effective_returning (>avg) keep most of prior rating
      - Teams with low effective_returning (<avg) regress toward mean

    Adjustment = (effective_returning - avg) / avg * prior_deviation * regression_factor

    Args:
        prior_ratings: Prior season final ratings
        returning_production: Returning production data
        config: RatingConfig with preseason parameters

    Returns:
        DataFrame with preseason projections
    """
    # Merge prior ratings with returning production
    # Drop any existing effective_returning column to avoid naming conflicts
    prior_cols = [c for c in prior_ratings.columns if c != 'effective_returning']
    df = prior_ratings[prior_cols].merge(
        returning_production[['team_id', 'effective_returning']],
        on='team_id',
        how='left'
    )

    # Fill missing returning production with average
    avg_returning = config.preseason_returning_avg
    df['effective_returning'] = df['effective_returning'].fillna(avg_returning)

    # Calculate deviation from mean rating
    mean_rating = df['power_rating'].mean()
    df['prior_deviation'] = df['power_rating'] - mean_rating

    # Calculate returning adjustment factor
    # Teams with 100% returning keep full deviation
    # Teams with 0% returning lose all deviation (regress to mean)
    # Formula: adjustment = (effective_returning / avg_returning) ^ regression_power
    df['returning_factor'] = (df['effective_returning'] / avg_returning).clip(0, 1.5)

    # Apply regression based on returning production
    # preseason = mean + (prior_deviation * returning_factor * regression_factor)
    regression = config.preseason_returning_regression
    df['preseason_rating'] = mean_rating + (df['prior_deviation'] * df['returning_factor'] * regression)

    # Keep some info for debugging
    df['prior_rating'] = df['power_rating']

    result = df[['team_id', 'team_display_name', 'prior_rating', 'effective_returning',
                 'returning_factor', 'preseason_rating']].copy()

    print(f"  Calculated preseason projections for {len(result)} teams")
    print(f"    Preseason rating range: {result['preseason_rating'].min():.1f} to {result['preseason_rating'].max():.1f}")

    return result


def calculate_prior_weight_preseason(games_played: int, config: RatingConfig) -> float:
    """
    Calculate weight for preseason projection vs current season.

    Power curve decay: weight = (1 - games/max_games)^power

    With power=1.32 and max_games=20:
    - Game 0:  100% preseason
    - Game 10: 40% preseason (60% current)
    - Game 15: 16% preseason (84% current)
    - Game 20: 0% preseason (100% current)

    Args:
        games_played: Number of games played this season
        config: RatingConfig with preseason_games_full_weight and prior_decay_power

    Returns:
        Weight for preseason (0.0 to 1.0)
    """
    max_games = config.preseason_games_full_weight
    power = config.prior_decay_power

    if games_played >= max_games:
        return 0.0

    # Power curve: faster decay early, approaches zero by max_games
    weight = (1 - games_played / max_games) ** power
    return max(0.0, weight)


# =============================================================================
# POSSESSION & EFFICIENCY CALCULATIONS
# =============================================================================

def parse_shooting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Parse/rename shooting stat columns to standard names."""
    df = df.copy()

    # Map column names to standard names
    # sportsdataverse uses separate columns (not combined strings)
    column_map = {
        'field_goals_made': 'fgm',
        'field_goals_attempted': 'fga',
        'three_point_field_goals_made': 'fg3m',
        'three_point_field_goals_attempted': 'fg3a',
        'free_throws_made': 'ftm',
        'free_throws_attempted': 'fta',
    }

    for old_col, new_col in column_map.items():
        if old_col in df.columns:
            df[new_col] = pd.to_numeric(df[old_col], errors='coerce')

    # Handle combined format if present (fallback)
    if 'fga' not in df.columns and 'field_goals_made_field_goals_attempted' in df.columns:
        fg_split = df['field_goals_made_field_goals_attempted'].str.split('-', expand=True)
        df['fgm'] = pd.to_numeric(fg_split[0], errors='coerce')
        df['fga'] = pd.to_numeric(fg_split[1], errors='coerce')

    if 'fg3a' not in df.columns and 'three_point_field_goals_made_three_point_field_goals_attempted' in df.columns:
        three_split = df['three_point_field_goals_made_three_point_field_goals_attempted'].str.split('-', expand=True)
        df['fg3m'] = pd.to_numeric(three_split[0], errors='coerce')
        df['fg3a'] = pd.to_numeric(three_split[1], errors='coerce')

    if 'fta' not in df.columns and 'free_throws_made_free_throws_attempted' in df.columns:
        ft_split = df['free_throws_made_free_throws_attempted'].str.split('-', expand=True)
        df['ftm'] = pd.to_numeric(ft_split[0], errors='coerce')
        df['fta'] = pd.to_numeric(ft_split[1], errors='coerce')

    # Calculate 2-point stats
    if 'fgm' in df.columns and 'fg3m' in df.columns:
        df['fg2m'] = df['fgm'] - df['fg3m']
        df['fg2a'] = df['fga'] - df['fg3a']

    return df


def estimate_possessions(row: pd.Series) -> float:
    """
    Estimate possessions using simplified formula (pre-merge, no opponent stats).
    Uses 0.4 FTA coefficient (KenPom standard) instead of 0.475.
    Possessions ≈ FGA - ORB + TO + (0.4 * FTA)

    This is recalculated after merge with full KenPom formula.
    """
    fga = row.get('fga', 0) or 0
    orb = row.get('offensive_rebounds', 0) or 0
    to = row.get('turnovers', 0) or 0
    fta = row.get('fta', 0) or 0

    return fga - orb + to + (0.4 * fta)


def calculate_kenpom_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate possessions using KenPom's full formula after opponent merge.

    KenPom formula:
    Team Poss = FGA + 0.4*FTA - 1.07*(ORB/(ORB+OppDRB))*(FGA-FGM) + TO
    Game Poss = 0.5 * (Team Poss + Opp Poss)

    This accounts for offensive rebounds more accurately by weighting
    by the probability of getting the rebound.
    """
    df = df.copy()

    # Get FGM (need to calculate from FGA and miss info, or from made shots)
    if 'fgm' not in df.columns:
        # Calculate from 2pt and 3pt makes
        df['fgm'] = df.get('fg2m', 0) + df.get('fg3m', 0)

    # Team possession estimate (KenPom formula)
    orb_plus_opp_drb = df['orb'].fillna(0) + df['opp_drb'].fillna(0)
    orb_rate = np.where(orb_plus_opp_drb > 0, df['orb'].fillna(0) / orb_plus_opp_drb, 0)
    missed_shots = df['fga'].fillna(0) - df['fgm'].fillna(0)

    team_poss = (
        df['fga'].fillna(0) +
        0.4 * df['fta'].fillna(0) -
        1.07 * orb_rate * missed_shots +
        df['turnovers'].fillna(0)
    )

    # Opponent possession estimate
    opp_orb_plus_drb = df['opp_orb'].fillna(0) + df['drb'].fillna(0)
    opp_orb_rate = np.where(opp_orb_plus_drb > 0, df['opp_orb'].fillna(0) / opp_orb_plus_drb, 0)

    # We need opponent FGA and FGM - approximate from efficiency if not available
    if 'opp_fga' in df.columns and 'opp_fgm' in df.columns:
        opp_missed = df['opp_fga'].fillna(0) - df['opp_fgm'].fillna(0)
        opp_poss = (
            df['opp_fga'].fillna(0) +
            0.4 * df.get('opp_fta', df['fta']).fillna(0) -
            1.07 * opp_orb_rate * opp_missed +
            df.get('opp_turnovers', df['turnovers']).fillna(0)
        )
        # Average both teams' estimates
        df['possessions'] = 0.5 * (team_poss + opp_poss)
    else:
        # Fallback: use team estimate only
        df['possessions'] = team_poss

    # Recalculate efficiency with new possessions
    df['off_efficiency'] = np.where(
        df['possessions'] > 0,
        (df['team_score'] / df['possessions']) * 100,
        np.nan
    )

    # Update defensive efficiency too
    if 'opp_score' in df.columns:
        df['def_efficiency'] = np.where(
            df['possessions'] > 0,
            (df['opp_score'] / df['possessions']) * 100,
            df['def_efficiency']
        )

    return df


def calculate_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-game statistics needed for ratings."""
    df = parse_shooting_stats(df)

    # Estimate possessions
    df['possessions'] = df.apply(estimate_possessions, axis=1)

    # Track wins/losses
    if 'team_winner' in df.columns:
        df['win'] = df['team_winner'].astype(int)

    # Calculate points (if not present)
    if 'team_score' not in df.columns:
        df['team_score'] = (df['fg2m'] * 2) + (df['fg3m'] * 3) + df['ftm']
    else:
        df['team_score'] = pd.to_numeric(df['team_score'], errors='coerce')

    # Points per 100 possessions (offensive efficiency)
    df['off_efficiency'] = np.where(
        df['possessions'] > 0,
        (df['team_score'] / df['possessions']) * 100,
        np.nan
    )

    # Turnover rate
    df['turnover_rate'] = np.where(
        df['possessions'] > 0,
        df['turnovers'] / df['possessions'],
        np.nan
    )

    # Rebounds
    df['orb'] = pd.to_numeric(df['offensive_rebounds'], errors='coerce')
    df['drb'] = pd.to_numeric(df['defensive_rebounds'], errors='coerce')
    df['total_rebounds'] = pd.to_numeric(df['total_rebounds'], errors='coerce')

    # Free throw rate
    df['ft_rate'] = np.where(
        df['fga'] > 0,
        df['fta'] / df['fga'],
        np.nan
    )

    # 3-point attempt rate
    df['three_rate'] = np.where(
        df['fga'] > 0,
        df['fg3a'] / df['fga'],
        np.nan
    )

    # 3-point percentage
    df['three_pct'] = np.where(
        df['fg3a'] > 0,
        df['fg3m'] / df['fg3a'],
        np.nan
    )

    return df


def apply_home_away_adjustment(df: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Adjust raw efficiency based on game location.

    Home teams get their offensive efficiency reduced and defensive efficiency increased
    (to neutralize home court advantage in the base ratings).

    Away teams get the opposite adjustment.
    """
    df = df.copy()
    boost = config.home_court_efficiency_boost

    if 'team_home_away' in df.columns:
        # Home games: reduce offensive efficiency, increase defensive efficiency
        # (this neutralizes the home advantage so ratings reflect neutral-court ability)
        home_mask = df['team_home_away'].str.lower() == 'home'
        away_mask = df['team_home_away'].str.lower() == 'away'

        # For home games: team played better than they would on neutral, so adjust down
        df.loc[home_mask, 'off_efficiency'] = df.loc[home_mask, 'off_efficiency'] - boost / 2
        df.loc[home_mask, 'def_efficiency'] = df.loc[home_mask, 'def_efficiency'] + boost / 2

        # For away games: team played worse than they would on neutral, so adjust up
        df.loc[away_mask, 'off_efficiency'] = df.loc[away_mask, 'off_efficiency'] + boost / 2
        df.loc[away_mask, 'def_efficiency'] = df.loc[away_mask, 'def_efficiency'] - boost / 2

    return df


def apply_margin_caps(df: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Cap extreme efficiency values to reduce impact of blowouts.

    Blowouts (40+ point wins) tell us less about team quality because:
    - Starters sit early
    - Effort decreases
    - Garbage time inflates/deflates stats

    Cap efficiency at X standard deviations from the mean.
    """
    df = df.copy()

    for col in ['off_efficiency', 'def_efficiency']:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()

            if std_val > 0:
                lower_cap = mean_val - config.margin_cap_stddev * std_val
                upper_cap = mean_val + config.margin_cap_stddev * std_val

                df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)

    return df


def apply_outlier_dampening(team_games: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Winsorize extreme games for a single team.

    Replace values in the top and bottom X percentile with the percentile boundary.
    This reduces the impact of single-game outliers (e.g., one 120 offensive efficiency game).
    """
    df = team_games.copy()
    pct = config.outlier_percentile

    for col in ['off_efficiency', 'def_efficiency']:
        if col in df.columns and len(df) > 2:
            lower = df[col].quantile(pct)
            upper = df[col].quantile(1 - pct)
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def calculate_raw_margin_sos(game_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Strength of Schedule based on raw point margins, not adjusted ratings.

    This breaks the circular dependency where SOS depends on opponent ratings
    which depend on their SOS.

    Raw margin SOS = average opponent point differential (per game)
    Positive = opponents win more than they lose on average = tough schedule
    """
    # Calculate raw average point differential for each team
    team_margins = game_data.groupby('team_id').agg(
        avg_margin=('point_diff', 'mean'),
        games=('game_id', 'count')
    ).reset_index()

    margin_lookup = dict(zip(team_margins['team_id'], team_margins['avg_margin']))

    # For each team, calculate average opponent margin
    sos_data = []
    for team_id in game_data['team_id'].unique():
        team_games = game_data[game_data['team_id'] == team_id]
        opp_margins = []

        for _, game in team_games.iterrows():
            opp_id = game.get('opp_team_id')
            if opp_id and opp_id in margin_lookup:
                opp_margins.append(margin_lookup[opp_id])

        if opp_margins:
            sos_data.append({
                'team_id': team_id,
                'raw_sos': np.mean(opp_margins)
            })

    return pd.DataFrame(sos_data)


# =============================================================================
# OPPONENT DATA MERGING
# =============================================================================

def merge_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Merge each game with opponent's stats for the same game."""
    df = df.copy()

    # Check if opponent data is already in the dataframe (sportsdataverse format)
    if 'opponent_team_id' in df.columns:
        # Data already has opponent info, just need to get opponent's stats
        df['opp_team_id'] = df['opponent_team_id']
        df['opp_score'] = pd.to_numeric(df['opponent_team_score'], errors='coerce')

        # Self-join to get opponent's calculated stats
        opp_stats = df[['game_id', 'team_id', 'off_efficiency', 'turnover_rate',
                        'orb', 'drb', 'possessions', 'fga', 'fg3a', 'three_pct']].copy()
        opp_stats.columns = ['game_id', 'opp_team_id_check', 'opp_off_efficiency',
                             'opp_turnover_rate', 'opp_orb', 'opp_drb', 'opp_possessions',
                             'opp_fga', 'opp_fg3a', 'opp_three_pct']

        merged = df.merge(opp_stats, on='game_id', how='left')
        # Keep only rows where the opponent ID matches
        merged = merged[merged['opp_team_id'] == merged['opp_team_id_check']]

    else:
        # Fallback: Create opponent stats by matching game_id
        opp_cols = ['game_id', 'team_id', 'team_display_name', 'off_efficiency',
                    'turnover_rate', 'orb', 'drb', 'possessions', 'team_score',
                    'fga', 'fg3a', 'three_pct']

        available_cols = [c for c in opp_cols if c in df.columns]
        opp_df = df[available_cols].copy()

        rename_dict = {
            'team_id': 'opp_team_id',
            'team_display_name': 'opp_team_name',
            'off_efficiency': 'opp_off_efficiency',
            'turnover_rate': 'opp_turnover_rate',
            'orb': 'opp_orb',
            'drb': 'opp_drb',
            'possessions': 'opp_possessions',
            'team_score': 'opp_score',
            'fga': 'opp_fga',
            'fg3a': 'opp_fg3a',
            'three_pct': 'opp_three_pct'
        }
        opp_df = opp_df.rename(columns={k: v for k, v in rename_dict.items() if k in opp_df.columns})

        merged = df.merge(opp_df, on='game_id', how='left')
        merged = merged[merged['team_id'] != merged['opp_team_id']]

    # Defensive efficiency = opponent's offensive efficiency (raw, before adjustment)
    merged['def_efficiency'] = merged['opp_off_efficiency']

    # Calculate rebound rates properly
    # ORB% = ORB / (ORB + Opp DRB)
    merged['orb_rate'] = np.where(
        (merged['orb'] + merged['opp_drb']) > 0,
        merged['orb'] / (merged['orb'] + merged['opp_drb']),
        np.nan
    )

    # DRB% = DRB / (DRB + Opp ORB)
    merged['drb_rate'] = np.where(
        (merged['drb'] + merged['opp_orb']) > 0,
        merged['drb'] / (merged['drb'] + merged['opp_orb']),
        np.nan
    )

    # Turnover margin
    merged['opp_to_rate'] = merged['opp_turnover_rate']
    merged['to_margin'] = merged['opp_to_rate'] - merged['turnover_rate']

    # Point differential for variance calculation
    merged['point_diff'] = merged['team_score'] - merged['opp_score']

    # Calculate win from point differential if team_winner not available
    if 'win' not in merged.columns:
        merged['win'] = (merged['point_diff'] > 0).astype(int)

    return merged


# =============================================================================
# SOS-BASED REGRESSION (Replaces hardcoded conference tiers)
# =============================================================================

def calculate_win_probability(rating_diff: float, home_advantage: float = 0) -> float:
    """
    Calculate win probability given rating difference.

    Uses logistic function calibrated so that ~11 points = 75% win probability
    (similar to KenPom's observed calibration).

    Args:
        rating_diff: Team rating minus opponent rating (positive = favorite)
        home_advantage: Additional points for home team (typically 3.5-4)

    Returns:
        Win probability (0 to 1)
    """
    # Logistic function: P = 1 / (1 + 10^(-diff/scale))
    # Scale of 10 means 10 point favorite ≈ 75% win probability
    adjusted_diff = rating_diff + home_advantage
    return 1 / (1 + 10 ** (-adjusted_diff / 10))


def calculate_win50_sos(game_data: pd.DataFrame, adj_efficiency: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Strength of Schedule using Sagarin's WIN50 method.

    WIN50 SOS = the rating a team would need to go .500 against their schedule.

    This method is more robust than simple averaging because:
    - Outliers (very weak or very strong opponents) have less impact
    - Whether you played #350 vs #351 doesn't matter much
    - Whether you played #1 vs #5 matters more appropriately

    Uses binary search to find the rating that gives 50% expected wins.
    """
    # Create opponent rating lookup
    adj_efficiency = adj_efficiency.copy()
    adj_efficiency['opp_rating'] = adj_efficiency['adj_off_eff'] - adj_efficiency['adj_def_eff']
    rating_lookup = dict(zip(adj_efficiency['team_id'], adj_efficiency['opp_rating']))

    sos_data = []

    for team_id in game_data['team_id'].unique():
        team_games = game_data[game_data['team_id'] == team_id]

        # Get opponent ratings for all games
        opp_ratings = []
        for _, game in team_games.iterrows():
            opp_id = game['opp_team_id']
            opp_rating = rating_lookup.get(opp_id, 0)
            opp_ratings.append(opp_rating)

        if not opp_ratings:
            continue

        num_games = len(opp_ratings)
        target_wins = num_games * 0.5  # We want 50% expected wins

        # Binary search to find rating that gives 50% expected wins
        low, high = -50, 50  # Search range for rating

        for _ in range(50):  # 50 iterations gives very precise result
            mid = (low + high) / 2

            # Calculate expected wins with this rating
            expected_wins = sum(
                calculate_win_probability(mid - opp_rating)
                for opp_rating in opp_ratings
            )

            if expected_wins < target_wins:
                low = mid  # Need higher rating to get more wins
            else:
                high = mid  # Rating is high enough

        win50_sos = (low + high) / 2

        # Also calculate simple average for comparison
        avg_sos = np.mean(opp_ratings)

        sos_data.append({
            'team_id': team_id,
            'sos': win50_sos,  # Use WIN50 as primary SOS
            'sos_avg': avg_sos,  # Keep average for reference
            'sos_std': np.std(opp_ratings) if len(opp_ratings) > 1 else 0,
            'num_games': num_games
        })

    return pd.DataFrame(sos_data)


def calculate_team_sos(game_data: pd.DataFrame, adj_efficiency: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Strength of Schedule for each team.

    Uses WIN50 method (Sagarin) - the rating that would go .500 against the schedule.
    This is more robust than simple averaging of opponent ratings.
    """
    return calculate_win50_sos(game_data, adj_efficiency)


def apply_sos_regression(
    team_stats: pd.DataFrame,
    game_data: pd.DataFrame,
    adj_efficiency: pd.DataFrame,
    config: RatingConfig
) -> pd.DataFrame:
    """
    Apply SOS-based regression to efficiency metrics using WIN50 SOS.

    WIN50 SOS represents the rating a team would need to go .500 against their schedule.
    - Positive WIN50 = tough schedule (need to be good to go .500)
    - Negative WIN50 = weak schedule (could be bad and still go .500)
    - Zero WIN50 = average schedule

    Teams with weak SOS get regressed toward the mean proportionally to how weak
    their schedule is. This is simpler than previous approaches because WIN50
    already handles outliers (playing #350 vs #351 doesn't matter much).
    """
    df = team_stats.copy()

    # Calculate WIN50 SOS for each team
    if config.use_raw_margin_sos:
        sos_df = calculate_raw_margin_sos(game_data)
        sos_df = sos_df.rename(columns={'raw_sos': 'sos'})
    else:
        sos_df = calculate_team_sos(game_data, adj_efficiency)

    # Merge SOS into team stats
    df = df.merge(sos_df, on='team_id', how='left')
    df['sos'] = df['sos'].fillna(0)

    # Use median SOS of top 50 teams as baseline
    # This compares everyone to contender-level schedules, not D1 average
    # Teams playing weak schedules compared to top teams get penalized
    top_teams = df.nlargest(50, 'adj_off_eff')
    baseline_sos = top_teams['sos'].median()

    # Calculate regression based on SOS gap from contender baseline
    # Positive gap = weaker schedule than contenders = regression toward mean
    df['sos_gap'] = baseline_sos - df['sos']  # Positive when schedule is weaker than baseline

    df['sos_regression'] = 0.0

    # Weak schedule: regress toward mean
    # Regression factor scales with how weak the schedule is
    weak_mask = df['sos_gap'] > 0
    weak_gap = df.loc[weak_mask, 'sos_gap']
    regression = (weak_gap * config.sos_regression_factor).clip(0, config.sos_max_regression)
    df.loc[weak_mask, 'sos_regression'] = regression

    # Strong schedule: small boost
    strong_mask = df['sos_gap'] < 0
    strong_gap = df.loc[strong_mask, 'sos_gap'].abs()
    boost = (strong_gap * config.sos_boost_factor).clip(0, 0.10)  # Cap boost at 10%
    df.loc[strong_mask, 'sos_regression'] = -boost  # Negative = boost

    # Apply regression to adjusted efficiency
    if 'adj_off_eff' in df.columns:
        d1_avg_off = df['adj_off_eff'].mean()
        d1_avg_def = df['adj_def_eff'].mean()

        # For weak schedules (positive sos_regression): PENALIZE by moving toward average
        # But only penalize GOOD stats (don't help bad stats by pulling them up)
        # For strong schedules (negative sos_regression): BOOST good stats

        for idx in df.index:
            reg = df.loc[idx, 'sos_regression']
            off = df.loc[idx, 'adj_off_eff']
            deff = df.loc[idx, 'adj_def_eff']

            if reg > 0:  # Weak schedule - penalize good stats only
                # Penalize good offense (above avg) by pulling toward average
                if off > d1_avg_off:
                    df.loc[idx, 'adj_off_eff'] = off - (off - d1_avg_off) * reg

                # Penalize good defense (below avg) by pulling toward average
                if deff < d1_avg_def:
                    df.loc[idx, 'adj_def_eff'] = deff + (d1_avg_def - deff) * reg

            elif reg < 0:  # Strong schedule - boost good stats
                boost = abs(reg)
                # Boost good offense by pushing further above average
                if off > d1_avg_off:
                    df.loc[idx, 'adj_off_eff'] = off + (off - d1_avg_off) * boost

                # Boost good defense by pushing further below average
                if deff < d1_avg_def:
                    df.loc[idx, 'adj_def_eff'] = deff - (d1_avg_def - deff) * boost

    return df


# =============================================================================
# RECENCY WEIGHTING
# =============================================================================

def calculate_recency_weights(df: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Calculate recency weights for each game using a bucketed approach.

    - Most recent N games (per team): full weight (1.0)
    - Earlier games: linearly decrease from 1.0 to min weight

    This ensures recent form is weighted equally (last 10 games all count the same),
    while early-season games when the team was still figuring things out count less.
    """
    df = df.copy()
    df['recency_weight'] = config.recency_weight_max  # Default to full weight

    # Convert game_date to datetime if needed
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

        # For each team in each season, calculate game number from most recent
        for (season, team_id), group in df.groupby(['season', 'team_id']):
            if len(group) == 0:
                continue

            # Sort by date descending (most recent first)
            sorted_idx = group.sort_values('game_date', ascending=False).index

            # Assign game number from most recent (1 = most recent)
            games_from_recent = np.arange(1, len(sorted_idx) + 1)

            # Calculate weights:
            # - Games 1 to N (most recent): full weight
            # - Games N+1 and earlier: linear falloff to min weight
            weights = []
            full_weight_games = config.recency_full_weight_games

            for game_num in games_from_recent:
                if game_num <= full_weight_games:
                    # Recent games: full weight
                    weight = config.recency_weight_max
                else:
                    # Earlier games: linear falloff
                    # At game N+1: start falling off
                    # At first game of season: hit min weight
                    total_games = len(sorted_idx)
                    games_in_falloff = total_games - full_weight_games

                    if games_in_falloff > 0:
                        # How far into the falloff zone (0 = just past bucket, 1 = first game)
                        falloff_position = (game_num - full_weight_games) / games_in_falloff
                        weight = config.recency_weight_max - (
                            (config.recency_weight_max - config.recency_weight_min) * falloff_position
                        )
                    else:
                        weight = config.recency_weight_max

                weights.append(weight)

            df.loc[sorted_idx, 'recency_weight'] = weights

    return df


# =============================================================================
# OPPONENT ADJUSTMENT (ITERATIVE ALGORITHM)
# =============================================================================

def calculate_opponent_adjusted_efficiency(
    game_data: pd.DataFrame,
    config: RatingConfig
) -> pd.DataFrame:
    """
    Calculate opponent-adjusted offensive and defensive efficiency.

    Uses an iterative algorithm similar to KenPom:
    1. Start with raw efficiency for each team
    2. For each game, adjust based on opponent's defensive/offensive strength
    3. Iterate until convergence
    4. Apply recency weighting (later games count more)

    Adjusted OE = Raw OE * (D1 Avg / Opponent's Adj DE)
    Adjusted DE = Raw DE * (D1 Avg / Opponent's Adj OE)
    """
    df = game_data.copy()

    # Calculate recency weights for each game
    df = calculate_recency_weights(df, config)

    # Get unique teams
    teams = df['team_id'].unique()

    # Initialize with recency-weighted raw averages
    def weighted_mean(group, col, weight_col='recency_weight'):
        weights = group[weight_col].fillna(1.0)
        values = group[col]
        valid = ~values.isna()
        if valid.sum() > 0 and weights[valid].sum() > 0:
            return np.average(values[valid], weights=weights[valid])
        return np.nan

    team_stats_list = []
    for team_id in df['team_id'].unique():
        team_games = df[df['team_id'] == team_id]
        team_stats_list.append({
            'team_id': team_id,
            'raw_off_eff': weighted_mean(team_games, 'off_efficiency'),
            'raw_def_eff': weighted_mean(team_games, 'def_efficiency'),
            'games': len(team_games)
        })
    team_stats = pd.DataFrame(team_stats_list)

    # Use 100 as the baseline for both offense and defense
    # This ensures consistent adjustment scale (prevents inflation from data asymmetry)
    d1_avg_off = 100.0
    d1_avg_def = 100.0

    # Initialize adjusted values with raw values
    team_stats['adj_off_eff'] = team_stats['raw_off_eff']
    team_stats['adj_def_eff'] = team_stats['raw_def_eff']

    print(f"  Running opponent adjustment ({config.opp_adjust_iterations} iterations)...")

    # Iterative adjustment
    for iteration in range(config.opp_adjust_iterations):
        prev_off = team_stats['adj_off_eff'].copy()
        prev_def = team_stats['adj_def_eff'].copy()

        # Create lookup dicts for current adjusted values
        off_lookup = dict(zip(team_stats['team_id'], team_stats['adj_off_eff']))
        def_lookup = dict(zip(team_stats['team_id'], team_stats['adj_def_eff']))

        # For each team, recalculate adjusted efficiency based on opponents
        new_adj_off = []
        new_adj_def = []

        for team_id in team_stats['team_id']:
            team_games = df[df['team_id'] == team_id]

            if len(team_games) == 0:
                new_adj_off.append(d1_avg_off)
                new_adj_def.append(d1_avg_def)
                continue

            # Calculate opponent-adjusted offensive efficiency
            # For each game: adjust based on opponent's defensive strength
            # Use recency weighting (later games count more)
            adj_off_values = []
            adj_off_weights = []
            adj_def_values = []
            adj_def_weights = []

            for _, game in team_games.iterrows():
                opp_id = game['opp_team_id']
                weight = game.get('recency_weight', 1.0)

                # Get opponent's adjusted values (or D1 average if not found)
                opp_adj_def = def_lookup.get(opp_id, d1_avg_def)
                opp_adj_off = off_lookup.get(opp_id, d1_avg_off)

                # Adjust this game's offensive efficiency (ADDITIVE adjustment)
                # If opponent has good defense (low adj_def), add the difference
                # Formula: AdjOE = RawOE + (D1_avg - Opp_adj_def)
                if not np.isnan(game['off_efficiency']):
                    adj_off = game['off_efficiency'] + (d1_avg_def - opp_adj_def)
                    adj_off_values.append(adj_off)
                    adj_off_weights.append(weight)

                # Adjust this game's defensive efficiency (ADDITIVE adjustment)
                # If opponent has good offense (high adj_off), subtract the difference
                # Formula: AdjDE = RawDE - (Opp_adj_off - D1_avg)
                if not np.isnan(game['def_efficiency']):
                    adj_def = game['def_efficiency'] - (opp_adj_off - d1_avg_off)
                    adj_def_values.append(adj_def)
                    adj_def_weights.append(weight)

            # Weighted average of adjusted values
            if adj_off_values and sum(adj_off_weights) > 0:
                new_adj_off.append(np.average(adj_off_values, weights=adj_off_weights))
            else:
                new_adj_off.append(d1_avg_off)

            if adj_def_values and sum(adj_def_weights) > 0:
                new_adj_def.append(np.average(adj_def_values, weights=adj_def_weights))
            else:
                new_adj_def.append(d1_avg_def)

        team_stats['adj_off_eff'] = new_adj_off
        team_stats['adj_def_eff'] = new_adj_def

        # Normalize after each iteration to prevent divergence
        # Keep mean at 100 to maintain stable scale
        team_stats['adj_off_eff'] = team_stats['adj_off_eff'] * (100 / team_stats['adj_off_eff'].mean())
        team_stats['adj_def_eff'] = team_stats['adj_def_eff'] * (100 / team_stats['adj_def_eff'].mean())

        # Apply shrinkage toward 100 if configured (0 = disabled for pure mathematical approach)
        shrinkage = config.opp_adjust_shrinkage
        if shrinkage > 0:
            team_stats['adj_off_eff'] = team_stats['adj_off_eff'] * (1 - shrinkage) + 100 * shrinkage
            team_stats['adj_def_eff'] = team_stats['adj_def_eff'] * (1 - shrinkage) + 100 * shrinkage

        # Check convergence
        off_change = np.abs(team_stats['adj_off_eff'] - prev_off).mean()
        def_change = np.abs(team_stats['adj_def_eff'] - prev_def).mean()

        if iteration % 3 == 0 or iteration == config.opp_adjust_iterations - 1:
            print(f"    Iteration {iteration + 1}: avg change = {(off_change + def_change) / 2:.4f}")

        if off_change < 0.01 and def_change < 0.01:
            print(f"    Converged at iteration {iteration + 1}")
            break

    # Normalize so D1 average is 100
    team_stats['adj_off_eff'] = team_stats['adj_off_eff'] * (100 / team_stats['adj_off_eff'].mean())
    team_stats['adj_def_eff'] = team_stats['adj_def_eff'] * (100 / team_stats['adj_def_eff'].mean())

    return team_stats[['team_id', 'adj_off_eff', 'adj_def_eff']]


# =============================================================================
# SEASON AGGREGATION
# =============================================================================

def aggregate_team_season(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-level stats to team-season level."""

    grouped = df.groupby(['team_id', 'team_display_name', 'season'])

    agg_dict = {
        'games_played': ('game_id', 'count'),
        'off_efficiency': ('off_efficiency', 'mean'),
        'def_efficiency': ('def_efficiency', 'mean'),
        'tempo': ('possessions', 'mean'),  # Average possessions per game
        'orb_rate': ('orb_rate', 'mean'),
        'drb_rate': ('drb_rate', 'mean'),
        'to_margin': ('to_margin', 'mean'),
        'ft_rate': ('ft_rate', 'mean'),
        'three_rate': ('three_rate', 'mean'),
        'avg_point_diff': ('point_diff', 'mean'),
        # Variance metrics
        'off_eff_std': ('off_efficiency', 'std'),
        'three_pct_std': ('three_pct', 'std'),
        'point_diff_std': ('point_diff', 'std'),
        # Floor metrics (10th percentile - worst games)
        'off_eff_floor': ('off_efficiency', lambda x: x.quantile(0.10)),
        'def_eff_floor': ('def_efficiency', lambda x: x.quantile(0.90)),  # Higher is worse for defense
        'point_diff_floor': ('point_diff', lambda x: x.quantile(0.10)),
    }

    # Add wins if available
    if 'win' in df.columns:
        agg_dict['wins'] = ('win', 'sum')

    result = grouped.agg(**agg_dict).reset_index()

    # Calculate losses and record string
    if 'wins' in result.columns:
        result['wins'] = result['wins'].astype(int)
        result['losses'] = result['games_played'] - result['wins']
        result['record'] = result['wins'].astype(str) + '-' + result['losses'].astype(str)

    return result


# =============================================================================
# VARIANCE / CONSISTENCY CALCULATION
# =============================================================================

def calculate_consistency_score(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency score (inverse of variance), adjusted for floor quality.

    A team with high variance but a high floor (their bad games are still good)
    shouldn't be penalized as harshly as a team with high variance and a low floor.

    Formula:
        floor_quality = team_floor / d1_average  (>1 means floor is above average)
        penalty_multiplier = 2 - floor_quality   (good floor reduces penalty)
        adjusted_variance = raw_variance * penalty_multiplier
    """
    df = team_stats.copy()

    # Calculate D1 averages for floor comparison
    d1_avg_off = df['off_efficiency'].mean()
    d1_avg_point_diff = df['avg_point_diff'].mean()  # Should be ~0

    # Calculate floor quality for each team
    # Offensive efficiency floor quality (higher floor = better)
    df['off_floor_quality'] = df['off_eff_floor'] / d1_avg_off

    # Point differential floor quality (less negative = better)
    # Normalize so that floor of 0 = quality of 1, floor of -20 = quality of 0.5, etc.
    df['diff_floor_quality'] = 1 + (df['point_diff_floor'] / 40)  # Scale factor of 40
    df['diff_floor_quality'] = df['diff_floor_quality'].clip(0.5, 1.5)

    # Calculate raw variance percentiles (higher = more volatile = worse)
    for col in ['off_eff_std', 'three_pct_std', 'point_diff_std']:
        if col in df.columns:
            df[f'{col}_pctile'] = df[col].rank(pct=True)

    # Calculate penalty multipliers based on floor quality
    # Good floor (quality > 1) -> multiplier < 1 -> reduced penalty
    # Bad floor (quality < 1) -> multiplier > 1 -> increased penalty
    df['off_penalty_mult'] = (2 - df['off_floor_quality']).clip(0.5, 1.5)
    df['diff_penalty_mult'] = (2 - df['diff_floor_quality']).clip(0.5, 1.5)

    # For 3pt variance, use offensive floor as proxy (good offensive team = more leeway)
    df['three_penalty_mult'] = df['off_penalty_mult']

    # Apply floor-adjusted penalties
    df['off_eff_adj_var'] = df.get('off_eff_std_pctile', 0.5) * df['off_penalty_mult']
    df['three_pct_adj_var'] = df.get('three_pct_std_pctile', 0.5) * df['three_penalty_mult']
    df['point_diff_adj_var'] = df.get('point_diff_std_pctile', 0.5) * df['diff_penalty_mult']

    # Re-normalize adjusted variances to 0-1 scale
    for col in ['off_eff_adj_var', 'three_pct_adj_var', 'point_diff_adj_var']:
        max_val = df[col].max()
        if max_val > 0:
            df[col] = df[col] / max_val

    # Weighted consistency score (inverted so high = good)
    # Weights: scoring variance 40%, 3pt variance 35%, margin variance 25%
    df['consistency_score'] = (
        (1 - df['off_eff_adj_var']) * 0.40 +
        (1 - df['three_pct_adj_var']) * 0.35 +
        (1 - df['point_diff_adj_var']) * 0.25
    )

    return df


# =============================================================================
# PRIOR CALCULATIONS WITH R DATA
# =============================================================================

def calculate_prior_weight(games_played: int, config: RatingConfig) -> float:
    """
    Calculate prior weight based on games played.
    Uses power curve: 1 - (games/20)^0.756
    """
    if games_played >= config.prior_decay_games:
        return 0.0

    weight = 1 - (games_played / config.prior_decay_games) ** config.prior_decay_power
    return max(0.0, weight)


def calculate_effective_returning(
    returning_pct: float,
    transfer_pct: float,
    transfer_tier: str = 'default',
    config: RatingConfig = None
) -> float:
    """
    Calculate effective returning production including transfers.
    """
    if config is None:
        config = RatingConfig()

    tier_scales = {
        'high_to_high': config.transfer_scale_high_to_high,
        'mid_to_high': config.transfer_scale_mid_to_high,
        'high_to_mid': config.transfer_scale_high_to_mid,
        'default': config.transfer_scale_default,
    }

    scale = tier_scales.get(transfer_tier, config.transfer_scale_default)
    effective = returning_pct + (transfer_pct * scale)

    return min(1.0, effective)


def build_prior_ratings(r_data: dict, config: RatingConfig) -> Optional[pd.DataFrame]:
    """
    Build prior ratings from R-exported data.

    Uses:
    - Prior season ratings (Barthag-based)
    - Returning production percentage
    - Transfer production with tier scaling
    """
    if r_data.get('combined_prior') is None:
        return None

    df = r_data['combined_prior'].copy()

    # Calculate effective returning for each team
    def calc_effective(row):
        returning = row.get('returning_min_pct', 0) / 100  # Convert to decimal
        transfer = row.get('transfer_min_pct', 0) / 100
        tier = row.get('transfer_tier', 'default')
        return calculate_effective_returning(returning, transfer, tier, config)

    df['effective_returning'] = df.apply(calc_effective, axis=1)

    # Calculate D1 average from prior ratings
    if 'prior_rating' in df.columns:
        d1_avg = df['prior_rating'].mean()
    else:
        d1_avg = config.d1_average_rating

    # Calculate adjusted prior rating with regression
    def calc_prior(row):
        if pd.isna(row.get('prior_rating')):
            return d1_avg

        last_year = row['prior_rating']
        eff_returning = row['effective_returning']

        gap = last_year - d1_avg
        lost_production = 1 - eff_returning
        regression = gap * lost_production * config.prior_regression_factor

        return last_year - regression

    df['adjusted_prior_rating'] = df.apply(calc_prior, axis=1)

    return df[['team', 'adjusted_prior_rating', 'effective_returning',
               'returning_min_pct', 'transfer_min_pct', 'transfer_tier']]


# =============================================================================
# POWER RATING CALCULATION
# =============================================================================

def normalize_metric(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize a metric to 0-100 scale using percentile rank."""
    pctile = series.rank(pct=True)
    if higher_is_better:
        return pctile * 100
    else:
        return (1 - pctile) * 100


def calculate_power_rating(
    team_stats: pd.DataFrame,
    config: RatingConfig,
    adj_efficiency: Optional[pd.DataFrame] = None,
    prior_data: Optional[pd.DataFrame] = None,
    preseason_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate power rating for each team.

    Rating is scaled so that the difference between two teams' ratings
    approximates the expected point spread (at neutral court).

    Formula: Rating = (AdjOE - AdjDE) * tempo_factor + supporting_metrics_bonus
    Where tempo_factor ≈ 0.70 (games have ~70 possessions, not 100)

    If preseason_data is provided, blends current season with preseason projection
    based on games played (more games = more weight on current season).
    """
    config.validate()
    df = team_stats.copy()

    # Merge adjusted efficiency if passed separately
    if adj_efficiency is not None:
        df = df.merge(adj_efficiency, on='team_id', how='left')

    # Use adjusted efficiency values if available
    if 'adj_off_eff' in df.columns:
        df['off_efficiency'] = df['adj_off_eff'].fillna(df['off_efficiency'])
    if 'adj_def_eff' in df.columns:
        df['def_efficiency'] = df['adj_def_eff'].fillna(df['def_efficiency'])

    # Calculate efficiency margin (points per 100 possessions above opponent)
    # AdjOE is points scored per 100 poss (higher = better)
    # AdjDE is points allowed per 100 poss (lower = better)
    # For margin, we want (offense above avg) + (defense below avg)
    # If avg = 100, then margin = (AdjOE - 100) + (100 - AdjDE) = AdjOE - AdjDE
    d1_avg = 100  # D1 average efficiency

    df['efficiency_margin'] = df['off_efficiency'] - df['def_efficiency']

    # Keep as per-100-possessions (no tempo adjustment here)
    # Spread predictions should use matchup-specific tempo
    df['margin_per_game'] = df['efficiency_margin']

    # Pure efficiency model: power rating = efficiency margin only
    # No supporting metrics (rebounding, TO margin, FT rate, consistency)
    # These are set to 0 but kept for backwards compatibility
    df['rebound_bonus'] = 0.0
    df['to_margin_bonus'] = 0.0
    df['ft_rate_bonus'] = 0.0
    df['consistency_bonus'] = 0.0
    df['supporting_bonus'] = 0.0

    # Power rating = pure efficiency margin (AdjO - AdjD per 100 possessions)
    df['power_rating_raw'] = df['margin_per_game']

    # Center around zero (so average D1 team = 0, elite teams = +15 to +25)
    df['power_rating_raw'] = df['power_rating_raw'] - df['power_rating_raw'].mean()

    # Apply preseason blending if available (new approach)
    if preseason_data is not None:
        # Merge preseason projections on team_id
        df = df.merge(
            preseason_data[['team_id', 'preseason_rating', 'effective_returning']],
            on='team_id',
            how='left'
        )

        # Calculate prior weight based on games played
        # Linear decay: 100% preseason at 0 games, 0% at preseason_games_full_weight
        df['prior_weight'] = df['games_played'].apply(
            lambda x: calculate_prior_weight_preseason(x, config)
        )

        # Fill missing preseason ratings with 0 (D1 average after centering)
        df['preseason_rating'] = df['preseason_rating'].fillna(0)

        # Blend preseason and current season
        df['power_rating'] = (
            df['preseason_rating'] * df['prior_weight'] +
            df['power_rating_raw'] * (1 - df['prior_weight'])
        )

    # Legacy prior support (R-exported data)
    elif prior_data is not None:
        # Try to match on team name
        df = df.merge(
            prior_data[['team', 'adjusted_prior_rating', 'effective_returning']],
            left_on='team_display_name',
            right_on='team',
            how='left'
        )

        df['prior_weight'] = df['games_played'].apply(
            lambda x: calculate_prior_weight(x, config)
        )

        df['prior_rating'] = df['adjusted_prior_rating'].fillna(config.d1_average_rating)

        df['power_rating'] = (
            df['prior_rating'] * df['prior_weight'] +
            df['power_rating_raw'] * (1 - df['prior_weight'])
        )
    else:
        # No preseason component - pure current season
        df['power_rating'] = df['power_rating_raw']
        df['prior_weight'] = 0.0
        df['effective_returning'] = np.nan

    return df


# =============================================================================
# SPREAD PREDICTION
# =============================================================================

def predict_spread(
    team_a: str,
    team_b: str,
    ratings_df: pd.DataFrame,
    neutral_court: bool = True,
    home_team: str = None
) -> dict:
    """
    Predict the spread for a matchup using tempo-adjusted efficiency.

    Formula:
        Expected Tempo = (Team A Tempo × Team B Tempo) / National Avg Tempo
        Spread = (Team A Rating - Team B Rating) × (Expected Tempo / 100)

    Args:
        team_a: First team name (partial match supported)
        team_b: Second team name (partial match supported)
        ratings_df: DataFrame with power ratings and tempo
        neutral_court: If True, no home court adjustment
        home_team: If not neutral, which team is home ('a' or 'b')

    Returns:
        dict with spread prediction details
    """
    HOME_COURT_ADVANTAGE = 3.5  # Points of home court advantage

    # Find teams (partial match)
    team_a_row = ratings_df[ratings_df['team_display_name'].str.contains(team_a, case=False, na=False)]
    team_b_row = ratings_df[ratings_df['team_display_name'].str.contains(team_b, case=False, na=False)]

    if len(team_a_row) == 0:
        raise ValueError(f"Team not found: {team_a}")
    if len(team_b_row) == 0:
        raise ValueError(f"Team not found: {team_b}")

    # Take first match if multiple
    team_a_row = team_a_row.iloc[0]
    team_b_row = team_b_row.iloc[0]

    # Get ratings and tempo
    rating_a = team_a_row['power_rating']
    rating_b = team_b_row['power_rating']
    tempo_a = team_a_row.get('tempo', 70.0)
    tempo_b = team_b_row.get('tempo', 70.0)

    # National average tempo (use average from the dataframe or default)
    if 'tempo' in ratings_df.columns:
        national_avg_tempo = ratings_df['tempo'].mean()
    else:
        national_avg_tempo = 70.0

    # Calculate expected game tempo
    expected_tempo = (tempo_a * tempo_b) / national_avg_tempo

    # Calculate raw spread (per 100 possessions -> actual game)
    raw_spread = (rating_a - rating_b) * (expected_tempo / 100)

    # Apply home court adjustment if applicable
    if not neutral_court and home_team:
        if home_team.lower() == 'a':
            raw_spread += HOME_COURT_ADVANTAGE
        elif home_team.lower() == 'b':
            raw_spread -= HOME_COURT_ADVANTAGE

    return {
        'team_a': team_a_row['team_display_name'],
        'team_b': team_b_row['team_display_name'],
        'team_a_rating': round(rating_a, 2),
        'team_b_rating': round(rating_b, 2),
        'team_a_tempo': round(tempo_a, 1),
        'team_b_tempo': round(tempo_b, 1),
        'expected_tempo': round(expected_tempo, 1),
        'spread': round(raw_spread, 1),
        'neutral_court': neutral_court,
        'favorite': team_a_row['team_display_name'] if raw_spread > 0 else team_b_row['team_display_name'],
        'line': f"{team_a_row['team_display_name']} {raw_spread:+.1f}" if raw_spread != 0 else "Pick'em"
    }


def print_matchup(prediction: dict) -> None:
    """Print a matchup prediction in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"MATCHUP: {prediction['team_a']} vs {prediction['team_b']}")
    print(f"{'=' * 60}")
    print(f"  {prediction['team_a']:<25} Rating: {prediction['team_a_rating']:>6.2f}")
    print(f"  {prediction['team_b']:<25} Rating: {prediction['team_b_rating']:>6.2f}")
    print(f"  Rating Diff (per 100 poss): {prediction['team_a_rating'] - prediction['team_b_rating']:>+6.2f}")
    print(f"  Expected Tempo: {prediction['expected_tempo']:.1f} possessions")
    print(f"  {'Neutral Court' if prediction['neutral_court'] else 'Home Court Applied'}")
    print(f"  -" * 30)
    print(f"  PREDICTED SPREAD: {prediction['line']}")
    print(f"{'=' * 60}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def filter_d1_teams(df: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Filter out non-D1 teams based on games played.

    D1 teams typically play 25-35 games per season. Teams with very few games
    are likely D2/D3 teams that only appear in the dataset from games against D1.
    """
    # Count games per team
    team_games = df.groupby('team_id').size().reset_index(name='game_count')

    # Keep only teams with enough games
    d1_teams = team_games[team_games['game_count'] >= config.min_d1_games]['team_id']

    # Filter the dataframe
    filtered = df[df['team_id'].isin(d1_teams)]

    removed_count = df['team_id'].nunique() - filtered['team_id'].nunique()
    if removed_count > 0:
        print(f"  Filtered out {removed_count} non-D1 teams (fewer than {config.min_d1_games} games)")

    return filtered


def filter_non_d1_opponents(df: pd.DataFrame, d1_team_ids: set) -> pd.DataFrame:
    """
    Filter out games against non-D1 opponents.

    Games against D2/D3/NAIA teams inflate efficiency numbers and should not
    be included in rating calculations.
    """
    before_count = len(df)
    filtered = df[df['opp_team_id'].isin(d1_team_ids)]
    removed_count = before_count - len(filtered)

    if removed_count > 0:
        print(f"  Removed {removed_count} games against non-D1 opponents")

    return filtered


def run_power_ratings(
    current_season: int,
    prior_season: Optional[int] = None,
    config: RatingConfig = None,
    use_r_data: bool = True
) -> pd.DataFrame:
    """
    Run the full power rating pipeline.
    """
    if config is None:
        config = RatingConfig()

    seasons_to_load = [current_season]

    # Load game data from sportsdataverse
    print("Loading game data from sportsdataverse...")
    boxscores = load_team_boxscores(seasons_to_load)

    # Supplement with ESPN data for recent games (current season only)
    espn_schedule_supplement = None
    if ESPN_AVAILABLE and current_season == 2026:
        try:
            espn_box, espn_sched = fetch_espn_supplement(boxscores, current_season)
            if not espn_box.empty:
                boxscores = pd.concat([boxscores, espn_box], ignore_index=True)
                boxscores = boxscores.drop_duplicates(
                    subset=['game_id', 'team_id'], keep='first'
                )
            if not espn_sched.empty:
                espn_schedule_supplement = espn_sched
        except Exception as e:
            print(f"  ESPN supplement failed ({e}), continuing with sportsdataverse only")

    # Calculate game-level stats
    print("Calculating game statistics...")
    game_stats = calculate_game_stats(boxscores)

    # Merge opponent data
    print("Merging opponent data...")
    with_opponents = merge_opponent_stats(game_stats)

    # Recalculate possessions using KenPom formula (requires opponent stats)
    print("Recalculating possessions (KenPom formula)...")
    with_opponents = calculate_kenpom_possessions(with_opponents)

    # Filter to current season
    current_games = with_opponents[with_opponents['season'] == current_season]

    # Calculate ALL-GAMES records from schedule data (more up-to-date than PBP)
    print("Calculating all-games records from schedule...")
    try:
        schedule = load_schedule([current_season])
        if espn_schedule_supplement is not None and not espn_schedule_supplement.empty:
            schedule = pd.concat([schedule, espn_schedule_supplement], ignore_index=True)
            if 'game_id' in schedule.columns:
                schedule = schedule.drop_duplicates(subset=['game_id'], keep='first')
        completed = schedule[
            schedule['home_score'].notna() & schedule['away_score'].notna() &
            ((schedule['home_score'] > 0) | (schedule['away_score'] > 0))
        ].copy()
        records = {}
        for _, game in completed.iterrows():
            home_id = game.get('home_id')
            away_id = game.get('away_id')
            home_score = game['home_score']
            away_score = game['away_score']
            if pd.isna(home_id) or pd.isna(away_id) or home_score == away_score:
                continue
            home_id = int(home_id)
            away_id = int(away_id)
            for tid in [home_id, away_id]:
                if tid not in records:
                    records[tid] = {'wins': 0, 'losses': 0}
            if home_score > away_score:
                records[home_id]['wins'] += 1
                records[away_id]['losses'] += 1
            else:
                records[away_id]['wins'] += 1
                records[home_id]['losses'] += 1
        all_games_records = pd.DataFrame([
            {'team_id': tid, 'total_games': r['wins'] + r['losses'],
             'total_wins': r['wins'], 'total_losses': r['losses'],
             'record': f"{r['wins']}-{r['losses']}"}
            for tid, r in records.items()
        ])
        print(f"  Got records for {len(all_games_records)} teams from schedule")
    except Exception as e:
        print(f"  Schedule records failed ({e}), falling back to PBP records...")
        if 'win' in current_games.columns:
            all_games_records = current_games.groupby('team_id').agg(
                total_games=('game_id', 'nunique'),
                total_wins=('win', 'sum')
            ).reset_index()
        else:
            all_games_records = current_games.groupby('team_id').agg(
                total_games=('game_id', 'nunique'),
                total_wins=('point_diff', lambda x: (x > 0).sum())
            ).reset_index()
        all_games_records['total_losses'] = all_games_records['total_games'] - all_games_records['total_wins']
        all_games_records['record'] = all_games_records['total_wins'].astype(int).astype(str) + '-' + all_games_records['total_losses'].astype(int).astype(str)

    # Filter out non-D1 teams
    print("Filtering D1 teams...")
    current_games = filter_d1_teams(current_games, config)

    # Get set of D1 team IDs for opponent filtering
    d1_team_ids = set(current_games['team_id'].unique())

    # Filter out games against non-D1 opponents
    print("Filtering games against non-D1 opponents...")
    current_games = filter_non_d1_opponents(current_games, d1_team_ids)

    # Apply home/away adjustment to neutralize home court in base ratings
    print("Applying home/away adjustment...")
    current_games = apply_home_away_adjustment(current_games, config)

    # Apply margin caps to reduce blowout impact
    print("Applying margin caps (diminishing returns on blowouts)...")
    current_games = apply_margin_caps(current_games, config)

    # Apply outlier dampening per team (winsorize extreme games)
    print("Applying outlier dampening...")
    dampened_games = []
    for team_id in current_games['team_id'].unique():
        team_data = current_games[current_games['team_id'] == team_id].copy()
        team_data = apply_outlier_dampening(team_data, config)
        dampened_games.append(team_data)
    current_games = pd.concat(dampened_games, ignore_index=True)

    # Calculate opponent-adjusted efficiency (with recency weighting)
    print("Calculating opponent-adjusted efficiency (with recency weighting)...")
    adj_efficiency = calculate_opponent_adjusted_efficiency(current_games, config)

    # Aggregate to team-season level
    print("Aggregating to team-season level...")
    team_season = aggregate_team_season(current_games)

    # Apply SOS-based regression (replaces hardcoded conference tiers)
    print("Applying SOS-based regression...")
    team_season = team_season.merge(adj_efficiency, on='team_id', how='left')
    team_season = apply_sos_regression(team_season, current_games, adj_efficiency, config)

    # Calculate consistency scores
    print("Calculating consistency scores...")
    team_season = calculate_consistency_score(team_season)

    # Load preseason projections (prior ratings + returning production)
    preseason_data = None
    if config.use_preseason:
        print("Loading preseason projections...")
        prior_ratings = load_prior_season_ratings(current_season, config)

        if prior_ratings is not None:
            # Get D1 teams from prior season for filtering
            prior_d1_teams = set(prior_ratings['team_id'].unique())

            print("Calculating returning production...")
            try:
                returning_production = calculate_returning_production(
                    current_season,
                    prior_d1_teams=prior_d1_teams,
                    config=config
                )

                print("Calculating preseason projections...")
                preseason_data = calculate_preseason_projection(
                    prior_ratings,
                    returning_production,
                    config
                )
            except Exception as e:
                print(f"  Warning: Could not calculate returning production: {e}")
                print("  Falling back to prior ratings without returning adjustment")
                # Use prior ratings directly without returning adjustment
                preseason_data = prior_ratings[['team_id', 'team_display_name', 'power_rating']].copy()
                preseason_data['preseason_rating'] = preseason_data['power_rating']
                preseason_data['effective_returning'] = config.preseason_returning_avg
        else:
            print("  No prior ratings available, skipping preseason component")

    # Legacy R-data support (fallback)
    prior_data = None
    if use_r_data and preseason_data is None:
        print("Loading R-exported prior data...")
        r_data = load_r_export_data(config)
        prior_data = build_prior_ratings(r_data, config)

        if prior_data is not None:
            print(f"  Built prior ratings for {len(prior_data)} teams")

    # Calculate power ratings
    print("Calculating power ratings...")
    results = calculate_power_rating(
        team_season,
        config,
        adj_efficiency=None,  # Already merged in team_season
        prior_data=prior_data,
        preseason_data=preseason_data
    )

    # Sort by power rating
    results = results.sort_values('power_rating', ascending=False).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)

    # Replace filtered records with all-games records
    if 'record' in results.columns:
        results = results.drop(columns=['record'])
    results = results.merge(
        all_games_records[['team_id', 'record']],
        on='team_id',
        how='left'
    )

    return results


def display_ratings(df: pd.DataFrame, top_n: int = 25) -> None:
    """Display top teams in a readable format."""
    display_cols = [
        'rank', 'team_display_name', 'power_rating', 'games_played',
        'adj_off_eff', 'adj_def_eff', 'tempo', 'consistency_score'
    ]

    # Fall back to raw efficiency if adjusted not available
    if 'adj_off_eff' not in df.columns:
        display_cols = [c.replace('adj_off_eff', 'off_efficiency')
                           .replace('adj_def_eff', 'def_efficiency')
                       for c in display_cols]

    available_cols = [c for c in display_cols if c in df.columns]

    print("\n" + "=" * 100)
    print(f"TOP {top_n} POWER RATINGS (Per 100 Possessions, Opponent-Adjusted)")
    print("=" * 100)

    display_df = df[available_cols].head(top_n).copy()

    # Format numeric columns
    numeric_cols = ['power_rating', 'off_efficiency', 'def_efficiency',
                    'adj_off_eff', 'adj_def_eff', 'tempo', 'consistency_score', 'prior_weight']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    print(display_df.to_string(index=False))

    # Print some summary stats
    print("\n" + "-" * 100)
    print("Rating = Efficiency margin per 100 possessions | Use tempo to calculate expected spreads")
    if 'tempo' in df.columns:
        avg_tempo = df['tempo'].mean()
        print(f"National Avg Tempo: {avg_tempo:.1f} possessions/game")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate college basketball power ratings")
    parser.add_argument('--season', type=int, default=2025, help='Season to analyze')
    parser.add_argument('--top', type=int, default=25, help='Number of top teams to display')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--no-priors', action='store_true', help='Skip R-exported prior data')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for R-exported data')
    parser.add_argument('--spread', nargs=2, metavar=('TEAM_A', 'TEAM_B'),
                        help='Predict spread between two teams (use quotes for team names)')
    parser.add_argument('--home', type=str, choices=['a', 'b'],
                        help='Home team for spread prediction (a or b), omit for neutral')
    parser.add_argument('--load-ratings', type=str,
                        help='Load existing ratings CSV instead of recalculating')

    args = parser.parse_args()

    # Load or calculate ratings
    if args.load_ratings:
        print(f"Loading ratings from {args.load_ratings}...")
        results = pd.read_csv(args.load_ratings)
    else:
        # Run the pipeline
        config = RatingConfig(r_data_dir=args.data_dir)

        results = run_power_ratings(
            current_season=args.season,
            config=config,
            use_r_data=not args.no_priors
        )

    # Spread prediction mode
    if args.spread:
        team_a, team_b = args.spread
        try:
            prediction = predict_spread(
                team_a, team_b, results,
                neutral_court=(args.home is None),
                home_team=args.home
            )
            print_matchup(prediction)
        except ValueError as e:
            print(f"Error: {e}")
    else:
        # Display results
        display_ratings(results, top_n=args.top)

    # Save to CSV if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
