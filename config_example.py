"""
Example configuration for tuning the power rating system.
Copy this file to config.py and adjust weights to your preference.
"""

from power_rating import RatingConfig

# Create custom configuration
config = RatingConfig(
    # Power rating component weights (must sum to 1.0)
    weight_defensive_efficiency=0.25,  # Defense matters most
    weight_offensive_efficiency=0.20,   # Offense still important
    weight_rebounding=0.20,             # Controls possessions
    weight_turnover_margin=0.12,        # Ball security
    weight_free_throw_rate=0.08,        # Getting to the line
    weight_consistency=0.15,            # Low variance bonus

    # Prior calculation
    prior_regression_factor=0.5,  # 0.3=light, 0.5=moderate, 0.7=heavy
    prior_decay_games=20,         # Games until priors = 0
    prior_decay_power=0.756,      # Curve shape (0.756 = 50% at game 8)

    # Transfer production scaling
    transfer_scale_high_to_high=0.80,  # Duke transfer to Kansas
    transfer_scale_mid_to_high=0.65,   # Mid-major star to power conf
    transfer_scale_high_to_mid=0.90,   # Power conf player to mid-major
    transfer_scale_default=0.75,       # Default if tier unknown
)

# Validate weights sum to 1.0
config.validate()

# Alternative: Defense-heavy configuration
defense_heavy_config = RatingConfig(
    weight_defensive_efficiency=0.30,
    weight_offensive_efficiency=0.15,
    weight_rebounding=0.22,
    weight_turnover_margin=0.13,
    weight_free_throw_rate=0.08,
    weight_consistency=0.12,
)

# Alternative: Consistency-focused configuration
consistency_config = RatingConfig(
    weight_defensive_efficiency=0.22,
    weight_offensive_efficiency=0.18,
    weight_rebounding=0.18,
    weight_turnover_margin=0.12,
    weight_free_throw_rate=0.08,
    weight_consistency=0.22,  # Higher consistency weight
)
