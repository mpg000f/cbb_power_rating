# =============================================================================
# College Basketball Data Extraction Script
# =============================================================================
# This script pulls transfer portal, recruiting, and returning production data
# from cbbdata/toRvik and exports to CSV for Python consumption.
#
# Setup:
#   install.packages("devtools")
#   devtools::install_github("andreweatherman/cbbdata")
#
# First-time setup:
#   cbbdata::cbd_create_account(username = "your_user", email = "you@email.com", password = "your_pw")
#   # Check email for API key
#
# Usage:
#   Rscript data_extract.R 2025
# =============================================================================

library(cbbdata)
library(dplyr)
library(readr)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
CURRENT_SEASON <- if (length(args) > 0) as.integer(args[1]) else 2025
PRIOR_SEASON <- CURRENT_SEASON - 1
OUTPUT_DIR <- "data"

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

cat("=== CBB Data Extraction ===\n")
cat(sprintf("Current Season: %d\n", CURRENT_SEASON))
cat(sprintf("Prior Season: %d\n", PRIOR_SEASON))
cat(sprintf("Output Directory: %s\n\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# Login to cbbdata API
# -----------------------------------------------------------------------------

# Option 1: Environment variables (recommended)
# Set CBD_USER and CBD_PW in your .Renviron file

# Option 2: Direct login (uncomment and fill in)
# cbd_login(username = "your_username", password = "your_password")

cat("Authenticating with cbbdata API...\n")
tryCatch({
  # This will use environment variables or cached credentials
  cbd_login()
  cat("Authentication successful!\n\n")
}, error = function(e) {
  cat("Authentication failed. Please set up credentials:\n")
  cat("  1. Run: cbbdata::cbd_create_account(username, email, password)\n")
  cat("  2. Set CBD_USER and CBD_PW in .Renviron\n")
  cat("  3. Or use: cbd_login(username, password)\n")
  stop(e)
})

# -----------------------------------------------------------------------------
# 1. Transfer Portal Data
# -----------------------------------------------------------------------------

cat("Fetching transfer portal data...\n")

tryCatch({
  # Get all transfers for current season
  transfers <- cbd_torvik_transfers(year = CURRENT_SEASON)

  # Clean and select relevant columns
  transfers_clean <- transfers %>%
    select(
      player_id,
      player_name,
      from_team = origin_team,
      from_conf = origin_conf,
      to_team = dest_team,
      to_conf = dest_conf,
      year,
      # Previous season stats if available
      matches("ppg|rpg|apg|mpg|min_pct")
    ) %>%
    mutate(
      # Classify transfer tier
      transfer_tier = case_when(
        from_conf %in% c("B12", "B10", "SEC", "ACC", "BE") &
          to_conf %in% c("B12", "B10", "SEC", "ACC", "BE") ~ "high_to_high",
        from_conf %in% c("B12", "B10", "SEC", "ACC", "BE") &
          !to_conf %in% c("B12", "B10", "SEC", "ACC", "BE") ~ "high_to_mid",
        !from_conf %in% c("B12", "B10", "SEC", "ACC", "BE") &
          to_conf %in% c("B12", "B10", "SEC", "ACC", "BE") ~ "mid_to_high",
        TRUE ~ "default"
      )
    )

  write_csv(transfers_clean, file.path(OUTPUT_DIR, "transfers.csv"))
  cat(sprintf("  Saved %d transfers to transfers.csv\n", nrow(transfers_clean)))

}, error = function(e) {
  cat(sprintf("  Warning: Could not fetch transfers - %s\n", e$message))
  # Create empty file
  write_csv(data.frame(), file.path(OUTPUT_DIR, "transfers.csv"))
})

# -----------------------------------------------------------------------------
# 2. Recruiting Rankings
# -----------------------------------------------------------------------------

cat("Fetching recruiting rankings...\n")

tryCatch({
  # Get recruiting rankings for incoming freshmen
  # These are players in their first year
  recruiting <- cbd_torvik_player_season(year = CURRENT_SEASON) %>%
    filter(exp == "Fr") %>%
    select(
      player_id,
      player_name = player,
      team,
      conf,
      matches("rsci|recruit_rank|stars")
    )

  write_csv(recruiting, file.path(OUTPUT_DIR, "recruiting.csv"))
  cat(sprintf("  Saved %d freshmen to recruiting.csv\n", nrow(recruiting)))

}, error = function(e) {
  cat(sprintf("  Warning: Could not fetch recruiting - %s\n", e$message))
})

# Try to get team-level recruiting rankings
tryCatch({
  # Get team recruiting class rankings
  # This aggregates individual recruit rankings to team level
  team_recruiting <- cbd_torvik_ratings(year = CURRENT_SEASON) %>%
    select(
      team,
      conf,
      matches("recruit|class_rank|talent")
    )

  write_csv(team_recruiting, file.path(OUTPUT_DIR, "team_recruiting.csv"))
  cat(sprintf("  Saved %d team recruiting records\n", nrow(team_recruiting)))

}, error = function(e) {
  cat(sprintf("  Note: Team recruiting data not available separately\n"))
})

# -----------------------------------------------------------------------------
# 3. Returning Production / Team Ratings
# -----------------------------------------------------------------------------

cat("Fetching team ratings and returning production...\n")

tryCatch({
  # Get current season ratings (includes returning minutes)
  current_ratings <- cbd_torvik_ratings(year = CURRENT_SEASON)

  # Get prior season ratings for comparison
  prior_ratings <- cbd_torvik_ratings(year = PRIOR_SEASON)

  # Clean current ratings
  current_clean <- current_ratings %>%
    select(
      team,
      conf,
      games = g,
      wins = wins,
      losses = losses,
      adj_oe = adj_o,      # Adjusted offensive efficiency
      adj_de = adj_d,      # Adjusted defensive efficiency
      adj_tempo = adj_t,   # Adjusted tempo
      barthag,             # Win probability vs average team
      proj_barthag = barthag_rk,
      # Returning production columns (names may vary)
      matches("ret|return|exp_rank")
    )

  write_csv(current_clean, file.path(OUTPUT_DIR, "team_ratings_current.csv"))
  cat(sprintf("  Saved %d current team ratings\n", nrow(current_clean)))

  # Clean prior ratings
  prior_clean <- prior_ratings %>%
    select(
      team,
      conf,
      games = g,
      wins = wins,
      losses = losses,
      adj_oe = adj_o,
      adj_de = adj_d,
      adj_tempo = adj_t,
      barthag
    ) %>%
    rename_with(~ paste0("prior_", .), -c(team, conf))

  write_csv(prior_clean, file.path(OUTPUT_DIR, "team_ratings_prior.csv"))
  cat(sprintf("  Saved %d prior team ratings\n", nrow(prior_clean)))

}, error = function(e) {
  cat(sprintf("  Warning: Could not fetch ratings - %s\n", e$message))
})

# -----------------------------------------------------------------------------
# 4. Returning Minutes Detail
# -----------------------------------------------------------------------------

cat("Fetching returning minutes data...\n")

tryCatch({
  # Get player-level data from prior season
  prior_players <- cbd_torvik_player_season(year = PRIOR_SEASON) %>%
    select(
      player_id,
      player_name = player,
      team,
      min_pct,       # Percentage of team's minutes played
      mpg,           # Minutes per game
      games = g,
      ppg, rpg, apg,
      obpr, dbpr     # Offensive/Defensive box plus-minus
    )

  # Calculate returning production by team
  # (Players who are on same team this year)
  current_rosters <- cbd_torvik_player_season(year = CURRENT_SEASON) %>%
    select(player_id, current_team = team)

  returning <- prior_players %>%
    inner_join(current_rosters, by = "player_id") %>%
    filter(team == current_team) %>%
    group_by(team) %>%
    summarise(
      returning_min_pct = sum(min_pct, na.rm = TRUE),
      returning_players = n(),
      returning_ppg = sum(ppg * min_pct, na.rm = TRUE) / 100,
      returning_rpg = sum(rpg * min_pct, na.rm = TRUE) / 100,
      .groups = "drop"
    )

  write_csv(returning, file.path(OUTPUT_DIR, "returning_production.csv"))
  cat(sprintf("  Saved returning production for %d teams\n", nrow(returning)))

}, error = function(e) {
  cat(sprintf("  Warning: Could not calculate returning production - %s\n", e$message))
})

# -----------------------------------------------------------------------------
# 5. Transfer Production (from prior team)
# -----------------------------------------------------------------------------

cat("Calculating transfer production...\n")

tryCatch({
  # Get prior season stats for all players
  prior_players <- cbd_torvik_player_season(year = PRIOR_SEASON) %>%
    select(
      player_id,
      player_name = player,
      prior_team = team,
      prior_conf = conf,
      min_pct,
      mpg,
      ppg, rpg, apg,
      obpr, dbpr
    )

  # Match with transfers
  if (file.exists(file.path(OUTPUT_DIR, "transfers.csv"))) {
    transfers <- read_csv(file.path(OUTPUT_DIR, "transfers.csv"), show_col_types = FALSE)

    transfer_production <- transfers %>%
      left_join(prior_players, by = c("player_id", "from_team" = "prior_team")) %>%
      select(
        player_id,
        player_name = player_name.x,
        from_team,
        to_team,
        transfer_tier,
        prior_min_pct = min_pct,
        prior_mpg = mpg,
        prior_ppg = ppg,
        prior_rpg = rpg,
        prior_apg = apg
      ) %>%
      filter(!is.na(prior_min_pct))

    # Aggregate to team level (for destination teams)
    transfer_by_team <- transfer_production %>%
      group_by(to_team) %>%
      summarise(
        transfer_count = n(),
        transfer_min_pct = sum(prior_min_pct, na.rm = TRUE),
        transfer_ppg = sum(prior_ppg * prior_min_pct, na.rm = TRUE) / 100,
        # Weighted tier (for scaling)
        avg_tier = first(transfer_tier),  # Simplified
        .groups = "drop"
      )

    write_csv(transfer_by_team, file.path(OUTPUT_DIR, "transfer_production.csv"))
    cat(sprintf("  Saved transfer production for %d teams\n", nrow(transfer_by_team)))
  }

}, error = function(e) {
  cat(sprintf("  Warning: Could not calculate transfer production - %s\n", e$message))
})

# -----------------------------------------------------------------------------
# 6. Combined Prior Data for Python
# -----------------------------------------------------------------------------

cat("Creating combined prior data file...\n")

tryCatch({
  # Read all the files we created
  current <- read_csv(file.path(OUTPUT_DIR, "team_ratings_current.csv"), show_col_types = FALSE)
  prior <- read_csv(file.path(OUTPUT_DIR, "team_ratings_prior.csv"), show_col_types = FALSE)

  returning <- if (file.exists(file.path(OUTPUT_DIR, "returning_production.csv"))) {
    read_csv(file.path(OUTPUT_DIR, "returning_production.csv"), show_col_types = FALSE)
  } else {
    data.frame(team = character(), returning_min_pct = numeric())
  }

  transfers <- if (file.exists(file.path(OUTPUT_DIR, "transfer_production.csv"))) {
    read_csv(file.path(OUTPUT_DIR, "transfer_production.csv"), show_col_types = FALSE)
  } else {
    data.frame(to_team = character(), transfer_min_pct = numeric(), avg_tier = character())
  }

  # Combine everything
  combined <- current %>%
    left_join(prior, by = c("team", "conf")) %>%
    left_join(returning, by = "team") %>%
    left_join(transfers, by = c("team" = "to_team")) %>%
    mutate(
      # Fill NAs with defaults
      returning_min_pct = coalesce(returning_min_pct, 0),
      transfer_min_pct = coalesce(transfer_min_pct, 0),
      avg_tier = coalesce(avg_tier, "default"),
      # Calculate prior rating from Barthag (0-1 scale -> 0-100)
      prior_rating = prior_barthag * 100,
      # Calculate effective returning
      # (will be properly calculated in Python with tier scaling)
      raw_effective_returning = (returning_min_pct + transfer_min_pct) / 100
    ) %>%
    select(
      team,
      conf,
      games,
      wins,
      losses,
      adj_oe,
      adj_de,
      adj_tempo,
      prior_rating,
      prior_adj_oe,
      prior_adj_de,
      returning_min_pct,
      transfer_min_pct,
      transfer_tier = avg_tier,
      raw_effective_returning
    )

  write_csv(combined, file.path(OUTPUT_DIR, "combined_prior_data.csv"))
  cat(sprintf("  Saved combined data for %d teams\n", nrow(combined)))

}, error = function(e) {
  cat(sprintf("  Warning: Could not create combined file - %s\n", e$message))
})

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

cat("\n=== Extraction Complete ===\n")
cat("Files created in", OUTPUT_DIR, ":\n")
list.files(OUTPUT_DIR, pattern = "\\.csv$") %>%
  walk(~ cat(sprintf("  - %s\n", .)))

cat("\nNext steps:\n")
cat("  1. Review the CSV files in the data/ directory\n")
cat("  2. Run the Python power rating script\n")
cat("     python power_rating.py --season", CURRENT_SEASON, "\n")
