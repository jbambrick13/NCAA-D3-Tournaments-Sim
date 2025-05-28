# Import necessary libraries
import streamlit as st
import plotly.express as px 
import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict 
import os 

# Part 1: Load Data and Create Dataframe with Latest Elos 

print("--- Part 1: Loading Data and Calculating Latest Elo ---")

# --- Load Datasets ---
# Initialize variables and df to hold raw games data
df_games = None 
df_teams = None
df_latest_elo = None
df_games_raw = None 
# Try to read in game and team data from CSV files
try:
    game_data_filename = r"C:\Users\jacks\Tournament_Prediction_Visualization\JacksonBambrickGameData20250502.csv"
    team_list_filename = r"C:\Users\jacks\Tournament_Prediction_Visualization\JacksonBambrickTeamList20250415.csv"
    df_games_raw = pd.read_csv(game_data_filename)
    df_teams = pd.read_csv(team_list_filename)
    print("Successfully loaded raw datasets.")
# Handle most likely error
except FileNotFoundError:
    print("Error: One or more files not found.")
    print(f"- {game_data_filename}")
    print(f"- {team_list_filename}")
except Exception as e:
    print(f"An error occurred while loading the files: {e}")

# --- Filter Game Data Columns ---
if df_games_raw is not None:
    columns_to_keep = [
        'game_date', 'team_ID', 'opp_team_ID',
        'team_elo_rating', 'opp_elo_rating'
    ]
    missing_cols = [col for col in columns_to_keep if col not in df_games_raw.columns]
    if not missing_cols:
        # Create new DataFrame with only the essential columns
        df_games = df_games_raw[columns_to_keep].copy()
        print(f"\nFiltered raw game data to keep only {len(columns_to_keep)} essential columns.")
    # Handle most likely error
    else:
        print(f"\nError: Could not filter columns (missing: {missing_cols}).")
        df_games = None
else:
    print("\nSkipping column filtering because raw game data failed to load.")

# --- Create Latest Elo Data ---
if df_games is not None and df_teams is not None:
    print("\nCalculating latest Elo ratings for each team...")
    try:
        # Convert 'game_date' to datetime and filter out invalid dates
        df_games['game_date'] = pd.to_datetime(df_games['game_date'], errors='coerce')
        original_rows = len(df_games)
        df_games.dropna(subset=['game_date'], inplace=True)
        if len(df_games) < original_rows:
            print(f"Warning: Removed {original_rows - len(df_games)} rows with invalid date formats.")

        if not df_games.empty:
            # Create a DataFrame with each team's latest game date and Elo rating
            latest_game_indices = df_games.loc[df_games.groupby('team_ID')['game_date'].idxmax()]
            latest_elo_data_temp = latest_game_indices[['team_ID', 'team_elo_rating']].copy()
            latest_elo_data_temp.rename(columns={'team_elo_rating': 'latest_elo_rating'}, inplace=True)
            # Create a DataFrame with the latest Elo ratings for each team
            df_latest_elo = pd.merge(latest_elo_data_temp, df_teams, on='team_ID', how='left')
            # Ensure display_name and latest_elo_rating exist before selecting
            if 'display_name' in df_latest_elo.columns and 'latest_elo_rating' in df_latest_elo.columns:
                 df_latest_elo = df_latest_elo[['display_name', 'latest_elo_rating']].copy()
                 print("\nSuccessfully created DataFrame 'df_latest_elo'.")
            else:
                 print("Error: 'display_name' or 'latest_elo_rating' column missing after merge.")
                 df_latest_elo = None
        else:
            print("Error: No valid game data remaining after date conversion.")
            df_latest_elo = None
    except Exception as e:
        print(f"\nAn unexpected error occurred while creating latest Elo data: {e}")
        df_latest_elo = None
elif df_games is None:
    print("\nSkipping latest Elo creation because filtered game data is not available.")
elif df_teams is None:
    print("\nSkipping latest Elo creation because team list data is not available.")


# Part 2: Run Monte Carlo Simulations and Store Results


print("\n" + "="*70)
print("--- Part 2: Monte Carlo Simulations ---")

# Check if df_latest_elo was created successfully 
if df_latest_elo is None:
    print("\nError: The 'df_latest_elo' DataFrame is not available from Part 1.")
    print("Cannot proceed with tournament simulations.")
else:
    # Configuration 
    num_simulations = 1000  
    verbose_simulation = False 

    # --- Helper Functions ---
    def calculate_win_prob(elo_i, elo_j):
        """Calculates win probability for team i against team j."""
        try:
            elo_i = float(elo_i); elo_j = float(elo_j)
            return 1.0 / (1.0 + math.pow(10, (elo_j - elo_i) / 400.0))
        except (ValueError, TypeError, OverflowError):
            return None

    def simulate_game(team_i_name, elo_i, team_j_name, elo_j, verbose=False):
        """Simulates a single game and returns winner {'name': name, 'elo': elo} or None."""
        prob_i_wins = calculate_win_prob(elo_i, elo_j)
        if prob_i_wins is None:
            if verbose: print(f"  Could not simulate game ({team_i_name} vs {team_j_name}) due to invalid Elo.")
            return None

        if verbose:
            print(f"  Simulating: {team_i_name} ({elo_i:.2f}) vs {team_j_name} ({elo_j:.2f}) | P({team_i_name} wins): {prob_i_wins:.4f}")

        random_outcome = random.random()
        winner = {'name': team_i_name, 'elo': elo_i} if random_outcome < prob_i_wins else {'name': team_j_name, 'elo': elo_j}

        if verbose: print(f"    Result: {winner['name']} wins!")
        return winner

    # --- Define Fixed Seeds for All Conferences ---
    conference_seeds = {
        "Centennial": {
            1: "Gettysburg", 2: "Dickinson", 3: "Swarthmore",
            4: "Muhlenberg", 5: "Franklin & Marshall"
        },
        "NESCAC": {
            1: "Tufts", 2: "Bowdoin", 3: "Wesleyan", 4: "Amherst",
            5: "Middlebury", 6: "Bates", 7: "Hamilton", 8: "Williams"
        },
        "Coastal": {
            1: "Salisbury", 2: "Christopher Newport", 3: "Kean",
            4: "Mary Washington", 5: "Stockton", 6: "Montclair State"
        },
        "Liberty": {
            1: "RIT", 2: "St. Lawrence", 3: "RPI", 4: "Skidmore", 5: "Ithaca"
        },
        "MAC Commonwealth": {
            1: "York", 2: "Stevenson", 3: "Eastern", 4: "Widener"
        }
    }

    # --- Validate All Teams and Get Elo Ratings ---
    print("\n--- Validating Teams and Retrieving Elo Ratings ---")
    all_teams_valid = True
    # Create dictionairy to store Elos for all teams in each conference
    master_team_elos = {} 
    # Create a set of all required teams from the conference seeds to avoid duplicates
    required_teams = set()
    for conf, seeds_dict in conference_seeds.items(): 
        for seed, team_name in seeds_dict.items():
            required_teams.add(team_name)
    # Create a list of all team names 
    valid_teams_in_elo_df = df_latest_elo['display_name'].tolist()
    # Extract Elo ratings for each required team
    for team_name in required_teams:
        if team_name in valid_teams_in_elo_df:
            try:
                elo = df_latest_elo.loc[df_latest_elo['display_name'] == team_name, 'latest_elo_rating'].iloc[0]
                master_team_elos[team_name] = elo
            except Exception as e:
                 print(f"  *** ERROR: Could not retrieve Elo for '{team_name}': {e} ***")
                 all_teams_valid = False
        else:
            print(f"  *** FATAL ERROR: Required team '{team_name}' not found in df_latest_elo! ***")
            all_teams_valid = False

    # --- Define Conference Simulation Functions ---
    # Each function uses simulate_game to simulate the tournament for that conference
    # Tournamentt structures are fixed and based on real tournament formats

    def simulate_centennial(current_seeds, current_team_elos, verbose=False): 
        """Simulates the Centennial Conference tournament."""
        if verbose: print("\n-- Centennial Simulation --")
        winner_g1 = simulate_game(current_seeds[4], current_team_elos[current_seeds[4]], current_seeds[5], current_team_elos[current_seeds[5]], verbose)
        if not winner_g1: return None
        winner_sf1 = simulate_game(current_seeds[1], current_team_elos[current_seeds[1]], winner_g1['name'], winner_g1['elo'], verbose)
        winner_sf2 = simulate_game(current_seeds[2], current_team_elos[current_seeds[2]], current_seeds[3], current_team_elos[current_seeds[3]], verbose)
        if not winner_sf1 or not winner_sf2: return None
        champion = simulate_game(winner_sf1['name'], winner_sf1['elo'], winner_sf2['name'], winner_sf2['elo'], verbose)
        return champion['name'] if champion else None

    def simulate_nescac(current_seeds, current_team_elos, verbose=False):
        """Simulates the NESCAC tournament."""
        if verbose: print("\n-- NESCAC Simulation --")
        # Intialize dictionairy to store quarterfinal winners
        qf_winners = {}
        qf_winners['1v8'] = simulate_game(current_seeds[1], current_team_elos[current_seeds[1]], current_seeds[8], current_team_elos[current_seeds[8]], verbose)
        qf_winners['2v7'] = simulate_game(current_seeds[2], current_team_elos[current_seeds[2]], current_seeds[7], current_team_elos[current_seeds[7]], verbose)
        qf_winners['3v6'] = simulate_game(current_seeds[3], current_team_elos[current_seeds[3]], current_seeds[6], current_team_elos[current_seeds[6]], verbose)
        qf_winners['4v5'] = simulate_game(current_seeds[4], current_team_elos[current_seeds[4]], current_seeds[5], current_team_elos[current_seeds[5]], verbose)
        if not all(qf_winners.values()): return None
        sf1_winner = simulate_game(qf_winners['1v8']['name'], qf_winners['1v8']['elo'], qf_winners['4v5']['name'], qf_winners['4v5']['elo'], verbose)
        sf2_winner = simulate_game(qf_winners['2v7']['name'], qf_winners['2v7']['elo'], qf_winners['3v6']['name'], qf_winners['3v6']['elo'], verbose)
        if not sf1_winner or not sf2_winner: return None
        champion = simulate_game(sf1_winner['name'], sf1_winner['elo'], sf2_winner['name'], sf2_winner['elo'], verbose)
        return champion['name'] if champion else None

    def simulate_coastal(current_seeds, current_team_elos, verbose=False):
        """Simulates the Coastal Conference tournament."""
        if verbose: print("\n-- Coastal Simulation --")
        winner_g1 = simulate_game(current_seeds[3], current_team_elos[current_seeds[3]], current_seeds[6], current_team_elos[current_seeds[6]], verbose)
        winner_g2 = simulate_game(current_seeds[5], current_team_elos[current_seeds[5]], current_seeds[4], current_team_elos[current_seeds[4]], verbose)
        if not winner_g1 or not winner_g2: return None
        sf1_winner = simulate_game(current_seeds[1], current_team_elos[current_seeds[1]], winner_g1['name'], winner_g1['elo'], verbose)
        sf2_winner = simulate_game(current_seeds[2], current_team_elos[current_seeds[2]], winner_g2['name'], winner_g2['elo'], verbose)
        if not sf1_winner or not sf2_winner: return None
        champion = simulate_game(sf1_winner['name'], sf1_winner['elo'], sf2_winner['name'], sf2_winner['elo'], verbose)
        return champion['name'] if champion else None

    def simulate_liberty(current_seeds, current_team_elos, verbose=False):
        """Simulates the Liberty Conference tournament."""
        if verbose: print("\n-- Liberty Simulation --")
        winner_g1 = simulate_game(current_seeds[4], current_team_elos[current_seeds[4]], current_seeds[5], current_team_elos[current_seeds[5]], verbose)
        if not winner_g1: return None
        sf1_winner = simulate_game(current_seeds[1], current_team_elos[current_seeds[1]], winner_g1['name'], winner_g1['elo'], verbose)
        sf2_winner = simulate_game(current_seeds[2], current_team_elos[current_seeds[2]], current_seeds[3], current_team_elos[current_seeds[3]], verbose)
        if not sf1_winner or not sf2_winner: return None
        champion = simulate_game(sf1_winner['name'], sf1_winner['elo'], sf2_winner['name'], sf2_winner['elo'], verbose)
        return champion['name'] if champion else None

    def simulate_mac_commonwealth(current_seeds, current_team_elos, verbose=False):
        """Simulates the MAC Commonwealth tournament."""
        if verbose: print("\n-- MAC Commonwealth Simulation --")
        sf1_winner = simulate_game(current_seeds[1], current_team_elos[current_seeds[1]], current_seeds[4], current_team_elos[current_seeds[4]], verbose)
        sf2_winner = simulate_game(current_seeds[2], current_team_elos[current_seeds[2]], current_seeds[3], current_team_elos[current_seeds[3]], verbose)
        if not sf1_winner or not sf2_winner: return None
        champion = simulate_game(sf1_winner['name'], sf1_winner['elo'], sf2_winner['name'], sf2_winner['elo'], verbose)
        return champion['name'] if champion else None

    conference_sim_functions = {
        "Centennial": simulate_centennial,
        "NESCAC": simulate_nescac,
        "Coastal": simulate_coastal,
        "Liberty": simulate_liberty,
        "MAC Commonwealth": simulate_mac_commonwealth
    }

    # --- Run Monte Carlo Simulation ---
    championship_counts = defaultdict(lambda: defaultdict(int))
    simulation_errors = defaultdict(int)

    if not all_teams_valid:
        print("\nHalting simulation due to missing team data or errors retrieving Elo.")
    else:
        print(f"\n--- Running {num_simulations} simulations for {len(conference_seeds)} conferences ---")

        for i in range(num_simulations):
            if (i + 1) % (num_simulations // 10 or 1) == 0:
                 print(f"  Running simulation {i+1}/{num_simulations}...")

            if verbose_simulation: print(f"\n===== Simulation Run #{i+1} =====")

            for conf_name, current_conf_seeds in conference_seeds.items(): # Renamed variable
                sim_function = conference_sim_functions[conf_name]
                champion_name = sim_function(current_conf_seeds, master_team_elos, verbose=verbose_simulation)

                if champion_name:
                    championship_counts[conf_name][champion_name] += 1
                else:
                    simulation_errors[conf_name] += 1
                    if verbose_simulation: print(f"  * Simulation Error in {conf_name} Run #{i+1}")

        print(f"\nFinished {num_simulations} simulations.")

        # --- Display Results and Save to CSV ---
        print("\n" + "="*70)
        print("--- Monte Carlo Simulation Results ---")
        print(f"Total Simulations Run per Conference: {num_simulations}")
        print("="*70)

        for conf_name, counts in championship_counts.items():
            print(f"\n--- {conf_name} Conference Results ---")
            conf_errors = simulation_errors[conf_name]
            successful_sims = num_simulations - conf_errors

            if conf_errors > 0:
                print(f"  (Note: {conf_errors} simulation runs for this conference encountered errors)")

            if not counts or successful_sims <= 0:
                 print("  No valid simulation results recorded for this conference.")
                 continue

            results_list = []
            for team, wins in counts.items():
                 initial_elo = master_team_elos.get(team, 'N/A')
                 probability = wins / successful_sims if successful_sims > 0 else 0
                 results_list.append({'Team': team, 'Initial Elo': initial_elo, 'Championships': wins, 'Probability': probability})

            results_df = pd.DataFrame(results_list)
            try:
                results_df['Initial Elo'] = pd.to_numeric(results_df['Initial Elo'], errors='coerce')
                elo_format = "%.2f"
            except:
                 elo_format = None

            results_df.sort_values(by='Probability', ascending=False, inplace=True)
            print(results_df.to_string(index=False, float_format="%.4f", formatters={'Initial Elo': lambda x: elo_format % x if pd.notna(x) and elo_format else x}))

            # --- Save Results to CSV ---
            try:
                safe_conf_name = conf_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                output_filename = f"{safe_conf_name}_results.csv"
                results_df.to_csv(output_filename, index=False)
                print(f"  Successfully saved results for {conf_name} to {output_filename}")
            except Exception as e:
                print(f"  Error saving results for {conf_name} to CSV: {e}")


# Set the title and layout for the browser tab and page
st.set_page_config(page_title="NCAA D3 Lacrosse Tournament Predictions", layout="wide")

# --- Load Data ---
# Dictionary to hold the results DataFrame for each conference
conference_data = {}
# List of conference names 
conference_names = ["Centennial", "NESCAC", "Coastal", "Liberty", "MAC Commonwealth"]
# Base directory where the script and CSV files are located
base_dir = os.path.dirname(__file__) 

print("Loading simulation results...") 

all_files_loaded = True
for conf_name in conference_names:
    # Create the expected filename
    safe_conf_name = conf_name.replace(" ", "_")
    file_path = os.path.join(base_dir, f"{safe_conf_name}_results.csv")

    try:
        # Check if the file exists before trying to load
        if os.path.exists(file_path):
            conference_data[conf_name] = pd.read_csv(file_path)
            print(f"  Loaded: {file_path}")
        else:
            st.error(f"Error: Result file not found for {conf_name} at {file_path}")
            print(f"  Error: File not found - {file_path}")
            conference_data[conf_name] = pd.DataFrame() # Add empty DataFrame to avoid key errors later
            all_files_loaded = False 

    except Exception as e:
        st.error(f"Error loading data for {conf_name}: {e}")
        print(f"  Error loading {file_path}: {e}")
        conference_data[conf_name] = pd.DataFrame() # Add empty DataFrame
        all_files_loaded = False

# --- Dashboard Title ---
st.title("NCAA D3 Men's Lacrosse Tournament Predictions")
st.markdown("Using actual tournament seedings and formats, conference winners for 5 conferences were predicted using a Monte Carlo simulation. " \
" Winning probabilites for each game were calculated using an ELO based formula. Simulation ran 1000 times for each conference.")


# --- Sidebar for Conference Selection ---
st.sidebar.header("Select Conference")
# Create a dropdown list with the conference names
selected_conference = st.sidebar.selectbox(
    "Choose a conference:",
    options=conference_names,
    # Default to the first conference in the list
    index=0 
)

# --- Display Results for Selected Conference ---
st.header(f"{selected_conference} Conference Championship Probability")
# Get the DataFrame for the selected conference
results_df = conference_data.get(selected_conference)
if results_df is not None and not results_df.empty:
    # Ensure probability is numeric for formatting/charting
    results_df['Probability'] = pd.to_numeric(results_df['Probability'], errors='coerce')
    results_df.dropna(subset=['Probability'], inplace=True) # Remove rows where conversion failed

    # --- Bar Chart ---
    st.subheader("Win Probability Distribution")

    # Create an interactive bar chart to display each team's conference championship probability
    fig = px.bar(
        results_df,
        x='Team',
        y='Probability',
        title=f"{selected_conference} Championship Win Probability",
        labels={'Probability': 'Probability (%)', 'Team': 'Team Name'},
        hover_data={'Initial Elo': ':.2f', 'Championships': True, 'Probability': ':.4f'}, 
        text='Probability' 
    )

    # Format the text on bars as percentage and adjust appearance
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        xaxis_title="Team",
        yaxis_title="Championship Probability",
        yaxis_tickformat=".0%", # Format y-axis as percentage
        xaxis={'categoryorder':'total descending'} # Order bars by probability
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # --- Data Table ---
    st.subheader("Detailed Results From 1000 Simulations")

    # Select and format columns for display
    display_df = results_df[['Team', 'Initial Elo', 'Championships', 'Probability']].copy()
    # Format the Probability column as a percentage string for better readability in the table
    display_df['Probability'] = display_df['Probability'].map('{:.2%}'.format)
    display_df.rename(columns={'Probability': 'Win Probability (%)'}, inplace=True)


    # Display the DataFrame as a table
    # Use st.dataframe for interactive table, st.table for static
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif not all_files_loaded:
     st.warning("Some result files could not be loaded. Display may be incomplete.")
else:
    st.warning(f"No simulation data available for the selected conference: {selected_conference}")


# --- Note on simulation limitations---
st.sidebar.markdown("---")
st.sidebar.info("Simulation based on pre-tournament Elo ratings.  Does not account for injuries, home field advantadge, or other dynamic factors.")

print("Dashboard script setup complete.")
