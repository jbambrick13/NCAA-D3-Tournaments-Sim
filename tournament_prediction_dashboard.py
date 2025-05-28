# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px 
import os 

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
