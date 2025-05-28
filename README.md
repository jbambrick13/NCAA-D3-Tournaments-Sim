# NCAA D3 Men's Lacrosse Conference Tournament Outcome Predictor

## Overview

This project simulates 5 of the main NCAA Division III Men's Lacrosse conference tournaments using a Monte Carlo method based on team Elo ratings. It calculates the probability of each participating team winning their respective conference championship. The results are then presented in an interactive web dashboard built with Streamlit.

The primary goal of this project was to apply data analysis, simulation techniques, and data visualization skills to a real-world sports scenario that I am passionate about.

**Live Dashboard Link:** [Link to  deployed Streamlit Dashboard]
## Features

* **Monte Carlo Simulation:** Runs a configurable number of simulations (usually set to 1000) for each tournament to estimate championship probabilities.
* **Elo-Based Predictions:** Utilizes team Elo ratings to calculate win probabilities for individual games using the standard Elo formula.
* **Multiple Conference Support:** Simulates distinct tournament structures for the following conferences:
    * Centennial Conference
    * NESCAC
    * Coastal Conference
    * Liberty Conference
    * MAC Commonwealth
* **Data-Driven:** Uses provided game data to derive the latest Elo ratings for teams leading into the simulated tournaments.
* **Interactive Dashboard:** A Streamlit application allows users to select a conference and view:
    * Championship win probabilities for each team in a bar chart.
    * A detailed table showing team names, initial Elo ratings, championship counts from simulations, and final win probabilities.
* **Data Export:** The main simulation script saves the probability results for each conference to CSV files, which are then used by the dashboard.

## Methodology

1.  **Data Loading & Preparation:**
    * Team list and historical game data (including pre-calculated Elo ratings per game) are loaded using Pandas.
    * The most recent Elo rating for each team is extracted from the game data to serve as the input for the tournament simulations.
2.  **Elo Win Probability:**
    * The probability of team `i` winning against team `j` is calculated using the formula:
        `P(i wins) = 1 / (1 + 10^((Elo_j - Elo_i) / 400))`
3.  **Tournament Simulation:**
    * Each conference tournament's unique bracket structure (including number of teams, seeds, and byes) is hardcoded.
    * For each matchup in a tournament round, a random number (0-1) is generated and compared against the calculated win probability to determine the simulated winner.
    * Winners advance through the bracket until a tournament champion is determined.
4.  **Monte Carlo Analysis:**
    * The entire tournament simulation for each conference is repeated many times (usually 1000).
    * The number of times each team wins the championship is recorded.
    * The final championship probability for each team is calculated as `(Number of times team won championship) / (Total number of successful simulations)`.
5.  **Dashboard Display:**
    * The aggregated results are presented using Streamlit, with interactive charts created using Plotly Express.

## Data Sources

The project uses two primary CSV files:

* `JacksonBambrickGameData20250415.csv`: Contains game-level statistics, including team IDs and their Elo ratings before each game.
* `JacksonBambrickTeamList20250415.csv`: A lookup table mapping team IDs to human-readable team names.

These files are processed to create a dataset of the latest available Elo rating for each team prior to the simulated tournaments.

## Technologies Used

* **Python 3.13.1**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Streamlit:** For building the interactive web dashboard.
* **Plotly Express:** For creating interactive charts within the dashboard.
* **Git & GitHub:** For version control and project hosting.
* **VS Code:** As the primary code editor.
