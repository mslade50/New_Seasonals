import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# User inputs
st.markdown("# Monte Carlo Simulation")
win_prob = st.number_input("Winning probability", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
win_loss_ratio = st.number_input("Win/Loss Ratio", min_value=0.1, max_value=20.0, value=2.0, step=0.01)
start_capital = st.number_input("Starting Capital", min_value=0, max_value=1000000, value=100000, step=100)
num_trials = st.number_input("Number of trials", min_value=0, max_value=100000, value=1000, step=100)
bet_sizing = st.number_input("Bet Sizing (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
num_paths = st.number_input("Number of paths", min_value=0, max_value=10000, value=50, step=5)
stake_type = st.selectbox("Stake Type", ('flat', 'variable'))
num_simulations = st.number_input("Number of simulations", min_value=1, max_value=100, value=1)

def monte_carlo_sim(win_prob, win_loss_ratio, start_capital, num_trials, num_paths, bet_sizing, stake_type, num_simulations):
    simulations = []
    for _ in range(num_simulations):
        paths = []
        negative_endings = 0
        for _ in range(num_paths):
            capital = start_capital
            capital_path = [0]  # Start at 0 for PnL
            for _ in range(num_trials):
                if stake_type == 'flat':
                    bet_size = (bet_sizing / 100) * start_capital  # Flat stake
                else:
                    bet_size = (bet_sizing / 100) * capital  # Variable stake
                outcome = np.random.choice(['win', 'loss'], p=[win_prob, 1-win_prob])
                if outcome == 'win':
                    capital += bet_size * win_loss_ratio
                else:
                    capital -= bet_size
                capital_path.append(capital - start_capital)  # Subtract start capital for PnL
            paths.append(capital_path)
            if capital_path[-1] < 0:  # Check if final value of path is negative
                negative_endings += 1
        simulations.append(paths)
    return simulations

def monte_carlo_app():
    if st.button("Run Simulation"):
        simulations = monte_carlo_sim(win_prob, win_loss_ratio, start_capital, num_trials, num_paths, bet_sizing, stake_type, num_simulations)

        for i, simulation in enumerate(simulations):
            st.markdown(f"### Simulation {i+1}")

            # Calculate the initial expected value (EV)
            initial_bet_size = bet_sizing * start_capital if stake_type == 'flat' else bet_sizing * start_capital
            EV = ((win_prob * initial_bet_size/100 * win_loss_ratio) - ((1 - win_prob) * initial_bet_size/100))
            st.write(f"Initial Expected Value (EV): ${EV}")

            # Calculate the Kelly criterion bet size and round to 2 decimal places
            kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_fraction = round(kelly_fraction, 2)  # round to 2 decimal places
            st.write(f"Kelly criterion bet size (% of capital): {kelly_fraction * 100}%")

            # Calculate the average percentage of paths ending with negative PnL
            negative_endings = [path[-1] < 0 for path in simulation].count(True)
            average_neg_endings = np.mean(negative_endings)
            average_neg_endings_rounded = round(average_neg_endings/num_paths * 100, 2)
            st.write(f"Avg % of paths ending with negative PnL: {average_neg_endings_rounded}%")

            # Calculate the realized EV per trade
            realized_EVs = [path[-1] for path in simulation]  # Profit or loss at the end of each path
            average_realized_EV = np.mean(realized_EVs) / num_trials  # Average profit or loss per trial
            average_realized_EV_rounded = round(average_realized_EV, 2)
            st.write(f"Realized Expected Value (EV) per trade: ${average_realized_EV_rounded}")
            
            # Create DataFrame for Plotly
            df = pd.DataFrame(simulation).T
            df.index.name = "Trial"
            df.columns.name = "Path"
            
            fig = go.Figure()
            
            for path in df.columns:
                fig.add_trace(go.Scatter(y=df[path], mode='lines', name=f'Path {path}'))


            fig.update_layout(height=600, width=800, title_text="Monte Carlo Simulation Paths")
            st.plotly_chart(fig)

# Call the function
monte_carlo_app()
