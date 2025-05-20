import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# === Helper functions ===
def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def calculate_kelly(fair_prob, win_loss_ratio):
    """Calculate Kelly criterion fraction."""
    return (fair_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio

def run_monte_carlo(fair_prob, win_loss_ratio, bankroll, kelly_fractions, num_trials):
    results = {}
    for kf in kelly_fractions:
        final_pnls = []
        for _ in range(1000):
            capital = bankroll
            for _ in range(num_trials):
                bet_size = kf * bankroll
                outcome = np.random.rand() < fair_prob
                capital += bet_size * win_loss_ratio if outcome else -bet_size
            final_pnls.append(capital - bankroll)
        results[kf] = final_pnls
    return results

# === Streamlit app ===
st.markdown("# Kelly Betting Calculator")

# Input section
col1, col2 = st.columns(2)
with col1:
    prob_input = st.number_input("Fair win probability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
with col2:
    american_odds = st.number_input("American Odds (optional)", value=0)

if american_odds != 0:
    prob_input = american_to_prob(american_odds)

win_loss_ratio = st.number_input("Win/Loss Ratio (Payout - 1)", min_value=0.1, max_value=20.0, value=1.0, step=0.01)
bankroll = st.number_input("Bankroll ($)", min_value=1, max_value=10000000, value=10000, step=100)
fraction_kelly = st.number_input("Fraction of Kelly to Use", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
num_trials = st.number_input("Number of Monte Carlo Trials", min_value=1, max_value=10000, value=100, step=10)

if st.button("Calculate Bet and Run Simulation"):
    # Kelly calculation
    kelly_fraction = calculate_kelly(prob_input, win_loss_ratio)
    recommended_bet = bankroll * kelly_fraction * fraction_kelly

    st.write(f"**Full Kelly Fraction:** {round(kelly_fraction, 4)}")
    st.write(f"**Recommended Bet Size at {fraction_kelly}x Kelly:** ${round(recommended_bet, 2)}")

    # Monte Carlo simulation
    kelly_fractions = [0.1, 0.25, 0.33, 0.5, 0.67, 1.0]
    sim_results = run_monte_carlo(prob_input, win_loss_ratio, bankroll, kelly_fractions, num_trials)

    fig = go.Figure()
    for kf, values in sim_results.items():
        fig.add_trace(go.Box(y=values, name=f"{kf}x Kelly"))

    fig.update_layout(title="Distribution of Ending PnL for Kelly Fractions (1000 simulations each)",
                      yaxis_title="Total PnL ($)", height=600, width=800)
    st.plotly_chart(fig)
