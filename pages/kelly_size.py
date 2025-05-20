import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# === Helper functions ===
def american_to_prob_and_payout(odds):
    if odds > 0:
        prob = 100 / (odds + 100)
        payout = odds / 100
    else:
        prob = -odds / (-odds + 100)
        payout = 100 / abs(odds)
    return prob, payout

def calculate_kelly(prob, payout):
    return (prob * (payout + 1) - 1) / payout

def run_monte_carlo(prob, payout, bankroll, kelly_fractions, num_trials):
    results = {}
    for kf in kelly_fractions:
        final_pnls = []
        for _ in range(1000):
            capital = bankroll
            for _ in range(num_trials):
                bet_size = kf * bankroll
                outcome = np.random.rand() < prob
                capital += bet_size * payout if outcome else -bet_size
            final_pnls.append(capital - bankroll)
        results[kf] = final_pnls
    return results

# === Streamlit app ===
st.markdown("# Kelly Betting Calculator")
input_method = st.radio("Choose Input Method", ["American Odds", "Implied Probability + Payout"])

if input_method == "American Odds":
    american_odds = st.number_input("Offered American Odds", value=+110)
    prob_input, payout = american_to_prob_and_payout(american_odds)
    st.write(f"Implied Probability: {round(prob_input, 4)} | Payout Multiplier: {round(payout, 4)}")
else:
    prob_input = st.number_input("Implied Win Probability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    payout = st.number_input("Payout Multiplier (e.g. 1.1 for +110)", min_value=0.1, max_value=100.0, value=1.1, step=0.01)

bankroll = st.number_input("Bankroll ($)", min_value=1, max_value=10000000, value=10000, step=100)
fraction_kelly = st.number_input("Fraction of Kelly to Use", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
num_trials = st.number_input("Number of Monte Carlo Trials", min_value=1, max_value=10000, value=100, step=10)

if st.button("Calculate Bet and Run Simulation"):
    kelly_fraction = calculate_kelly(prob_input, payout)
    recommended_bet = bankroll * kelly_fraction * fraction_kelly

    st.write(f"**Full Kelly Fraction:** {round(kelly_fraction, 4)}")
    st.write(f"**Recommended Bet Size at {fraction_kelly}x Kelly:** ${round(recommended_bet, 2)}")

    kelly_fractions = [0.1, 0.25, 0.33, 0.5, 0.67, 1.0]
    sim_results = run_monte_carlo(prob_input, payout, bankroll, kelly_fractions, num_trials)

    fig = go.Figure()
    for kf, values in sim_results.items():
        fig.add_trace(go.Box(y=values, name=f"{kf}x Kelly"))

    fig.update_layout(title="Distribution of Ending PnL for Kelly Fractions (1000 simulations each)",
                      yaxis_title="Total PnL ($)", height=600, width=800)
    st.plotly_chart(fig)
