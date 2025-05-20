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

def calculate_kelly(fair_prob, offered_payout):
    return (fair_prob * (offered_payout + 1) - 1) / offered_payout

def run_monte_carlo_paths(fair_prob, offered_payout, bankroll, kelly_fractions, num_trials):
    results = {}
    for kf in kelly_fractions:
        all_paths = []
        for _ in range(100):
            capital = bankroll
            path = [capital - bankroll]
            for _ in range(num_trials):
                bet_size = kf * capital  # dynamic stake with bankroll growth/shrinkage
                outcome = np.random.rand() < fair_prob
                capital += bet_size * offered_payout if outcome else -bet_size
                path.append(capital - bankroll)
            all_paths.append(path)
        results[kf] = all_paths
    return results

# === Streamlit app ===
st.markdown("# Kelly Betting Calculator")

st.markdown("### Your Fair Line")
fair_input_method = st.radio("Enter your fair value as:", ["American Odds", "Implied Probability"], key="fair")
if fair_input_method == "American Odds":
    fair_american_odds = st.number_input("Your Fair American Odds", value=+110, key="fair_odds")
    fair_prob, _ = american_to_prob_and_payout(fair_american_odds)
else:
    fair_prob = st.number_input("Your Fair Implied Win Probability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)

st.markdown("### Book's Offered Line")
offered_input_method = st.radio("Enter the offered value as:", ["American Odds", "Implied Probability"], key="offered")
if offered_input_method == "American Odds":
    offered_american_odds = st.number_input("Book's Offered American Odds", value=+110, key="offered_odds")
    _, offered_payout = american_to_prob_and_payout(offered_american_odds)
else:
    offered_prob = st.number_input("Book's Implied Win Probability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.47, step=0.01)
    offered_payout = (1 / offered_prob) - 1

bankroll = st.number_input("Bankroll ($)", min_value=1, max_value=10000000, value=10000, step=100)
fraction_kelly = st.number_input("Fraction of Kelly to Use", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
num_trials = st.number_input("Number of Monte Carlo Trials", min_value=1, max_value=10000, value=100, step=10)

if st.button("Calculate Bet and Run Simulation"):
    kelly_fraction = calculate_kelly(fair_prob, offered_payout)
    recommended_bet = bankroll * kelly_fraction * fraction_kelly

    st.write(f"**Fair Win Probability:** {round(fair_prob, 4)}")
    st.write(f"**Offered Payout Multiplier:** {round(offered_payout, 4)}")
    st.write(f"**Full Kelly Fraction:** {round(kelly_fraction, 4)}")
    st.write(f"**Recommended Bet Size at {fraction_kelly}x Kelly:** ${round(recommended_bet, 2)}")

    kelly_fractions = [0.1, 0.25, 0.33, 0.5, 0.67, 1.0]
    sim_paths = run_monte_carlo_paths(fair_prob, offered_payout, bankroll, kelly_fractions, num_trials)

    fig = go.Figure()
    for kf, paths in sim_paths.items():
        for path in paths:
            fig.add_trace(go.Scatter(y=path, mode='lines', name=f"{kf}x Kelly", opacity=0.2, showlegend=False))
        avg_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(y=avg_path, mode='lines', name=f"{kf}x Kelly Avg", line=dict(width=3)))

    fig.update_layout(title="Running PnL Over Time for Kelly Fractions",
                      xaxis_title="Trial",
                      yaxis_title="Cumulative PnL ($)",
                      height=600, width=800)
    st.plotly_chart(fig)
