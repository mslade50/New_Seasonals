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

def run_single_kelly_monte_carlo(fair_prob, offered_payout, bankroll, kelly_fraction, num_trials, num_paths=10):
    paths = []
    for _ in range(num_paths):
        capital = bankroll
        path = [capital - bankroll]
        for _ in range(num_trials):
            bet_size = kelly_fraction * capital
            outcome = np.random.rand() < fair_prob
            capital += bet_size * offered_payout if outcome else -bet_size
            path.append(capital - bankroll)
        paths.append(path)
    return paths

# === Streamlit app ===
st.markdown("# Kelly Betting Calculator")

st.markdown("### Your Fair Line")
fair_input_method = st.radio("Enter your fair value as:", ["American Odds", "Implied Probability"], key="fair")
if fair_input_method == "American Odds":
    fair_american_odds = st.number_input("Your Fair American Odds", value=+110, key="fair_odds")
    fair_prob, _ = american_to_prob_and_payout(fair_american_odds)
    st.write(f"**Implied Probability:** {round(fair_prob * 100, 2)}%")
else:
    fair_prob = st.number_input("Your Fair Implied Win Probability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    fair_american_odds = (100 * (1 - fair_prob) / fair_prob) if fair_prob >= 0.5 else (-100 * fair_prob / (1 - fair_prob))
    st.write(f"**Equivalent American Odds:** {round(fair_american_odds, 1)}")

st.markdown("### Book's Offered Line")
offered_input_method = st.radio("Enter the offered value as:", ["American Odds", "Implied Probability"], key="offered")
if offered_input_method == "American Odds":
    offered_american_odds = st.number_input("Book's Offered American Odds", value=+110, key="offered_odds")
    offered_prob, offered_payout = american_to_prob_and_payout(offered_american_odds)
    st.write(f"**Implied Probability:** {round(offered_prob * 100, 2)}%")
else:
    offered_prob = st.number_input("Book's Implied Win Probability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.47, step=0.01)
    offered_payout = (1 / offered_prob) - 1
    offered_american_odds = (100 * (1 - offered_prob) / offered_prob) if offered_prob >= 0.5 else (-100 * offered_prob / (1 - offered_prob))
    st.write(f"**Equivalent American Odds:** {round(offered_american_odds, 1)}")

bankroll = st.number_input("Bankroll ($)", min_value=1, max_value=10000000, value=10000, step=100)
fraction_kelly = st.number_input("Fraction of Kelly to Use", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
num_trials = st.number_input("Number of Monte Carlo Trials", min_value=1, max_value=10000, value=100, step=10)

if st.button("Calculate Bet and Run Simulation"):
    kelly_fraction = calculate_kelly(fair_prob, offered_payout)
    recommended_bet = bankroll * kelly_fraction * fraction_kelly

    expected_value = (fair_prob * (offered_payout * recommended_bet)) - ((1 - fair_prob) * recommended_bet)
    edge = fair_prob * (offered_payout + 1) - 1

    st.write(f"**Fair Win Probability:** {round(fair_prob, 4)}")
    st.write(f"**Offered Payout Multiplier:** {round(offered_payout, 4)}")
    st.write(f"**Full Kelly Fraction:** {round(kelly_fraction, 4)}")
    st.write(f"**Recommended Bet Size at {fraction_kelly}x Kelly:** ${round(recommended_bet, 2)}")
    st.write(f"**Expected Value of the Bet:** ${round(expected_value, 2)}")
    st.write(f"**Edge of the Bet:** {round(edge * 100, 2)}%")

    sim_paths = run_single_kelly_monte_carlo(fair_prob, offered_payout, bankroll, kelly_fraction * fraction_kelly, num_trials)
    avg_path = np.mean(sim_paths, axis=0)

    fig = go.Figure()
    for path in sim_paths:
        fig.add_trace(go.Scatter(y=path, mode='lines', name="Path", opacity=0.3))
    fig.add_trace(go.Scatter(y=avg_path, mode='lines', name="Average", line=dict(width=3)))

    fig.update_layout(title="Running PnL Over Time for Selected Kelly Fraction",
                      xaxis_title="Trial",
                      yaxis_title="Cumulative PnL ($)",
                      height=600, width=800)
    st.plotly_chart(fig)
