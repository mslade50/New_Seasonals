"""Robust-Bayesian shrinkage of every conditional sizing rule on the table
(2026-09-02). Inputs are the cell contrasts reported by the six repo analyses
in the evidence pack (no ledger re-run); the output is the POSTERIOR multiplier
each rule earns once (i) the search that found it is priced as a prior
probability of being noise and (ii) the effect is shrunk toward zero by its
standard error against the scale of real effects in this book.

Two-groups model per rule:
  P(real) = pi1 ; real effect delta ~ N(0, tau^2) ; observed d ~ N(delta, se^2)
  BF = N(d; 0, tau^2 + se^2) / N(d; 0, se^2)
  post = pi1 BF / (pi1 BF + 1 - pi1)
  E[delta | d] = post * d * tau^2 / (tau^2 + se^2)
Multiplier (Kelly on the conditional cell, half-Kelly on the tilt because sdR is
flat across states in this book): mult = 1 + 0.5 * E[delta|d] / mu_base,
clipped to [0.5, 1.5].

pi1 is given two ways and both are reported: (a) PROVENANCE prior from how the
cell was found (pre-registered single hypothesis 0.5; small targeted scan with an
outside mechanism 0.3; medium scan 0.1; large scan 0.03), (b) EMPIRICAL-BAYES
prior from the analysis' own excess of hits over the null expectation
(signal_quality: 14 full-rule passes vs ~1.5 expected -> 0.89; 46 monotone
|t|>=2 hits vs 27 expected -> 0.41). The plan uses the geometric mean.

tau is 0.25R centrally (the spread of surviving cell effects in this book),
sensitivities at 0.15 and 0.40. Writes robust_bayes_02_rule_shrinkage.json.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

def norm_pdf(x, s):
    return math.exp(-0.5 * (x / s) ** 2) / (s * math.sqrt(2 * math.pi))

def shrink(d, se, pi1, tau, mu_base, half=0.5, lo=0.5, hi=1.5):
    bf = norm_pdf(d, math.sqrt(tau ** 2 + se ** 2)) / norm_pdf(d, se)
    post = pi1 * bf / (pi1 * bf + (1 - pi1))
    e_delta = post * d * tau ** 2 / (tau ** 2 + se ** 2)
    mult = 1 + half * e_delta / mu_base
    return post, e_delta, min(hi, max(lo, mult))

# id, description, d (cell minus complement, R), se, pi1_prov, pi1_eb, mu_base, live_or_proposed_mult, source
RULES = [
    ("SQ-OVS-EXT", "OVS mean short-window rank < 94 (bottom cell) vs rest", -0.30, 0.09, 0.10, 0.89, 0.39, 0.50, "signal_quality f1: 0.17 vs 0.47, t_cluster 5.4 on z-feature, LOYO 80%, all eras/tiers/paths"),
    ("FLOW-DIP", "dip_buy 5d family candidate count top tercile vs rest", 0.20, 0.074, 0.30, 0.41, 0.45, 1.25, "flow_conditional f1: terciles 0.216/0.657/0.569, t_ep 2.70, per-year 82%"),
    ("FLOW-OSH", "oversold_hold 5d family count top tercile vs rest", 0.35, 0.117, 0.30, 0.41, 0.57, 1.25, "flow_conditional f1: 0.317/0.571/0.855, t_ep 3.00"),
    ("FLOW-SF", "short_fade 5d family count top tercile vs rest", 0.30, 0.094, 0.30, 0.41, 0.38, 1.25, "flow_conditional f1: 0.298/0.367/0.678, t_ep 3.21"),
    ("OVS-SAMEDAY5", "OVS same-day own count >= 5 vs < 5", 0.42, 0.087, 0.30, 0.41, 0.39, 1.25, "seasonality_flow f5: 0.648 vs 0.230, t_ep 4.84, q 0.002, LOYO floor +0.34"),
    ("OLV-SPYDD", "OLV signal with SPY 3-10% off 252d high vs rest", 0.63, 0.21, 0.10, 0.41, 0.76, 1.25, "cycle_macro f6: 1.184 vs 0.554, t 2.98, 8/1 years, WF 11/15"),
    ("LTT-PIT50", "LT Trend ST OS at PIT dial >= 50 vs < 50", -0.375, 0.16, 0.30, 0.41, 0.32, 0.50, "cycle_macro f5: 0.107 vs 0.482, t_cluster -2.33, 9 episodes, LOYO all negative"),
    ("WCDS-ADDS", "WCDS leg entered with >= 1 WCDS open vs solo", 0.32, 0.14, 0.30, 0.41, 0.34, 1.25, "within_strategy_adds f1: +0.32R, boot P 0.011, 18/22 years"),
    ("LTT-ADDS", "LT Trend leg at depth >= 3 vs solo", 0.40, 0.17, 0.30, 0.41, 0.32, 1.25, "within_strategy_adds f1: +0.40R, P 0.009, t 2.65, LOYO 2.0"),
    ("OLV-DEPTH", "OLV add (>= 1 open) vs solo", 0.39, 0.19, 0.30, 0.41, 0.76, 1.00, "within_strategy_adds f1: +0.39R, P 0.017 (ladder currently cuts solo to 0.5x)"),
    ("OLV-SOLO", "OLV solo leg vs adds (the ladder's cut cell)", -0.39, 0.19, 0.30, 0.41, 0.76, 0.50, "same contrast, signed from the solo side"),
    ("52WH-DEEP6", "52wh leg at >= 6 open vs rest", -0.10, 0.45, 0.10, 0.41, 0.53, 0.50, "within_strategy_adds f7: -0.10R, boot P 0.57, era flip"),
    ("FAM-NEARHIGH", "single-stock dip-buys with SPY within 2% of high, ALPHA part (residual after SPY beta)", -0.15, 0.107, 0.10, 0.50, 0.45, 0.50, "signal_quality f4: raw -0.30 t -2.87; residual t -1.40 non-monotone"),
    ("52WH-RV10", "52wh at SPY rv21 < 10% vs rest", -0.61, 0.27, 0.05, 0.41, 0.53, 0.50, "signal_quality f7: 0.19 vs 0.80, t 2.0-2.4, binary LOYO 64%"),
    ("DIP-SUMMER", "dip-buy family May-Oct vs Nov-Apr", -0.19, 0.09, 0.10, 0.10, 0.40, 0.75, "seasonality_flow f4: t_ep -2.12, post-2013 only, 0/1012 cells survive FDR"),
    ("OVS-MIDTERM", "OVS path 1 in midterm years vs other (live 0.75x)", -0.21, 0.105, 0.30, 0.41, 0.39, 0.75, "cycle_macro f3: 0.156 vs 0.366, year-clustered t -2.0, 6/6 years"),
    ("FAM4-DIAL50", "FAMILY4 dip-buys at PIT dial >= 50 vs < 50 (live 0.25x / 0 by P/C state)", -0.73, 0.37, 0.30, 0.41, 0.50, 0.25, "CLAUDE.md PIT gate 2026-07-03: hi -0.10 vs lo +0.63, clustered t -1.96, 6/9 years negative"),
    ("OVS-CAPBOUND", "OVS trades on cap-bound days vs unbound (cap relief candidate)", 0.42, 0.12, 0.30, 0.41, 0.39, 1.50, "flow_conditional f6: 0.62 vs 0.20 on 101 bound days"),
    ("OVS-P2CAP", "OVS P2 trades that were capped vs uncapped (P2 aggregate cap)", 0.19, 0.13, 0.20, 0.41, 0.20, 1.50, "signal_quality f3: 0.37 vs 0.18, 202 capped of 609, 11/14 years"),
    ("EARN-SEASON", "single-stock strategies in earnings season vs off", 0.11, 0.07, 0.10, 0.10, 0.45, 1.25, "seasonality_flow f6: 0.52 vs 0.41, t_yr 1.49, LOYO [+0.06,+0.15], q 0.27"),
]
rows = []
for tau in [0.15, 0.25, 0.40]:
    for rid, desc, d, se, p_prov, p_eb, mu, live, src in RULES:
        p_geo = math.sqrt(p_prov * p_eb)
        post_p, e_p, m_p = shrink(d, se, p_prov, tau, mu)
        post_e, e_e, m_e = shrink(d, se, p_eb, tau, mu)
        post_g, e_g, m_g = shrink(d, se, p_geo, tau, mu)
        rows.append(dict(id=rid, tau=tau, d=d, se=se, t=round(d / se, 2), pi1_prov=p_prov, pi1_eb=p_eb, pi1_geo=round(p_geo, 3),
                         post_prov=round(post_p, 3), post_eb=round(post_e, 3), post_geo=round(post_g, 3),
                         E_delta_geo=round(e_g, 3), mult_prov=round(m_p, 3), mult_eb=round(m_e, 3), mult_geo=round(m_g, 3),
                         mult_live_or_proposed=live, desc=desc, source=src))
df = pd.DataFrame(rows)
pd.set_option("display.width", 260, "display.max_columns", 30)
print("=== posterior multipliers at tau = 0.25 (central) ===")
print(df[df.tau == 0.25][["id", "d", "se", "t", "pi1_prov", "pi1_eb", "post_geo", "E_delta_geo", "mult_prov", "mult_eb", "mult_geo", "mult_live_or_proposed"]].to_string(index=False))
print("\n=== sensitivity: mult_geo by tau ===")
print(df.pivot(index="id", columns="tau", values="mult_geo").round(3).to_string())

# hedge: episode-level version. PIT 50/45: 13 episodes, mean +$8.0k, clustered t 1.68 -> se 4.76k; tau = $8k
def hedge_post(mean, se, pi1, tau):
    bf = norm_pdf(mean, math.sqrt(tau ** 2 + se ** 2)) / norm_pdf(mean, se)
    post = pi1 * bf / (pi1 * bf + 1 - pi1)
    return post, post * mean * tau ** 2 / (tau ** 2 + se ** 2)
H = {}
for label, mean, se, pi1 in [("PIT_50_45", 8.0, 4.76, 0.5), ("PIT_65_60", 6.9, 6.97, 0.5), ("LIVE_50_45", 3.5, 4.9, 0.5),
                             ("PIT_50_45_vixgate_rel30", 10.75, 5.3, 0.15)]:
    post, e = hedge_post(mean, se, pi1, 8.0)
    H[label] = dict(mean_episode_k=mean, se_k=se, pi1=pi1, post_real=round(post, 3), E_episode_k=round(e, 2),
                    E_per_year_k=round(e * 1.3, 2))
print("\n=== hedge episode posterior (tau $8k/episode, ~1.3 episodes/yr) ===")
print(pd.DataFrame(H).T.to_string())

# noise base rate: how many of the pack's claimed cells would survive a Bonferroni-style prior at their own family size
summary = dict(central=df[df.tau == 0.25].set_index("id")[["mult_geo", "post_geo", "mult_live_or_proposed"]].to_dict("index"), hedge=H)
json.dump(dict(rules=df.to_dict("records"), summary=summary), open(HERE / "robust_bayes_02_rule_shrinkage.json", "w"), indent=1)
print("\nwrote robust_bayes_02_rule_shrinkage.json")
