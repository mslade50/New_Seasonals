"""Merge the robust-Bayesian sizing study outputs into one deliverable
(robust_bayes_results.json) with the plan's decision table, so every number in
the plan traces to a JSON beside this file."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parts = {k: json.load(open(HERE / f)) for k, f in [
    ("grm", "robust_bayes_01_grm.json"), ("margin_sens", "robust_bayes_01b_margin_sens.json"),
    ("rule_shrinkage", "robust_bayes_02_rule_shrinkage.json"), ("allocation", "robust_bayes_03_allocation.json")]}

decisions = [
    dict(id="GM-1", level="L4", change="GRM 1.5 -> 1.875 (m 1.25) after GM-2 and the IBKR what-if check; 2.25 only after L5 haircut measured on >= 150 fills",
         posterior_basis="growth monotone in m under every keep in the prior (even 0.30); margin wall m 1.63-1.80 TIMS / 1.01 rules-3x; live NLV already 1.19x"),
    dict(id="GM-2", level="L4", change="margin-feasibility guard in order_staging (projected requirement on live NLV; trim NEW entries pro rata above 70%); broker constraint, not an ATR cap"),
    dict(id="GM-3", level="L4", change="sizing base stays flat $750k until NLV > 750k; then half-compounding quarterly, never rebased down"),
    dict(id="HC-1", level="L5", change="haircut on the ledger mean 40% central (keep 0.60), range 27-71%; per-strategy keep_s used as mu shrinkage; pilots capped at keep 0.5"),
    dict(id="AL-1", level="L1", change="keep-adjusted half-tilt clip [0.7, 1.3], annual refit; fit through 2025: 52wh 0.70, WCDS 0.75, SectorBO 0.87, StOS 0.88, IOB 0.89, OVS 1.0 (held), LTT 1.04, MondayDip 1.09, ATRExt 1.10, OLV 1.17, 3xETF 1.27, MonFri 1.30"),
    dict(id="AD-1", level="L2", change="OLV ladder re-keyed to max(ticker-recency rung, sleeve-depth rung), depth incl. working entries"),
    dict(id="AD-2", level="L2", change="WCDS and LT Trend ST OS solo 0.8x / adds 1.2x (shrunk from the 0.75/1.25 study form)"),
    dict(id="AD-3", level="L2", change="OVS bottom-extremity 0.7x when mean rank_2/5/10/21d < 94 (posterior 0.68; not 0.5)"),
    dict(id="AD-4", level="L2", change="IOB SPY+QQQ same-day 0.5x each; dip-buy cross-strategy same-ticker clamp extended (variance-only, from baseline)"),
    dict(id="AD-5", level="L2", change="NO 52wh depth rule (posterior 1.00); L1 cut + P4 exit prereg carry the 52wh tail"),
    dict(id="FL-1", level="L2", change="flow-aware per-strategy cap 250 -> 375 bps on the family's top-tercile 5d raw-candidate days (dip_buy >= 6, oversold_hold >= 7, short_fade >= 104); gated by GM-2"),
    dict(id="FL-2", level="L2", change="up-only family flow multiplier 1.2x (posterior 1.17-1.33) on the same days; dip_buy gated dial < 50; no breakout carrier; post-ship review written"),
    dict(id="FL-3", level="L2", change="OVS P2 aggregate cap 0.75% -> 1.0% nominal (half-step; posterior 1.11 on t 1.46)"),
    dict(id="XS-1", level="L3", change="dial-armed whole-book beta hedge: paper-track at 50/45 with 126d (or shrunk) beta, 1.0x, ES/MES; ship after one logged episode; posterior +$3.5k/episode"),
    dict(id="XS-2", level="L3", change="LT Trend ST OS band [[50,999,0.75]] as a PREREG candidate (posterior 0.72 at PIT 50; threshold moved post hoc so it needs a fresh registration)"),
    dict(id="XS-3", level="L3", change="FAMILY4 / P/C bands unchanged (appetite); robust posterior 0.80-0.90 says the post-ship review's fallback should be 0.5x not 0.25x/zero; re-score on hedged returns after XS-1"),
    dict(id="RG-1", level="L3", change="OLV 1.15x when SPY is 3-10% off its 252d high at the signal close (posterior 1.17; half the study's 1.25-1.5); prereg-then-ship"),
    dict(id="S-1", level="S", change="no calendar multiplier anywhere; watch items only (dip-buy Sep, family summer gap)"),
    dict(id="M-1", level="L5", change="live-vs-ledger reconciliation from broker fills; commissions + MKT-exit spread in the engine; margin headroom monitor (gross/NAV, blended rate, -30% scenario); IBKR what-if on 2023-02-03 / 2016-06-14 / 2019-06-26 books"),
]
json.dump(dict(asof="2026-09-02", lens="robust-bayesian", decisions=decisions, **parts), open(HERE / "robust_bayes_results.json", "w"), indent=1, default=str)
print("wrote robust_bayes_results.json with", len(decisions), "decisions")
