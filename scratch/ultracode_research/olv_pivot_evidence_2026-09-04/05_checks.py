"""Assemble checks.json (the brief's keys) from the script outputs."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ev = json.loads((HERE / "evidence_search.json").read_text())
    rp = json.loads((HERE / "replay_summary.json").read_text())
    bs = json.loads((HERE / "basis_summary.json").read_text())
    ls = json.loads((HERE / "live_status.json").read_text())
    live_latest = ls["2026-09-03_scan_PM_and_2026-09-04_scan_AM"]
    live_prev = ls["2026-09-03_scan_AM"]

    def era(arm: str, e: str) -> dict:
        s = rp["eras"][e][arm]
        return {k: s[k] for k in ("signals_total", "signals_staged", "fills_completed", "avgR", "pnl_per_unit_risk",
                                  "total_flat_pnl", "worst_21d_flat_pnl", "max_dd_flat_pnl")}

    checks = {
        "evidence_found": True,
        "evidence_location": (
            "artifacts/olv-pivot-age-252-20260901/summary.json policy_counterfactual (gitignored, untracked): "
            f"age-capped policy total_r {ev['claim_reconstruction']['policy_v2_age_capped_252']['total_r']:.3f} minus "
            f"unlimited-age policy total_r {ev['claim_reconstruction']['policy_v1_unlimited_age']['total_r']:.3f} = "
            f"+{ev['claim_reconstruction']['difference_total_r_v2_minus_v1']:.2f}R on the 359 completed OLV fills of the "
            "2026-08-31 ledger (artifacts/olv-level-proximity-20260831); it is policy-v2 vs policy-v1, NOT policy vs "
            f"no-policy (no-policy baseline on the same sample {ev['claim_reconstruction']['no_policy_baseline_total_r_same_sample']:.3f}R)"
        ),
        "with_policy": {e: era("with_policy", e) for e in ("2010", "2016H2", "2024")},
        "without_policy": {e: era("without_policy", e) for e in ("2010", "2016H2", "2024")},
        "affected_signals_n": rp["affected"]["2010"]["n"],
        "affected_diff_R": rp["affected"]["2010"]["sum_diff_R"],
        "affected_t_clustered_2010": rp["affected"]["2010"]["t_clustered_signal_date"],
        "affected_t_clustered_2016": rp["affected"]["2016H2"]["t_clustered_signal_date"],
        "skipped_n": rp["skipped"]["2010"]["n_signals"],
        "skipped_wouldbe_R": rp["skipped"]["2010"]["wouldbe_R_without"],
        "basis_flip_share": bs["primary_cache_win_vs_yf_full_context"]["flip_share_of_affected"],
        "basis_flips_n": bs["primary_cache_win_vs_yf_full_context"]["n_flips_among_affected"],
        "live_nondefault_n": live_latest["n_nondefault_cache"],
        "live_flips_n": live_latest["n_flips_among_nondefault"],
        "_notes": {
            "affected_definition": "candidates whose proposed matched_rule != default (2-3, 4-5, >5 bands), signal date 2010-01-01..2026-08-31, censored-on-either-arm excluded",
            "affected_diff_R": "sum over affected signals of R_with (0 if skipped/unfilled) - R_without (0 if unfilled); flat $750k basis",
            "basis_flip_share": "share of liquid-tier signals since 2023-09-04 that are non-default on the cache (same yf window) whose band differs on the yf auto_adjust series, restricted to signals with >= 300 yf bars of context",
            "basis_flip_share_all_windows": bs["cache_win_vs_yf_all"]["flip_share_of_affected"],
            "basis_flip_share_production_vs_yf": bs["cache_full_vs_yf_all"]["flip_share_of_affected"],
            "live_prev_session_2026-09-02_close": {"nondefault_n": live_prev["n_nondefault_cache"], "flips_n": live_prev["n_flips_among_nondefault"]},
            "live_latest_close": live_latest["close_date"],
            "policy_off_value": rp["policy_off_value"],
        },
    }
    (HERE / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
