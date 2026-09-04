"""Locate the origin of the '359 completed-fill sample, +8.68R' claim.

Writes evidence_search.json. Every number here is quoted from a file whose
path is recorded next to it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "evidence_search.json"


def git(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return res.stdout.strip()


def main() -> None:
    out: dict = {}
    out["git_head"] = git("rev-parse", "HEAD")
    out["git_log_S"] = {
        "8.68 (tracked py/md only)": git("log", "--all", "--oneline", "-S", "8.68", "--", "*.py", "*.md"),
        "ClosePivot": git("log", "--all", "--oneline", "-S", "ClosePivot"),
        "pivot_entry_policy": git("log", "--all", "--oneline", "-S", "pivot_entry_policy"),
    }
    out["policy_commits"] = {
        c: git("show", "--no-patch", "--format=%H %an %ci %s", c)
        for c in ("dec62f06", "a3036527", "1119efae")
    }
    out["artifacts_gitignored"] = git("check-ignore", "-v", "artifacts/olv-pivot-age-252-20260901/summary.json")

    # The comment text that carries the claim, with the commit that introduced it.
    blame = git("log", "--all", "--oneline", "-S", "359 completed-fill", "--", "strategy_config.py")
    out["claim_comment_introduced_by"] = blame

    # Untracked research folders under artifacts/ that hold the sample.
    dirs = {
        "level_proximity": ROOT / "artifacts" / "olv-level-proximity-20260831",
        "pivot_high_threshold": ROOT / "artifacts" / "olv-pivot-high-threshold-20260831",
        "pivot_20_vs_40": ROOT / "artifacts" / "olv-pivot-20-vs-40-20260831",
        "pivot_entry_path": ROOT / "artifacts" / "olv-pivot-entry-path-20260831",
        "pivot_age_252": ROOT / "artifacts" / "olv-pivot-age-252-20260901",
        "below_second_pivot": ROOT / "artifacts" / "olv-below-second-pivot-20260904",
    }
    out["artifact_dirs"] = {k: {"path": str(v), "exists": v.exists(),
                                "files": sorted(p.name for p in v.iterdir()) if v.exists() else []}
                            for k, v in dirs.items()}

    lp = json.loads((dirs["level_proximity"] / "summary.json").read_text())
    out["sample_definition"] = {
        "source": str(dirs["level_proximity"] / "summary.json"),
        "completed_trades": lp["completed_trades"],
        "excluded_censored_trades": lp["excluded_censored_trades"],
        "signal_date_min": lp["signal_date_min"],
        "signal_date_max": lp["signal_date_max"],
        "price_cache_max_date": lp["price_cache_max_date"],
        "ledger_vintage": {k: v for k, v in lp["ledger_metadata"].items() if k != "pandas"},
        "baseline_no_policy": lp["baseline"],
        "censor_rule": "analyze_olv_levels.py load_completed_olv: drop rows with Exit Type == Time and Exit Date < Time Stop",
    }

    age = json.loads((dirs["pivot_age_252"] / "summary.json").read_text())
    pc = {row["definition"]: row for row in age["policy_counterfactual"]}
    unlimited = pc["Unlimited age"]
    hard = pc["Hard source-age <=252"]
    out["claim_reconstruction"] = {
        "source": str(dirs["pivot_age_252"] / "summary.json"),
        "policy_v1_unlimited_age": unlimited,
        "policy_v2_age_capped_252": hard,
        "difference_total_r_v2_minus_v1": hard["total_r"] - unlimited["total_r"],
        "classification_changes": age["hard_252"]["type_switch_n"],
        "policy_decision_changes": age["hard_252"]["policy_decision_change_n"],
        "no_policy_baseline_total_r_same_sample": lp["baseline"]["total_r"],
        "no_policy_baseline_avg_r_same_sample": lp["baseline"]["avg_r"],
        "no_policy_baseline_max_dd_r": lp["baseline"]["max_drawdown_r"],
        "policy_v2_minus_no_policy_total_r": hard["total_r"] - lp["baseline"]["total_r"],
        "method_note": (
            "Post-hoc rebook on the 2026-08-31 ledger's 359 completed OLV fills: deeper limits "
            "re-filled from daily bars over T+1..T+3, exits re-simulated with the current OLV exit "
            "model (artifacts/olv-pivot-entry-path-20260831/summary.json method block); NOT an "
            "engine replay, no per-strategy cap, no recency ladder, no notional cap."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["claim_reconstruction"], indent=2))


if __name__ == "__main__":
    main()
