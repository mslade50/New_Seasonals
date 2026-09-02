"""Judge 2 (growth/drawdown arithmetic): inspect the structure of the analysts' result files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    "unconstrained_growth_01_growth.json",
    "unconstrained_growth_01b_capaware.json",
    "unconstrained_growth_02_margin.json",
    "unconstrained_growth_02b_margin_refine.json",
    "unconstrained_growth_04_compounding.json",
    "growthmax_1_margin_tiered.json",
    "growthmax_2_growth_dd_acf.json",
    "growthmax_3_alloc_keep.json",
    "robust_bayes_01_grm.json",
    "robust_bayes_01b_margin_sens.json",
    "robust_bayes_03_allocation.json",
    "practitioner_01_acf_drawdown.json",
    "practitioner_02_package_replay.json",
    "risk_arch_grm_frontier.json",
    "risk_arch_stack_stress.json",
    "risk_arch_theme_budget.json",
    "estimation_haircut_results.json",
]


def describe(obj, depth: int = 0, max_depth: int = 2, prefix: str = "") -> None:
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{pad}{k}: dict[{len(v)}]")
                if depth < max_depth:
                    describe(v, depth + 1, max_depth)
            elif isinstance(v, list):
                inner = type(v[0]).__name__ if v else "empty"
                print(f"{pad}{k}: list[{len(v)}] of {inner}")
            else:
                s = str(v)
                if len(s) > 80:
                    s = s[:77] + "..."
                print(f"{pad}{k}: {s}")


for name in FILES:
    p = ROOT / name
    if not p.exists():
        print(f"=== {name}: MISSING")
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"=== {name} ({p.stat().st_size // 1024} KB)")
    describe(data, max_depth=1)
    print()
