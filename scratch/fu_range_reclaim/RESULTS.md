# Range-reclaim revision 2 — 2026-09-22

The detector now uses a defined horizontal range above a rising SMA200, with support established before the sweep. It is a new range-based interpretation; it does not preserve the original detector's requirement for a preceding breakout and single first-pullback anchor.

## Visual review

Latest 15 signals were selected by date and inspected with subsequent prices hidden. The most useful shape examples are PG 2022-02-14, MPWR 2023-04-20, CMRE 2021-04-21 and DPZ 2020-01-24. These choices reflect range clarity, not return ranking.

The initial 31-signal pass exposed an issue in POST 2017-05-02: a positive 20-session change in the SMA can coexist with a recent downward turn. A regression test reproduced that failure. The final version also requires the SMA to rise on the signal day. No range or execution parameter was tuned to returns. Both initial and final artifacts are retained.

Above a rising SMA200 is a specific long-term trend definition. It can still admit consolidations after a recent decline, such as GLNG and PCG. If those shapes are unwanted, that is an additional trend/structure requirement, not something this filter already guarantees.

## Final run

- Location: `artifacts/fu_range_reclaim/run_v2b/`.
- Input: immutable research snapshot from `run_v2`, originally read from the local price cache on September 22; 931 usable equities/ETFs through 2026-09-22.
- The original quality gate excluded 67 invalid rows across 64 whole tickers. The final run consumes the already-filtered snapshot; `review_audit.json` preserves that lineage.
- 28 signals from 2000 onward: 5 liquid and 23 overflow. Only four signals occur in 2023 or later.
- Ten-session mean return after 10 bps costs: liquid -0.129%, overflow -0.485%; same-date SPY excess -1.352% and -0.834%, respectively. These tiny samples cannot support a reliable performance conclusion. Full fixed 5/10/20-session and stop-only results are in `summary.csv` and `results.html`.
- Current-constituent selection and unresolved corporate-action issues remain. No qualifying setup/trade window crossed a flagged absolute 40%+ overnight gap; this check is not a comprehensive corporate-action reconciliation.

## Verification and scope

14 tests passed, including causal pivots, prefix/future mutation invariance, repeated support, frozen boundaries, trend rejection, rescaling and execution. The newly added SMA-turn regression failed before the fix and passed afterward. All 28 actual signals passed independent trend/confirmation assertions. The 111 exported trade records passed timing, overlap, cost, SPY-date and input/source-hash checks. All 15 final gallery setups were visually inspected; the final new SLGN example was checked separately.

Source changes are confined to this research directory and `tests/test_fu_range_reclaim.py`. No new branch/worktree, production integration, cache edit, upload, push or order was made. The workspace check flagged an unrelated concurrent change to `fundamental/financing_opportunity_report.py`; it was not touched or reset by this task.

The next decision is visual fidelity. These deliberately strict rules may be too selective; their low signal count is not a reason to loosen them based on the observed returns.
