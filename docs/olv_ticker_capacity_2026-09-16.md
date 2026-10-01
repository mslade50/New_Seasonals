# Whole-ticker sizing capacity

Candidate correction to the installed September 15 capacity repair.

The 50%-NAV single-stock cap sizes new OLV entries using all broker-held stock
value in the ticker, remaining OLV buy-order reservations, and actual Primary
NAV. The new quantity is limited to the whole shares that fit in the remaining
capacity at the proposed limit price. Existing ETF exemptions and the explicit
unavailable-capacity bypass policy are unchanged.

Sizing never prefers strategy-attributed inventory, even when that inventory
is verified: it can omit other holdings in the same ticker. `load_capacity`
therefore no longer accepts exit inventory. It reads the valid closing broker
observation for bookend scans, or a fresh broker observation. No reviewed seed
or historical execution bridge is required. The existing short collection-time
fill check prevents a buy from disappearing between positions and pending orders.
Exit reconciliation remains separate and cannot supply capacity inputs.

Regression: with $600,000 NAV, $280,000 held and $15,000 pending, a proposed
100-share order at $100 becomes 50 shares. The old verified-exit-inventory path
incorrectly admitted all 100 shares when those holdings were absent from its
attribution. Both verified and unknown exit-inventory cases now use the broker.

Verification: 79 capacity/closing-inventory tests and 34 sizing/collector tests
passed. The new regression failed before the change and passed after it. A
read-only live broker query on September 16 returned valid capacity with 11
held tickers and one pending ticker; no scan, order, or publication was run.

Activation is pending owner approval because this changes quantities staged
for live trading. Promote the same commit to the pinned runtime and its cloud
fallback, preserving the previous runtime commit and marker for rollback.
At preparation, the installed runtime was d4999e5993dabb1de5bec58a057efd915951673d
with fallback ref automation-runtime-2026-09-15.1. Deployment does not itself
authorize running a scan, rewriting Sheets, or placing orders.
