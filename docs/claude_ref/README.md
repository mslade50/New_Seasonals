# CLAUDE.md reference docs

CLAUDE.md holds the live operating contract. These files hold the detail that used to sit
inside it: evidence numbers, study results, incident write-ups, history and retired features.
The text was moved close to verbatim on 2026-09-23. The only edits on the move were path
fixes (`C:\Users\mckin\OneDrive\trading_ibkr` to `C:\Users\McKinley Slade\OneDrive\trading_ibkr`,
and a TAB character in `C:\Scripts\trigger_cboe_putcall.ps1`) plus two deleted references to
files that no longer exist.

When a live rule changes, update CLAUDE.md and the matching doc here in the same change.

| Doc | Covers |
|---|---|
| `repo_structure.md` | Full annotated repo tree |
| `fragility_dial.md` | Risk dials, fragility contract, consumers, simple-dial shadow, negative results, NYSE net highs, breadth collection, signal downside tables, forward-returns table |
| `sizing.md` | GRM and base-bps tilt, daily risk caps, OLV recency ladder, overlap clamp, fragility risk bands (incl. the retired OLV band), P/C fear bands, gap-size derate |
| `ovs.md` | OVS earnings blackout, 2-path sizing, scale-out, precedence, EOD-DD, cycle-year tilt |
| `olv.md` | OLV vol-confirmed stop, notional cap, capacity fallback, retired OLV Book Cap, T+3 entry window |
| `strategies_3x_and_pilots.md` | 3x Bear Fade + same-day de-rate, 3x Leader Gap Fade, Monthly Weak Close, Trend Sleeve |
| `event_sleeve.md` | Event sleeve trades, flow, MOC encoding incident, visibility and journal |
| `daily_pitch.md` | Daily Pitch contract, short slate, stand-down, coverage incident, conventions |
| `daily_posts_and_context.md` | Daily Posts (X account) and the Market Context brief |
| `ledger_and_fills.md` | Ledger survivorship/provenance/replay caveats, stop-arming and stop-fill conventions, live fills store |
| `automation_and_r2.md` | Local-primary automation, R2 secrets, bucket contents, `cache_io.py`, Sunday pipeline |
| `automation_history.md` | Retired GitHub-first schedule, old Task Scheduler state, old AM dispatch architecture, retired radar digest |
| `private_site.md` | Private site tabs, payload contract, charts, sizing-basis rule, shared Denali risk tab |
| `radar.md` | Momentum radar staging and trail sync |
| `sheets.md` | Google Sheets tab layout |
