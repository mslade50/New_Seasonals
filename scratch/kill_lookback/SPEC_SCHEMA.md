# Kill look-back: spec schema

One JSON object per kill, written BEFORE any forward price is looked at.
The goal is "the trade the candidate would have been had it shipped that
morning", best effort, not perfect.

```json
{
  "kill_id": "2026-08-07-k01",
  "tradeable": true,
  "why_not": null,
  "legs": [
    {"ticker": "TLT", "side": "LONG", "weight": 1.0},
    {"ticker": "SPY", "side": "SHORT", "weight": "beta"}
  ],
  "entry": {"type": "close", "date": "2026-08-07"},
  "horizon_td": 5,
  "gate_live": true,
  "spec_confidence": "medium",
  "basis": "h=5 from reason; MOC on pitch date (pitch default); tickers from kX_c3_r1.py"
}
```

Field rules

- `tickers`: Yahoo symbols as in `data/master_prices.parquet` (SPY, TLT, ^TNX is NOT
  tradeable, DX-Y.NYB, GC=F, CL=F, NG=F, UUP, MXN=X, JPY=X, single stocks by symbol).
  For vol: long vol -> UVXY or VXX, short vol -> SVXY. A "^VIX" idea -> UVXY/SVXY.
  USDJPY long = LONG "JPY=X"; long yen = SHORT "JPY=X". Long peso = SHORT "MXN=X".
- `weight`: notional fraction per leg (legs of a plain pair are 1.0 / 1.0). For
  "against beta-X" hedges use the string `"beta"` (the pricer fits a 63-session
  beta of the first leg on that leg before entry) or a number if the kill states it
  (e.g. "0.93-beta SPY" -> 0.93).
- `entry.type`: `close` (MOC) or `open` (MOO). `entry.date`: the concrete session.
  DEFAULT when nothing else is stated: `close` on the pitch date itself (the pitch
  runs pre-open on that date off the prior close, and ideas default to MOC that day).
  Event-anchored ideas: the session the candidate names (e.g. "k=-6 before NFP"
  -> the close 6 sessions before that NFP; "from QE-5" -> 5 sessions before
  quarter end). Use `data/macro_events.csv` for event dates. If that entry session
  is already past on the pitch date, use the pitch-date close instead.
- `horizon_td`: sessions held, close to close from entry. Take the horizon the kill
  reason / check script actually tested (h=5, h=2, "to the NFP close", "QE to QE+5").
  If an idea ends on an event close, convert to the session count. If several
  horizons were tested, take the one the candidate was pitched on (the first /
  pre-specified one), not the best one.
- `gate_live`: false when the reason says the trigger/gate was NOT live that day
  ("not live", "gate is unarmed", "missed its threshold"). Still spec it.
- `tradeable: false` + `why_not` only when no instrument/direction can honestly be
  assigned (e.g. "nearest-neighbour analogue across ten proxies", an unnamed
  cross-section with no names recoverable from the script, a pure research note).
  Multi-name baskets are fine if the names are recoverable from the check script
  (cap 15 legs, equal weight).
- When the title offers alternatives ("long SPY or SVXY"), take the first.
- `spec_confidence`: high = direction, instrument, entry and horizon all explicit;
  medium = one of them inferred; low = two or more inferred.
