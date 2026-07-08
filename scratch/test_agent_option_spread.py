"""Unit test for exec_agent.py's option_spread _describe/_validate/_preview —
imported directly, no WebSocket / broker / IBKR."""
import sys

sys.path.insert(0, r"C:\Users\McKinley Slade\OneDrive\trading_ibkr")
import exec_agent as ag

ok = True
def chk(n, c):
    global ok; ok = ok and bool(c); print(("OK   " if c else "FAIL ") + n)

def CMD(**kw):
    p = {"symbol": "XOM", "action": "BUY", "quantity": 5, "limit": 1.55, "tif": "DAY",
         "structure": "call_debit_vertical", "debit_risk": 1.55,
         "legs": [{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0, "ratio": 1},
                  {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0, "ratio": 1}],
         "strategy": "OLV", "signal_date": "2026-07-07"}
    p.update(kw)
    return {"type": "option_spread", "account": "pa", "payload": p}

# valid command passes
v, r = ag._validate(CMD())
chk("valid vertical passes", v and not r)

# describe + preview render
d = ag._describe(CMD())
chk("describe names combo + net", "combo" in d and "1.55" in d and "B105.0C" in d)
pv = ag._preview(CMD())
chk("preview: 2 legs + combo line", len(pv["legs"]) == 3 and pv["legs"][-1].startswith("COMBO"))
chk("preview summary: risk + comm + width + max profit",
    "$775" in pv["summary"] and "width 10" in pv["summary"] and "$4,225" in pv["summary"])

# rejections
chk("SELL rejected", not ag._validate(CMD(action="SELL"))[0])
chk("qty 0 rejected", not ag._validate(CMD(quantity=0))[0])
chk("qty > cap rejected", any("cap" in x for x in ag._validate(CMD(quantity=99))[1]))
chk("limit 0 rejected", not ag._validate(CMD(limit=0))[0])
chk("debit >= width rejected", any("width" in x for x in ag._validate(CMD(quantity=2, limit=10.0, debit_risk=10.0))[1]))
chk("1 leg rejected", any("2-4 legs" in x for x in ag._validate(CMD(legs=[{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0}]))[1]))
chk("bad right rejected", not ag._validate(CMD(legs=[
    {"side": "BUY", "right": "X", "expiry": "20260814", "strike": 105.0},
    {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0}]))[0])
chk("bad expiry rejected", not ag._validate(CMD(legs=[
    {"side": "BUY", "right": "C", "expiry": "2026-8", "strike": 105.0},
    {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0}]))[0])
chk("risk over acct cap rejected",
    any("cap" in x for x in ag._validate(CMD(quantity=10, debit_risk=30.0, limit=30.0, legs=[
        {"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0},
        {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 145.0}]))[1]))
chk("unknown account rejected", not ag._validate({**CMD(), "account": "zzz"})[0])

# NLV gate: with a fake book, risk > 5% NLV rejected
ag._BOOK["book"] = {"accounts": [{"key": "pa", "nlv": 10_000.0, "positions": [], "orders": []}]}
chk("risk > 5% NLV rejected", any("NLV" in x for x in ag._validate(CMD(quantity=8))[1]))
ag._BOOK["book"] = None

# live gate: option_spread not armed -> dry-run path
elig, why = ag._live_eligible(CMD())
chk("option_spread not live-eligible (unarmed)", not elig)

print("\n" + ("ALL PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
