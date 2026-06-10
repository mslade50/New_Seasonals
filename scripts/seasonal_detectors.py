"""
seasonal_detectors.py - the six discretionary idea-detector channels.

AUTO-ASSEMBLED from the validated per-channel implementations. Each detector
has signature detect_<channel>(asof, ctx=None) -> list[dict] (candidates built
by se.make_candidate). The near-miss channel also exposes live_book_tickers()
used by the engine's negative filter. Validated against real local data
for 2026-06-09 before assembly.
"""
from __future__ import annotations


# =============================================================================
# CHANNEL: seasonal  ->  detect_seasonal (megacap equities, all-horizon swing tickets)
# =============================================================================
import pandas as pd
import numpy as np
import scripts.seasonal_edge as se


def detect_seasonal(asof, ctx=None) -> list:
    """Seasonal swing tickets over the megacap-equity universe (all horizons,
    R/R-gated). Thin wrapper over the shared scan_seasonal_tickets primitive."""
    min_rr = float((ctx or {}).get("min_rr", 2.0))
    return se.scan_seasonal_tickets(se.MEGACAP_TICKERS, asof, "detect_seasonal", min_rr=min_rr)


# =============================================================================
# CHANNEL: near_miss  ->  detect_near_miss  (plus live_book_tickers and helpers _build_nearmiss_ctx / _diag_check_signal / _gate_family / _near_enough)
# =============================================================================

import os, sys
import pandas as pd, numpy as np

# repo-root import bootstrap so `import daily_scan` works when assembled into scripts/seasonal_detectors.py
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import scripts.seasonal_edge as se

# NOTE: this is the ONLY channel permitted to import daily_scan / strategy_config.
try:
    import daily_scan as _ds
    from indicators import calculate_indicators as _calc_ind
    _ENGINE_OK, _ENGINE_ERR = True, None
except Exception as _e:  # pragma: no cover
    _ds, _calc_ind, _ENGINE_OK, _ENGINE_ERR = None, None, False, str(_e)

_ATR_SZNL_COLS = se.ATR_SZNL_COLS
_NM_CTX_CACHE: dict = {}


def _nm_norm(t):
    return se._norm_ticker(t)


def _build_nearmiss_ctx(asof):
    """Reconstruct per-ticker indicator frames EXACTLY as daily_scan's main loop
    does (calculate_indicators + atr_sznl merge), LOCAL master_prices.parquet only
    (via se.load_prices with include_overflow=False). Cached by asof date."""
    key = pd.Timestamp(asof).normalize()
    if key in _NM_CTX_CACHE:
        return _NM_CTX_CACHE[key]
    if not _ENGINE_OK:
        ctx = {"ok": False, "err": _ENGINE_ERR, "asof": key}
        _NM_CTX_CACHE[key] = ctx
        return ctx
    sznl_map = _ds.load_seasonal_map()
    atr_sznl_map = _ds.load_atr_seasonal_map()
    book = _ds.build_effective_strategy_book("liquid")
    tickers = set()
    for s in book:
        tickers.update(s["universe_tickers"])
    tickers |= {"SPY", "^GSPC", "^VIX"}
    prices = se.load_prices(list(tickers), include_overflow=False)
    spy = prices.get("SPY")
    market_series = None
    if spy is not None and not spy.empty:
        tmp = spy.copy(); tmp["SMA200"] = tmp["Close"].rolling(200).mean()
        market_series = tmp["Close"] > tmp["SMA200"]
    vix = prices.get("^VIX")
    vix_series = vix["Close"] if (vix is not None and not vix.empty) else None
    last_dates = [df.index.max() for df in prices.values() if df is not None and not df.empty]
    data_asof = max([d for d in last_dates if d <= key], default=None) if last_dates else None
    calc_cache: dict = {}

    def calc_for(tk):
        t = _nm_norm(tk)
        if t in calc_cache:
            return calc_cache[t]
        df = prices.get(t)
        if df is None or df.empty:
            calc_cache[t] = None; return None
        df = df[df.index <= key]
        if len(df) < 250:
            calc_cache[t] = None; return None
        c = _calc_ind(df.copy(), sznl_map, t, market_series, vix_series)
        if atr_sznl_map and t in atr_sznl_map:
            ar = atr_sznl_map[t]; dts = c.index.normalize()
            for col in _ATR_SZNL_COLS:
                if col in ar.columns:
                    c[col] = ar[col].reindex(dts).values
        calc_cache[t] = c; return c

    ctx = {"ok": True, "asof": key, "data_asof": data_asof, "sznl_map": sznl_map,
           "atr_sznl_map": atr_sznl_map, "book": book, "prices": prices, "calc_for": calc_for}
    _NM_CTX_CACHE[key] = ctx
    return ctx


def _diag_check_signal(df, params, ticker=None):
    """Diagnostic clone of daily_scan.check_signal. Records (name, passed, margin,
    detail) for every ACTIVE gate instead of early-returning. margin is signed in
    the gate's native units: >0 == passed with that slack, <0 == failed by that
    much, None for boolean gates. VALIDATED equivalent: diag-all-pass <=>
    check_signal True across 1227 ticker-strategy evals on 2026-06-09 (0 mismatch)."""
    rec = []
    _exempt = getattr(_ds, "ETF_ATR_EXEMPT", {"SPY", "QQQ", "IWM", "DIA"})

    def add(name, passed, margin=None, detail=""):
        rec.append((name, bool(passed), (None if margin is None else float(margin)), detail))

    last = df.iloc[-1]
    if params.get("use_dow_filter", False):
        add("dow", last.name.dayofweek in params.get("allowed_days", []), None, f"day={last.name.dayofweek}")
    if "allowed_cycles" in params:
        ac = params["allowed_cycles"]
        if ac and len(ac) < 4:
            add("cycle", (last.name.year % 4) in ac, None, f"cyc={last.name.year % 4}")
    if params.get("use_month_filter", False):
        am = params.get("allowed_months", list(range(1, 13)))
        add("month", last.name.month in am, None, f"m={last.name.month}")
    mp = params.get("min_price", 0); add("min_price", last["Close"] >= mp, last["Close"] - mp, f"close={last['Close']:.2f}")
    mv = params.get("min_vol", 0); add("min_vol", last["vol_ma"] >= mv, last["vol_ma"] - mv, f"vol_ma={last['vol_ma']:.0f}")
    mn_a = params.get("min_age", 0); add("min_age", last["age_years"] >= mn_a, last["age_years"] - mn_a, f"age={last['age_years']:.1f}")
    mx_a = params.get("max_age", 100); add("max_age", last["age_years"] <= mx_a, mx_a - last["age_years"], f"age={last['age_years']:.1f}")
    if "ATR_Pct" in df.columns:
        cap = last["ATR_Pct"]
        if not (ticker and ticker.upper() in _exempt):
            mn = params.get("min_atr_pct", 0.0); add("min_atr_pct", cap >= mn, cap - mn, f"atr%={cap:.2f}")
        mx = params.get("max_atr_pct", 1000.0); add("max_atr_pct", cap <= mx, mx - cap, f"atr%={cap:.2f}")
    if params.get("use_today_return", False):
        tr = last.get("today_return_atr", 0)
        if pd.isna(tr): add("today_return", False, None, "NaN")
        else:
            rmin = params.get("return_min", -100); rmax = params.get("return_max", 100)
            add("today_return", (tr >= rmin) and (tr <= rmax), min(tr - rmin, rmax - tr), f"ret_atr={tr:.2f}")
    if params.get("use_atr_ret_filter", False):
        tr = last.get("today_return_atr", 0)
        if pd.isna(tr): add("atr_ret", False, None, "NaN")
        else:
            rmin = params.get("atr_ret_min", -100); rmax = params.get("atr_ret_max", 100)
            add("atr_ret", (tr >= rmin) and (tr <= rmax), min(tr - rmin, rmax - tr), f"ret_atr={tr:.2f}")
    topt = params.get("trend_filter", "None")
    if topt == "Price > 200 SMA":
        add("trend", last["Close"] > last["SMA200"], last["Close"] - last["SMA200"], "px>200sma")
    elif topt == "Price > Rising 200 SMA":
        prev = df.iloc[-2]
        add("trend", (last["Close"] > last["SMA200"]) and (last["SMA200"] > prev["SMA200"]), last["Close"] - last["SMA200"], "px>rising200")
    elif topt == "Not Below Declining 200 SMA":
        prev = df.iloc[-2]
        add("trend", not ((last["Close"] < last["SMA200"]) and (last["SMA200"] < prev["SMA200"])), None, "not below declining 200")
    elif topt == "Price < 200 SMA":
        add("trend", last["Close"] < last["SMA200"], last["SMA200"] - last["Close"], "px<200sma")
    elif topt == "Price < Falling 200 SMA":
        prev = df.iloc[-2]
        add("trend", (last["Close"] < last["SMA200"]) and (last["SMA200"] < prev["SMA200"]), last["SMA200"] - last["Close"], "px<falling200")
    elif "Market" in topt or "SPY" in topt:
        if "Market_Above_SMA200" in df.columns:
            above = last["Market_Above_SMA200"]
            if ">" in topt: add("mkt_trend", bool(above), None, "mkt>200")
            if "<" in topt: add("mkt_trend", not bool(above), None, "mkt<200")
    for maf in params.get("ma_consec_filters", []):
        length = maf["length"]; col = f"SMA{length}"
        if col not in df.columns: continue
        mask = (df["Close"] > df[col]) if maf["logic"] == "Above" else (df["Close"] < df[col])
        consec = maf.get("consec", 1)
        run = 0
        for v in mask.values[::-1]:
            if v: run += 1
            else: break
        ok = (mask.rolling(consec).sum() == consec).iloc[-1] if consec > 1 else bool(mask.iloc[-1])
        add(f"ma_consec_{length}_{maf['logic']}", ok, run - consec, f"run={run} need>={consec}")
    if params.get("use_range_atr_filter", False):
        atr = last.get("ATR", 0)
        if atr > 0:
            ria = (last["High"] - last["Low"]) / atr; logic = params.get("range_atr_logic", "Between")
            if logic == ">": add("range_atr", ria > params.get("range_atr_min", 0), ria - params.get("range_atr_min", 0), f"ria={ria:.2f}")
            elif logic == "<": add("range_atr", ria < params.get("range_atr_max", 99), params.get("range_atr_max", 99) - ria, f"ria={ria:.2f}")
            else:
                lo = params.get("range_atr_min", 0); hi = params.get("range_atr_max", 99)
                add("range_atr", lo <= ria <= hi, min(ria - lo, hi - ria), f"ria={ria:.2f}")
    if params.get("require_close_gt_open", False):
        add("green_candle", last["Close"] > last["Open"], last["Close"] - last["Open"], "close>open")
    bk = params.get("breakout_mode", "None")
    if bk != "None":
        prev = df.iloc[-2]
        if bk == "Close > Prev Day High": add("breakout", last["Close"] > prev["High"], last["Close"] - prev["High"], "close>prevH")
        elif bk == "Close < Prev Day Low": add("breakout", last["Close"] < prev["Low"], prev["Low"] - last["Close"], "close<prevL")
    if params.get("vol_gt_prev", False):
        prev = df.iloc[-2]; add("vol_gt_prev", last["Volume"] > prev["Volume"], last["Volume"] - prev["Volume"], "vol>prev")
    if params.get("use_range_filter", False):
        rn = last["RangePct"] * 100; rmin = params.get("range_min", 0); rmax = params.get("range_max", 100)
        add("range_pct", (rn >= rmin) and (rn <= rmax), min(rn - rmin, rmax - rn), f"rangepct={rn:.1f} in[{rmin},{rmax}]")
    _pf = list(params.get("perf_filters", []))
    if params.get("use_perf_rank", False):
        legacy = {"window": params["perf_window"], "logic": params["perf_logic"], "thresh": params["perf_thresh"], "consecutive": params.get("perf_consecutive", 1)}
        if not any(pf["window"] == legacy["window"] and pf["logic"] == legacy["logic"] and pf["thresh"] == legacy["thresh"] for pf in _pf):
            _pf.append(legacy)
    for pf in _pf:
        col = f"rank_ret_{pf['window']}d"; consec = pf.get("consecutive", 1)
        if col not in df.columns:
            add(f"perf_{pf['window']}d", False, None, f"{col} missing"); continue
        val = df[col].iloc[-1]; thr = pf["thresh"]; thrmax = pf.get("thresh_max", 100.0); logic = pf["logic"]
        if logic == "<": cond_f = df[col] < thr; margin = thr - val
        elif logic == "Between": cond_f = (df[col] >= thr) & (df[col] <= thrmax); margin = min(val - thr, thrmax - val)
        elif logic == "Not Between": cond_f = (df[col] < thr) | (df[col] > thrmax); margin = max(thr - val, val - thrmax)
        else: cond_f = df[col] > thr; margin = val - thr
        ok = (cond_f.rolling(consec).sum() == consec).iloc[-1] if consec > 1 else bool(cond_f.iloc[-1])
        if pd.isna(val): add(f"perf_{pf['window']}d_{logic}", False, None, f"{col} NaN")
        else:
            band = f"{logic}{thr}" + (f"-{thrmax}" if logic in ("Between", "Not Between") else "")
            cstr = f" x{consec}" if consec > 1 else ""
            add(f"perf_{pf['window']}d_{logic}", ok, margin, f"{col}={val:.1f} {band}{cstr}")
    for asf in params.get("atr_sznl_filters", []):
        col = f"atr_sznl_{asf['window']}d"
        if col not in df.columns:
            add(f"atr_sznl_{asf['window']}d", False, None, f"{col} missing"); continue
        val = df[col].iloc[-1]; consec = asf.get("consecutive", 1); logic = asf.get("logic", ">")
        thr = asf["thresh"]; thrmax = asf.get("thresh_max", 100.0)
        if logic == "<": cond_f = df[col] < thr; margin = thr - val
        elif logic == ">": cond_f = df[col] > thr; margin = val - thr
        elif logic == "Between": cond_f = (df[col] >= thr) & (df[col] <= thrmax); margin = min(val - thr, thrmax - val)
        else: continue
        ok = (cond_f.rolling(consec).sum() == consec).iloc[-1] if consec > 1 else bool(cond_f.iloc[-1])
        if pd.isna(val): add(f"atr_sznl_{asf['window']}d_{logic}", False, None, f"{col} NaN")
        else:
            band = f"{logic}{thr}" + (f"-{thrmax}" if logic == "Between" else "")
            add(f"atr_sznl_{asf['window']}d_{logic}", ok, margin, f"{col}={val:.1f} {band}")
    if params.get("use_gap_filter", False):
        lb = params.get("gap_lookback", 21); col = f"GapCount_{lb}" if f"GapCount_{lb}" in df.columns else "GapCount_21"
        gv = last.get(col, 0); gl = params.get("gap_logic", ">"); gt = params.get("gap_thresh", 0)
        if gl == ">": add("gap", gv > gt, gv - gt, f"gap={gv}")
        elif gl == "<": add("gap", gv < gt, gt - gv, f"gap={gv}")
        elif gl == "=": add("gap", gv == gt, gv - gt, f"gap={gv}")
    if params.get("use_acc_count_filter", False):
        w = params.get("acc_count_window", 21); col = f"AccCount_{w}"
        if col in df.columns:
            av = last[col]; al = params.get("acc_count_logic", "="); at = params.get("acc_count_thresh", 0)
            if al == "=": add("acc_count", av == at, av - at, f"acc={av}")
            elif al == ">": add("acc_count", av > at, av - at, f"acc={av}")
            elif al == "<": add("acc_count", av < at, at - av, f"acc={av}")
    if params.get("use_dist_count_filter", False):
        w = params.get("dist_count_window", 21); col = f"DistCount_{w}"
        if col in df.columns:
            dv = last[col]; dl = params.get("dist_count_logic", ">"); dt = params.get("dist_count_thresh", 0)
            if dl == "=": add("dist_count", dv == dt, dv - dt, f"dist={dv}")
            elif dl == ">": add("dist_count", dv > dt, dv - dt, f"dist={dv}")
            elif dl == "<": add("dist_count", dv < dt, dt - dv, f"dist={dv}")
    if params.get("use_ma_dist_filter", False) or params.get("use_dist_filter", False):
        mt = params.get("dist_ma_type", "SMA 200"); cm = {"52-Week High": "High_52w", "All-Time High": "ATH_Level"}
        mc = cm.get(mt, mt.replace(" ", ""))
        if mc in df.columns:
            mvv = last[mc]; atr = last["ATR"]; close = last["Close"]
            du = (((close - mvv) / mvv) / (atr / close)) if (atr > 0 and mvv > 0 and close > 0) else 0
            dl = params.get("dist_logic", "Between"); dmin = params.get("dist_min", 0); dmax = params.get("dist_max", 0)
            if dl == "Greater Than (>)": add("ma_dist", du > dmin, du - dmin, f"dist={du:.2f}")
            elif dl == "Less Than (<)": add("ma_dist", du < dmax, dmax - du, f"dist={du:.2f}")
            elif dl == "Between": add("ma_dist", dmin <= du <= dmax, min(du - dmin, dmax - du), f"dist={du:.2f}")
    if params.get("use_sznl", False):
        sv = df["Sznl"].iloc[-1]
        if params["sznl_logic"] == "<": raw = df["Sznl"] < params["sznl_thresh"]; margin = params["sznl_thresh"] - sv
        else: raw = df["Sznl"] > params["sznl_thresh"]; margin = sv - params["sznl_thresh"]
        fin = raw
        if params.get("sznl_first_instance", False):
            lb = params.get("sznl_lookback", 21); fin = raw & (raw.shift(1).rolling(lb).sum() == 0)
        if pd.isna(sv): add("sznl", False, None, "Sznl NaN")
        else: add("sznl", bool(fin.iloc[-1]), margin, f"sznl={sv:.1f} {params['sznl_logic']}{params['sznl_thresh']}")
    if params.get("use_market_sznl", False):
        val = last.get("Mkt_Sznl_Ref", np.nan)
        if params["market_sznl_logic"] == "<": add("mkt_sznl", val < params["market_sznl_thresh"], params["market_sznl_thresh"] - val, f"mktsznl={val}")
        else: add("mkt_sznl", val > params["market_sznl_thresh"], val - params["market_sznl_thresh"], f"mktsznl={val}")
    if params.get("use_52w", False):
        cond = df["is_52w_high"] if params["52w_type"] == "New 52w High" else df["is_52w_low"]
        if params.get("52w_first_instance", True):
            lb = params.get("52w_lookback", 21); cond = cond & (cond.shift(1).rolling(lb).sum() == 0)
        add("is_52w", bool(cond.iloc[-1]), None, params["52w_type"])
    if params.get("exclude_52w_high", False):
        add("excl_52wh", not last["is_52w_high"], None, "not 52wH")
    if params.get("use_ath", False):
        if params.get("ath_type") == "Today is ATH": add("ath", bool(last["is_ath"]), None, "is ATH")
        else: add("ath", not bool(last["is_ath"]), None, "not ATH")
    if params.get("use_recent_52w", False):
        lb = params.get("recent_52w_lookback", 21); r = df["is_52w_high"].rolling(window=lb, min_periods=1).max().iloc[-1]
        add("recent_52w", (not bool(r)) if params.get("recent_52w_invert", False) else bool(r), None, "recent52wH")
    if params.get("use_recent_52w_low", False):
        lb = params.get("recent_52w_low_lookback", 21); r = df["is_52w_low"].rolling(window=lb, min_periods=1).max().iloc[-1]
        add("recent_52w_low", (not bool(r)) if params.get("recent_52w_low_invert", False) else bool(r), None, "recent52wL")
    if params.get("use_recent_ath", False):
        lb = params.get("ath_lookback_days", 21); r = df["is_ath"].rolling(window=lb, min_periods=1).max().iloc[-1]
        add("recent_ath", (not bool(r)) if params.get("recent_ath_invert", False) else bool(r), None, "recentATH")
    if params.get("use_vix_filter", False):
        vmin = params.get("vix_min", 0); vmax = params.get("vix_max", 100); vv = last.get("VIX_Value", 0)
        add("vix", (vv >= vmin) and (vv <= vmax), min(vv - vmin, vmax - vv), f"vix={vv:.1f}")
    if params.get("use_ref_ticker_filter", False) and params.get("ref_filters"):
        for rf in params["ref_filters"]:
            col = f"Ref_rank_ret_{rf['window']}d"; val = last.get(col, 50.0)
            if rf["logic"] == "<": add(f"ref_{rf['window']}d", val < rf["thresh"], rf["thresh"] - val, f"{col}={val:.1f}")
            if rf["logic"] == ">": add(f"ref_{rf['window']}d", val > rf["thresh"], val - rf["thresh"], f"{col}={val:.1f}")
    if params.get("use_xsec_filter", False) and params.get("xsec_filters"):
        for xf in params["xsec_filters"]:
            col = f"xsec_rank_ret_{xf['window']}d"; val = last.get(col, 50.0)
            if xf["logic"] == "<": add(f"xsec_{xf['window']}d", val < xf["thresh"], xf["thresh"] - val, f"{col}={val:.1f}")
            if xf["logic"] == ">": add(f"xsec_{xf['window']}d", val > xf["thresh"], val - xf["thresh"], f"{col}={val:.1f}")
            if xf["logic"] == "Between":
                hi = xf.get("thresh_max", 100.0); add(f"xsec_{xf['window']}d", xf["thresh"] <= val <= hi, min(val - xf["thresh"], hi - val), f"{col}={val:.1f}")
    for gi, group in enumerate(params.get("or_filter_groups", [])):
        any_pass = False
        for cond in group:
            ctype = cond.get("type", "perf"); window = cond["window"]; logic = cond["logic"]; thr = cond["thresh"]
            if ctype == "perf": val = last.get(f"rank_ret_{window}d", 50.0)
            elif ctype == "xsec": val = last.get(f"xsec_rank_ret_{window}d", 50.0)
            else: continue
            if logic == "<" and val < thr: any_pass = True
            if logic == ">" and val > thr: any_pass = True
        add(f"or_group_{gi}", any_pass, None, "OR group")
    if params.get("use_vol", False):
        vt = params["vol_thresh"]; add("vol_ratio", last["vol_ratio"] > vt, last["vol_ratio"] - vt, f"volratio={last['vol_ratio']:.2f}>{vt}")
    dial_filters = params.get("dial_filters", [])
    if dial_filters:
        frag = _ds._get_fragility_df_cached()
        if frag is None or frag.empty:
            add("dial", False, None, "fragility cache missing")
        else:
            sd = df.index[-1]
            try: sd = pd.Timestamp(sd).normalize().tz_localize(None)
            except (TypeError, AttributeError): sd = pd.Timestamp(sd).normalize()
            for df_f in dial_filters:
                dc = df_f.get("dial")
                if dc not in frag.columns:
                    add(f"dial_{dc}", False, None, "dial col missing"); continue
                win = max(1, int(df_f.get("window", 1)))
                ser = frag[dc].rolling(win, min_periods=win).mean() if win > 1 else frag[dc]
                try: val = float(ser.reindex([sd], method="ffill").iloc[0])
                except (IndexError, KeyError): add(f"dial_{dc}", False, None, "dial reindex fail"); continue
                if pd.isna(val): add(f"dial_{dc}", False, None, "dial NaN"); continue
                thr = float(df_f.get("thresh", 0)); logic = df_f.get("logic", ">")
                if logic == ">": add(f"dial_{dc}", val > thr, val - thr, f"dial={val:.1f}")
                elif logic == "<": add(f"dial_{dc}", val < thr, thr - val, f"dial={val:.1f}")
                elif logic == ">=": add(f"dial_{dc}", val >= thr, val - thr, f"dial={val:.1f}")
                elif logic == "<=": add(f"dial_{dc}", val <= thr, thr - val, f"dial={val:.1f}")
    if params.get("use_vol_rank", False):
        val = last["vol_ratio_10d_rank"]
        if params["vol_rank_logic"] == "<": add("vol_rank", val < params["vol_rank_thresh"], params["vol_rank_thresh"] - val, f"volrank={val:.1f}")
        else: add("vol_rank", val > params["vol_rank_thresh"], val - params["vol_rank_thresh"], f"volrank={val:.1f}")
    return rec


# ----- GOAL B: live-book negative filter (exact live check_signal) -----
def live_book_tickers(asof, ctx=None) -> set:
    c = _build_nearmiss_ctx(asof)
    if not c.get("ok"):
        return set()
    out = set()
    for s in c["book"]:
        st = s["settings"]
        for tk in s["universe_tickers"]:
            calc = c["calc_for"](tk)
            if calc is None or len(calc) < 250:
                continue
            t = _nm_norm(tk)
            try:
                if _ds.check_signal(calc, st, c["sznl_map"], ticker=t):
                    out.add(t)
            except Exception:
                continue
    return out


# ----- GOAL A: near-miss detector -----
_PCTILE_GATE_TOL = 8.0   # perf_/atr_sznl_/xsec_/vol_rank/sznl/range_pct/dial/ref: 0-100 scale, pts
_ATR_UNIT_TOL = 0.20     # today_return / atr_ret gates: ATR units


def _gate_family(name):
    if name.startswith(("perf", "atr_sznl", "xsec", "vol_rank", "sznl", "range_pct", "dial", "ref")) and "ratio" not in name:
        return "pctile"
    if name.startswith(("today_return", "atr_ret")):
        return "atr_units"
    return "other"


def _near_enough(name, margin):
    if margin is None:
        return False, None
    fam = _gate_family(name); am = abs(margin)
    if fam == "pctile":
        return (am <= _PCTILE_GATE_TOL), f"{am:.1f}pts (0-100 scale)"
    if fam == "atr_units":
        return (am <= _ATR_UNIT_TOL), f"{am:.2f} ATR"
    return False, f"{margin:+.2f}"   # non-percentile gates: scale ambiguous, not surfaced


def detect_near_miss(asof, ctx=None) -> list:
    asof = pd.Timestamp(asof).normalize()
    c = _build_nearmiss_ctx(asof)
    out = []
    if not c.get("ok"):
        return out  # engine unavailable -> graceful empty
    data_asof = c.get("data_asof")
    stale_note = ""
    if data_asof is not None and data_asof < asof:
        stale_note = (f"local prices last bar {data_asof.date()} "
                      f"(evaluated on that bar; asof {asof.date()} not yet in parquet)")
    live = live_book_tickers(asof)
    hold_map = {s["name"]: int(s["execution"].get("hold_days", 5) or 5) for s in c["book"]}
    cand_rows = []
    for s in c["book"]:
        st = s["settings"]; sname = s["name"]
        direction = "short" if str(st.get("trade_direction", "Long")).lower().startswith("short") else "long"
        hold = hold_map.get(sname, 5)
        for tk in s["universe_tickers"]:
            calc = c["calc_for"](tk)
            if calc is None or len(calc) < 250:
                continue
            t = _nm_norm(tk)
            recs = _diag_check_signal(calc, st, ticker=t)
            fails = [r for r in recs if not r[1]]
            if len(fails) != 1:      # need EXACTLY one failed gate
                continue
            gname, _, gmargin, gdetail = fails[0]
            ok, mdisp = _near_enough(gname, gmargin)
            if not ok or t in live:  # margin must be tight AND not already a live signal
                continue
            cand_rows.append((sname, t, direction, gname, gmargin, gdetail, len(recs), hold, mdisp))
    # realized-outcome gate (the ORLY guard): the seasonal window over the
    # strategy's hold horizon must lean the strategy DIRECTION in BOTH midterm
    # (2026 phase) and all-years samples. Kills "low rank on a structural
    # uptrend" false shorts.
    for (sname, t, direction, gname, gmargin, gdetail, n_gates, hold, mdisp) in cand_rows:
        px = c["prices"].get(t)
        if px is None or px.empty:
            continue
        mt = se.seasonal_window_returns(px, asof, hold, cycle_phase_filter=2)
        al = se.seasonal_window_returns(px, asof, hold)
        if not mt or mt.get("insufficient") or not al or al.get("insufficient"):
            continue
        mt_down = mt["pct_down"]; al_down = al["pct_down"]
        if direction == "short":
            supports = (mt_down >= 0.60 and al_down >= 0.55 and mt["mean"] < 0)
            edge = (mt_down + al_down) / 2.0
        else:
            supports = ((1 - mt_down) >= 0.60 and (1 - al_down) >= 0.55 and mt["mean"] > 0)
            edge = ((1 - mt_down) + (1 - al_down)) / 2.0
        if not supports:
            continue
        mt_dirn = mt["n_down"] if direction == "short" else mt["n_up"]
        al_dirn = al["n_down"] if direction == "short" else al["n_up"]
        dirn_word = "down" if direction == "short" else "up"
        gpass = n_gates - 1
        headline = (f"near-{sname}-{direction}: cleared {gpass}/{n_gates} gates; "
                    f"missed only {gname} ({gdetail}), margin {mdisp}")
        evidence = {
            "missed_gate": f"{gname} ({gdetail})", "margin": mdisp,
            f"midterm {hold}d": f"{mt_dirn}/{mt['n']} {dirn_word} ({mt['mean']:+.2%})",
            f"all-years {hold}d": f"{al_dirn}/{al['n']} {dirn_word} ({al['mean']:+.2%})",
            "gates_cleared": f"{gpass}/{n_gates}",
        }
        notes = "seasonal-confirmed near-miss; would fire if the one gate flips."
        if stale_note:
            notes += " " + stale_note
        sk = float(edge) * 100.0 - (abs(gmargin) if gmargin is not None else 0.0)
        out.append(se.make_candidate(
            channel="near_miss", ticker=t, direction=direction, headline=headline,
            horizon=f"{hold}d", evidence=evidence, conviction="C",
            p_value=al.get("p_value"), sort_key=sk, asof=asof, notes=notes))
    out.sort(key=lambda x: -x["sort_key"])
    return out


# =============================================================================
# CHANNEL: regime_sleeve  ->  detect_regime_sleeve
# =============================================================================

import os
import numpy as np
import pandas as pd

import scripts.seasonal_edge as se

_LEDGER_PATH = os.path.join(se.DATA_DIR, "backtest_trades_full.parquet")

# Per-(strategy, direction) cell must carry at least this many trades in the
# regime slice before we will report it. Below this, win% is noise.
_MIN_CELL_N = 15
# How many sleeves to surface on each side (favorable / unfavorable).
_TOP_K_SIDE = 4
# A sleeve gap must be at least this many win-% points to be worth a line.
_MIN_GAP_PP = 3.0


def _load_ledger():
    if not os.path.exists(_LEDGER_PATH):
        return None
    df = pd.read_parquet(
        _LEDGER_PATH,
        columns=["Strategy", "Direction", "Signal Date", "Return_Pct", "R_Multiple"],
    )
    df = df.copy()
    df["Signal Date"] = pd.to_datetime(df["Signal Date"])
    df = df.dropna(subset=["Signal Date", "Return_Pct"])
    df["win"] = (df["Return_Pct"] > 0).astype(float)
    df["cycle"] = df["Signal Date"].dt.year % 4
    df["month"] = df["Signal Date"].dt.month
    return df


def _vix_bucket(asof):
    """Current VIX tercile bucket vs full history (low/mid/high). Optional;
    returns (label, level, edges) or (None, nan, None) if VIX is unavailable."""
    try:
        v = se.load_prices(["^VIX"]).get("^VIX")
        if v is None or v.empty:
            return None, float("nan"), None
        close = v["Close"].dropna()
        close = close[close.index <= pd.Timestamp(asof).normalize()]
        if close.empty:
            return None, float("nan"), None
        lo, hi = close.quantile([0.33, 0.66]).values
        lvl = float(close.iloc[-1])
        lbl = "low" if lvl < lo else ("mid" if lvl < hi else "high")
        return lbl, lvl, (float(lo), float(hi))
    except Exception:
        return None, float("nan"), None


def _cell_stats(regime_df, full_df):
    """Per-sleeve (Strategy, Direction) regime vs all-years comparison.
    Returns rows that clear the n>=_MIN_CELL_N gate."""
    rows = []
    for (strat, direction), sub in regime_df.groupby(["Strategy", "Direction"]):
        n = len(sub)
        if n < _MIN_CELL_N:
            continue
        base = full_df[(full_df["Strategy"] == strat) & (full_df["Direction"] == direction)]
        if base.empty:
            continue
        regime_win = sub["win"].mean() * 100.0
        base_win = base["win"].mean() * 100.0
        rows.append({
            "strategy": strat,
            "direction": direction,
            "n": int(n),
            "regime_win": regime_win,
            "base_win": base_win,
            "win_gap": regime_win - base_win,
            "regime_R": float(sub["R_Multiple"].mean()),
            "base_R": float(base["R_Multiple"].mean()),
            "n_base": int(len(base)),
        })
    return rows


def detect_regime_sleeve(asof, ctx=None):
    asof = pd.Timestamp(asof).normalize()
    df = _load_ledger()
    cands = []

    if df is None or df.empty:
        return cands

    phase = se.cycle_phase(asof.year)            # 2 == midterm in 2026
    phase_lbl = se.cycle_label(asof.year)
    month = int(asof.month)
    month_name = asof.strftime("%b")

    vbuck, vlvl, _edges = _vix_bucket(asof)
    vix_str = f"VIX {vlvl:.1f} ({vbuck} tercile)" if vbuck else "VIX n/a"

    # ---- Book-level regime context line -------------------------------------
    regime = df[df["cycle"] == phase]
    if len(regime) >= _MIN_CELL_N:
        book_regime_win = regime["win"].mean() * 100.0
        book_all_win = df["win"].mean() * 100.0
        book_gap = book_regime_win - book_all_win
        book_regime_R = float(regime["R_Multiple"].mean())
        book_all_R = float(df["R_Multiple"].mean())
        stance = "de-risk" if book_gap <= -2.0 else ("press" if book_gap >= 2.0 else "neutral")
        head = (f"{phase_lbl} regime: book win% {book_regime_win:.1f} vs {book_all_win:.1f} "
                f"all-years, n={len(regime)} - {stance}")
        cands.append(se.make_candidate(
            channel="regime_sleeve",
            ticker="BOOK",
            direction="context",
            headline=head,
            horizon="regime",
            evidence={
                "regime": f"{phase_lbl} (cycle={phase}), {month_name}",
                "book_win": f"{book_regime_win:.1f}% vs {book_all_win:.1f}% (gap {book_gap:+.1f}pp)",
                "book_R": f"{book_regime_R:+.2f}R vs {book_all_R:+.2f}R all-years",
                "n": f"{len(regime)} midterm trades",
                "vol": vix_str,
            },
            conviction="A" if abs(book_gap) >= 5 else "B",
            sort_key=abs(book_gap) * len(regime),
            asof=asof,
            notes=(f"Sleeve cells gated at n>={_MIN_CELL_N}; favorable=lean into, "
                   f"unfavorable=fade. Current {vix_str}."),
        ))

    # ---- Sleeve-level fade / lean-into candidates ---------------------------
    # Primary regime slice = cycle phase (midterm). Well-populated; the month
    # intersection is too thin to gate at n>=15 for more than one cell, so the
    # month is reported as supporting context inside evidence when available.
    cell_rows = _cell_stats(regime, df)

    # month-conditioned (all cycles) win% per sleeve, for a supporting note
    month_df = df[df["month"] == month]
    month_lookup = {}
    for (strat, direction), sub in month_df.groupby(["Strategy", "Direction"]):
        if len(sub) >= _MIN_CELL_N:
            month_lookup[(strat, direction)] = (sub["win"].mean() * 100.0, len(sub))

    fav = sorted([r for r in cell_rows if r["win_gap"] >= _MIN_GAP_PP],
                 key=lambda r: -(abs(r["win_gap"]) * r["n"]))
    unf = sorted([r for r in cell_rows if r["win_gap"] <= -_MIN_GAP_PP],
                 key=lambda r: -(abs(r["win_gap"]) * r["n"]))

    def _emit(r, side):
        strat, direction = r["strategy"], r["direction"]
        lean = "lean into" if side == "fav" else "fade"
        head = (f"{phase_lbl} {strat} ({direction}): win% {r['regime_win']:.1f} vs "
                f"{r['base_win']:.1f} all-years (n={r['n']}) - {lean}")
        ev = {
            "regime_win": f"{r['regime_win']:.1f}% vs {r['base_win']:.1f}% (gap {r['win_gap']:+.1f}pp)",
            "regime_R": f"{r['regime_R']:+.2f}R vs {r['base_R']:+.2f}R",
            "n": f"{r['n']} midterm / {r['n_base']} all-years",
        }
        mk = month_lookup.get((strat, direction))
        if mk is not None:
            ev["month"] = f"{month_name} win% {mk[0]:.1f} (n={mk[1]}, all cycles)"
        conv = "A" if (abs(r["win_gap"]) >= 10 and r["n"] >= 40) else ("B" if abs(r["win_gap"]) >= 5 else "C")
        return se.make_candidate(
            channel="regime_sleeve",
            ticker=strat,
            direction="context",
            headline=head,
            horizon="regime",
            evidence=ev,
            conviction=conv,
            sort_key=abs(r["win_gap"]) * r["n"],
            asof=asof,
            notes=f"{vix_str}. Sleeve = Strategy x Direction; {lean} in {phase_lbl} years.",
        )

    for r in fav[:_TOP_K_SIDE]:
        cands.append(_emit(r, "fav"))
    for r in unf[:_TOP_K_SIDE]:
        cands.append(_emit(r, "unf"))

    return cands


# =============================================================================
# CHANNEL: cross_asset  ->  detect_cross_asset (macro / cross-asset swing tickets)
# =============================================================================

def detect_cross_asset(asof, ctx=None) -> list:
    """Seasonal swing tickets over the macro / cross-asset universe (indices,
    commodities, FX, crypto, bonds). Thin wrapper over scan_seasonal_tickets."""
    min_rr = float((ctx or {}).get("min_rr", 2.0))
    return se.scan_seasonal_tickets(se.MACRO_TICKERS, asof, "detect_cross_asset", min_rr=min_rr)


# =============================================================================
# CHANNEL: sentiment  ->  detect_sentiment
# =============================================================================

import os
import numpy as np
import pandas as pd

import scripts.seasonal_edge as se

_PUTCALL_PATH = os.path.join(se.DATA_DIR, "cboe_putcall.parquet")


def _pctile_rank(series: pd.Series, value: float) -> float:
    """Full-history midrank percentile in [0,100] (== scipy percentileofscore
    kind='mean'). Handles ties so a value sitting on a cluster lands between the
    strict-less and less-or-equal bounds (matters here: ~16 ties at 0.67)."""
    s = pd.Series(series).dropna().values
    if len(s) == 0:
        return float("nan")
    below = float((s < value).sum())
    equal = float((s == value).sum())
    return (below + 0.5 * equal) / len(s) * 100.0


def detect_sentiment(asof, ctx=None) -> list:
    """CBOE put/call sentiment -> market-level 'context' candidate.

    Percentile-ranks the latest equity AND total P/C (asof, else last available)
    vs full ~2.4yr history. Gate: |pctile-50| > 30 on EITHER reading (i.e. the
    driving reading is >80th or <20th). High P/C = fear -> contrarian-long; low
    P/C = complacency -> contrarian-short/hedge. Validates the call with a real
    ^GSPC (SPY fallback) forward-return event study at similar equity-P/C
    readings via se.run_event_study over [5,10,21]d. Reports forward mean,
    pct_pos, and N. Conviction capped at B (thin history). direction='context',
    ticker='SPY' (market-level, no single tradeable name).
    """
    asof = pd.Timestamp(asof).normalize()
    out: list = []

    if not os.path.exists(_PUTCALL_PATH):
        return out
    pc = pd.read_parquet(_PUTCALL_PATH)
    if pc.empty or not {"equity", "total"}.issubset(pc.columns):
        return out
    pc = pc.sort_index()

    asof_pc = pc[pc.index <= asof]
    if asof_pc.empty:
        return out
    last_pc_date = asof_pc.index.max()

    # latest reading + full-history percentile for each series
    readings = {}
    for col in ("equity", "total"):
        ser = pc[col].dropna()
        latest_ser = asof_pc[col].dropna()
        if ser.empty or latest_ser.empty:
            continue
        val = float(latest_ser.iloc[-1])
        readings[col] = (val, _pctile_rank(ser, val))
    if not readings:
        return out

    # gate: |pctile-50| > 30 on at least one reading
    triggered = {k: v for k, v in readings.items() if abs(v[1] - 50.0) > 30.0}
    if not triggered:
        return out

    # direction from the most extreme triggered reading
    drive_col = max(triggered, key=lambda k: abs(triggered[k][1] - 50.0))
    drive_val, drive_pct = triggered[drive_col]
    high = drive_pct > 50.0  # high P/C = fear -> contrarian-long

    # --- forward-return event study (validation) ---
    eq_val = readings.get("equity", (None, None))[0]
    if eq_val is None:
        return out
    equity_hist = pc["equity"].dropna()

    px = se.load_one_price("^GSPC")
    proxy = "^GSPC"
    if px is None or "Close" not in px:
        px = se.load_one_price("SPY")
        proxy = "SPY"

    es = None
    if px is not None and "Close" in px:
        close = px["Close"].dropna().sort_index()
        # reindex equity P/C onto price dates; do NOT ffill past PC history so
        # the condition is only ever True inside the ~2.4yr window we actually
        # observe -> no synthetic signals on the 2000-2023 price span.
        eq_on_px = equity_hist.reindex(close.index)
        cond = (eq_on_px >= eq_val) if high else (eq_on_px <= eq_val)
        cond = cond.where(eq_on_px.notna(), False).fillna(False).astype(bool)
        es = se.run_event_study(close, cond, forward_windows=[5, 10, 21])

    # evidence (short pre-formatted strings)
    ev = {}
    for col in ("equity", "total"):
        if col in readings:
            v, p = readings[col]
            ev[f"{col} P/C"] = f"{v:.2f} (pctile {p:.0f})"
    p21 = None
    if es is not None:
        for w in (5, 10, 21):
            d = es["windows"].get(w)
            if d:
                ev[f"fwd {w}d"] = f"{d['mean']:+.2%} mean, {d['pct_pos']:.0%} pos, N={d['n']}"
                if w == 21:
                    p21 = d["p_value"]
        ev["episodes"] = f"{es['n_episodes']} declustered"
        ev["proxy"] = proxy

    if high:
        headline = (f"CBOE put/call elevated ({drive_col} {drive_val:.2f}, "
                    f"pctile {drive_pct:.0f}) - fear/washout, contrarian-long context")
    else:
        headline = (f"CBOE put/call depressed ({drive_col} {drive_val:.2f}, "
                    f"pctile {drive_pct:.0f}) - complacency, contrarian-short/hedge context")

    stale_days = (asof - last_pc_date).days
    notes = (f"Thin ~2.4yr P/C history (since {pc.index.min().date()}); "
             f"unconditional baseline is a bull-market window so absolute means "
             f"run rich - read the lift, not the level. Conviction capped at B. "
             f"Reading asof {last_pc_date.date()}"
             + (f" ({stale_days}d stale)." if stale_days > 4 else "."))

    out.append(se.make_candidate(
        channel="sentiment",
        ticker="SPY",
        direction="context",
        headline=headline,
        horizon="5-21d",
        evidence=ev,
        conviction="B",
        p_value=p21,
        sort_key=abs(drive_pct - 50.0),
        asof=asof,
        notes=notes,
    ))
    return out


# =============================================================================
# CHANNEL: analyst  ->  detect_analyst
# =============================================================================

import os
import numpy as np
import pandas as pd
import scripts.seasonal_edge as se

_ANALYST_PATH = os.path.join(se.DATA_DIR, "analyst_grades.parquet")

# Bullish/bearish grade lexicons (for transition direction + to sanity-check that
# an upgrade actually lands in bullish territory rather than a notch-up that stays
# neutral, e.g. Underweight -> Equal Weight).
_BULL_TOKENS = ("strong buy", "conviction buy", "buy", "outperform", "overweight",
                "accumulate", "add", "positive")
_NEUTRAL_TOKENS = ("hold", "neutral", "equal weight", "equal-weight", "market perform",
                   "sector perform", "in-line", "in line", "peer perform", "perform")
_BEAR_TOKENS = ("strong sell", "sell", "underperform", "underweight", "reduce",
                "negative", "sector underperform")


def _grade_bucket(g):
    """Map a free-text rating to -1 (bearish) / 0 (neutral) / +1 (bullish) / None."""
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    s = str(g).strip().lower()
    if not s:
        return None
    for tok in _BULL_TOKENS:
        if tok in s:
            return 1
    for tok in _BEAR_TOKENS:
        if tok in s:
            return -1
    for tok in _NEUTRAL_TOKENS:
        if tok in s:
            return 0
    return None


def detect_analyst(asof, ctx=None):
    """Rating-action clusters (analyst_grades.parquet) crossed with seasonal ranks.

    Source is STALE (~2026-05-04). We anchor the window to the last ~10 trading
    days OF AVAILABLE DATA, not vs asof. A name with >=2 distinct-firm upgrades
    (long) or >=2 distinct-firm downgrades (short) over that window is a cluster;
    a same-window up+down split is a 'transition'. Confluence with the seasonal
    cross-section (upgrades on a high 5/10d rank, downgrades on a low one) bumps
    conviction C->B, but ONLY for a clean >=2 same-direction cluster (a 1up/1dn
    transition is weaker and stays C even when the rank aligns). Everything else
    stays C. The data as-of date is printed in every headline + notes because the
    feed is materially stale.
    """
    asof = pd.Timestamp(asof).normalize()
    if not os.path.exists(_ANALYST_PATH):
        return []
    df = pd.read_parquet(_ANALYST_PATH)
    if df.empty:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df["gc_norm"] = df["grading_company"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()

    data_asof = df["date"].max()
    stale_td = int(np.busday_count(data_asof.date(), asof.date()))  # trading days behind
    asof_str = data_asof.date().isoformat()

    dates = np.sort(df["date"].unique())
    window_dates = dates[-10:]
    win_start = pd.Timestamp(window_dates[0]).date()
    win_label = f"{win_start}..{asof_str}"

    recent = df[df["date"].isin(window_dates)].copy()
    moves = recent[recent["action"].isin(["upgrade", "downgrade"])].copy()
    # dedupe identical firm+date+action (kills the 'Rothschild & Co'/'Rothschild& Co' double)
    moves = moves.drop_duplicates(subset=["ticker", "date", "action", "gc_norm"])
    if moves.empty:
        return []

    # Severe staleness guard: > 30 trading days behind -> degrade hard (still return,
    # marked stale, conviction floored to C, no seasonal bump).
    too_stale = stale_td > 30

    cs = se.seasonal_cross_section(asof=asof)

    rows = []
    for tkr, g in moves.groupby("ticker"):
        ups = g[g["action"] == "upgrade"]
        downs = g[g["action"] == "downgrade"]
        n_up, n_dn = len(ups), len(downs)

        if n_up >= 2 and n_dn == 0:
            direction, kind, n_act = "long", "upgrade cluster", n_up
        elif n_dn >= 2 and n_up == 0:
            direction, kind, n_act = "short", "downgrade cluster", n_dn
        elif n_up >= 1 and n_dn >= 1:
            # transition: net by bucket delta of the grades involved
            net = 0
            for _, r in g.iterrows():
                b = _grade_bucket(r["new_grade"])
                pb = _grade_bucket(r["previous_grade"])
                if b is not None and pb is not None:
                    net += (b - pb)
                elif r["action"] == "upgrade":
                    net += 1
                elif r["action"] == "downgrade":
                    net -= 1
            if net > 0:
                direction = "long"
            elif net < 0:
                direction = "short"
            else:
                direction = "short" if n_dn >= n_up else "long"
            kind, n_act = f"transition ({n_up}up/{n_dn}dn)", n_up + n_dn
        else:
            continue  # single action, not a cluster

        firms = sorted(set(g["gc_norm"]))
        last_date = g["date"].max().date()
        is_clean_cluster = (n_up >= 2 and n_dn == 0) or (n_dn >= 2 and n_up == 0)

        key = se._norm_ticker(tkr)
        r5 = r10 = float("nan")
        rank_date = None
        if key in cs.index:
            r5 = float(cs.loc[key, "atr_sznl_5d"])
            r10 = float(cs.loc[key, "atr_sznl_10d"])
            rank_date = cs.loc[key, "Date"].date()

        # confluence: longs want HIGH rank, shorts want LOW rank. Only a CLEAN
        # >=2 same-direction cluster earns the B bump -- a 1up/1dn transition is a
        # weaker, more ambiguous signal and stays C even if it sits on an
        # aligned rank.
        confluence = False
        if not too_stale and is_clean_cluster and not np.isnan(r5):
            if direction == "long" and (r5 >= 65 or r10 >= 65):
                confluence = True
            elif direction == "short" and (r5 <= 15 or r10 <= 15):
                confluence = True

        conviction = "B" if confluence else "C"

        # realized seasonal stat for the bumped (confluence) names -- a stat with N,
        # not a bare percentile.
        realized = None
        if confluence:
            px = se.load_one_price(tkr)
            if px is not None and len(px) > 300:
                mt = se.seasonal_window_returns(px, asof, 5, cycle_phase_filter=2)
                ally = se.seasonal_window_returns(px, asof, 5)
                if mt and not mt.get("insufficient") and ally and not ally.get("insufficient"):
                    realized = (
                        f"midterm {mt['n_down']}/{mt['n']} down ({mt['mean']:+.2%}), "
                        f"all-years {ally['n_down']}/{ally['n']} down ({ally['mean']:+.2%})"
                    )

        rank_str = f"{r5:.0f}/{r10:.0f}" if not np.isnan(r5) else "n/a"
        firms_str = ", ".join(f.title() for f in firms)
        head = (f"{kind}: {n_act} firm action(s) thru {last_date} (data as-of {asof_str}, "
                f"{stale_td}td stale) | 5/10d sznl rank {rank_str}")

        ev = {
            "data_asof": asof_str,
            "window": win_label,
            "actions": f"{n_up}up/{n_dn}dn ({firms_str})",
            "sznl_5_10d": rank_str,
        }
        if realized:
            ev["seasonal_realized"] = realized
        if confluence:
            ev["confluence"] = ("upgrades + high rank" if direction == "long"
                                else "downgrades + low rank")

        note_bits = [f"DATA AS-OF {asof_str} (analyst feed stale {stale_td} trading days vs {asof.date()})."]
        if too_stale:
            note_bits.append("Feed too stale to be actionable (>30td) -- informational only, conviction floored C, no seasonal bump.")
        if confluence:
            note_bits.append(f"Seasonal confluence: {ev['confluence']} (rank date {rank_date}).")
        else:
            note_bits.append("No seasonal confluence -> conviction C (rating cluster only).")
        notes = " ".join(note_bits)

        rank_extreme = 0.0
        if not np.isnan(r5):
            rank_extreme = (100 - r5) / 100 if direction == "short" else r5 / 100
        sort_key = (2.0 if confluence else 0.0) + 0.2 * n_act + 0.5 * rank_extreme

        rows.append((sort_key, se.make_candidate(
            channel="analyst_grades",
            ticker=tkr,
            direction=direction,
            headline=head,
            horizon="5-21d",
            evidence=ev,
            conviction=conviction,
            p_value=None,
            sort_key=sort_key,
            asof=asof,
            notes=notes,
        )))

    rows.sort(key=lambda x: -x[0])
    return [c for _, c in rows[:8]]
