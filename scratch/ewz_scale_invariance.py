"""Prove (1) self-consistent re-run is scale-invariant, (2) phantom only in mixed basis."""

# RAW (live, pre-ex) basis from local cache / yfinance auto_adjust=False
raw = {
    "2026-06-08": dict(O=33.98, H=34.10, L=33.53, C=33.69),
    "2026-06-09": dict(O=33.99, H=34.28, L=33.58, C=33.92),
    "2026-06-10": dict(O=33.74, H=34.07, L=33.63, C=33.78),
}
# ADJUSTED (post-ex) basis from yfinance auto_adjust=True
adj = {
    "2026-06-08": dict(O=33.6596, H=33.7784, L=33.2138, C=33.3723),
    "2026-06-09": dict(O=33.6695, H=33.9567, L=33.2633, C=33.6001),
    "2026-06-10": dict(O=33.4218, H=33.7487, L=33.3129, C=33.4614),
}
f = 33.6001 / 33.92  # adj/raw close on 6/9
print(f"dividend factor f = {f:.6f}")

# ATR on 6/8 signal: raw 0.731428; under multiplicative adj it scales by f
atr_raw = 0.731428
atr_adj = atr_raw * f
mult = 0.25

def fill_long(limit, bars, days):
    for d in days:
        b = bars[d]
        if b["L"] <= limit:
            return d, b["L"]
    return None, None

days = ["2026-06-09", "2026-06-10"]  # forward bars after 6/8 signal

# --- RUN A: self-consistent RAW run (the live basis on staging day) ---
limit_raw = raw["2026-06-08"]["C"] - mult * atr_raw
print(f"\n[A] RAW self-consistent: limit={limit_raw:.4f} (round2={round(limit_raw,2)})")
print("    fill?", fill_long(limit_raw, raw, days))

# --- RUN B: self-consistent ADJUSTED run (clean post-ex re-run, recompute limit) ---
limit_adj = adj["2026-06-08"]["C"] - mult * atr_adj
print(f"\n[B] ADJ self-consistent: limit={limit_adj:.4f} (limit_raw*f={limit_raw*f:.4f})")
print("    fill?", fill_long(limit_adj, adj, days))

# --- RUN C: MIXED BASIS: frozen pre-ex limit 33.51 vs ADJUSTED bars ---
frozen = 33.51
print(f"\n[C] MIXED: frozen pre-ex limit {frozen} vs ADJUSTED post-ex bars")
print("    fill?", fill_long(frozen, adj, days))

# --- RUN D: MIXED BASIS the other way: frozen 33.51 vs RAW bars (= live truth) ---
print(f"\n[D] frozen {frozen} vs RAW bars (live truth)")
print("    fill?", fill_long(frozen, raw, days))

print("\nSCALE-INVARIANCE CHECK: limit_adj == f*limit_raw ?",
      abs(limit_adj - f * limit_raw) < 1e-9)
print("adj 6/9 low == f*raw 6/9 low ?",
      abs(adj['2026-06-09']['L'] - f * raw['2026-06-09']['L']) < 1e-3,
      f"({adj['2026-06-09']['L']:.4f} vs {f*raw['2026-06-09']['L']:.4f})")
