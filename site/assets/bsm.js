/* bsm.js — pure European Black-Scholes-Merton pricing for the options workbench.

   Used for: re-pricing shootout legs at the planned exit date (with an IV-shift
   assumption), greeks fallback when IBKR modelGreeks are null on illiquid legs,
   and implied vol back-out from a mid. European with continuous dividend yield q
   — an APPROXIMATION (no early-exercise premium); consumers must label it.
   No DOM, no globals beyond the BSM namespace. */
"use strict";

const BSM = (() => {
  // Abramowitz-Stegun normal CDF (max abs error ~7.5e-8 — fine for display math)
  function normCdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327 * Math.exp(-x * x / 2);
    let p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - p : p;
  }
  function normPdf(x) { return 0.3989422804014327 * Math.exp(-x * x / 2); }

  function d1d2(S, K, T, sigma, r, q) {
    const v = sigma * Math.sqrt(T);
    const d1 = (Math.log(S / K) + (r - q + sigma * sigma / 2) * T) / v;
    return [d1, d1 - v];
  }

  /* price(S, K, T_years, sigma, r, q, right "C"|"P") -> option value.
     T <= 0 or sigma <= 0 -> intrinsic. */
  function price(S, K, T, sigma, r, q, right) {
    const call = String(right).toUpperCase() === "C";
    if (!(S > 0) || !(K > 0)) return null;
    if (!(T > 0) || !(sigma > 0)) {
      return Math.max(0, call ? S - K : K - S);
    }
    const [d1, d2] = d1d2(S, K, T, sigma, r, q);
    const dfq = Math.exp(-q * T), dfr = Math.exp(-r * T);
    return call
      ? S * dfq * normCdf(d1) - K * dfr * normCdf(d2)
      : K * dfr * normCdf(-d2) - S * dfq * normCdf(-d1);
  }

  /* greeks -> {delta, gamma, theta (per day), vega (per vol pt)} */
  function greeks(S, K, T, sigma, r, q, right) {
    const call = String(right).toUpperCase() === "C";
    if (!(S > 0) || !(K > 0) || !(T > 0) || !(sigma > 0)) {
      const itm = call ? S > K : S < K;
      return { delta: itm ? (call ? 1 : -1) : 0, gamma: 0, theta: 0, vega: 0 };
    }
    const [d1, d2] = d1d2(S, K, T, sigma, r, q);
    const dfq = Math.exp(-q * T), dfr = Math.exp(-r * T);
    const delta = call ? dfq * normCdf(d1) : -dfq * normCdf(-d1);
    const gamma = dfq * normPdf(d1) / (S * sigma * Math.sqrt(T));
    const vegaRaw = S * dfq * normPdf(d1) * Math.sqrt(T);
    const thetaYr = -(S * dfq * normPdf(d1) * sigma) / (2 * Math.sqrt(T))
      - (call ? r * K * dfr * normCdf(d2) - q * S * dfq * normCdf(d1)
              : -r * K * dfr * normCdf(-d2) + q * S * dfq * normCdf(-d1));
    return { delta, gamma, theta: thetaYr / 365, vega: vegaRaw / 100 };
  }

  /* impliedVol(mid, ...) — bisection; null when the mid is outside no-arb bounds. */
  function impliedVol(mid, S, K, T, r, q, right) {
    if (!(mid > 0) || !(T > 0)) return null;
    let lo = 0.005, hi = 5.0;
    if (price(S, K, T, lo, r, q, right) > mid) return null;
    if (price(S, K, T, hi, r, q, right) < mid) return null;
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2;
      if (price(S, K, T, m, r, q, right) > mid) hi = m; else lo = m;
    }
    return (lo + hi) / 2;
  }

  return { price, greeks, impliedVol, normCdf };
})();
