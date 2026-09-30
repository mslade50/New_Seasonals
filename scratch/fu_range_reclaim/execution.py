"""Research execution reused from FU v1; no portfolio sizing or live integration."""
import numpy as np
import pandas as pd


def simulate(df, signal, horizon=10, stop=False, cost_bps=10):
    i = signal["signal_i"]
    entry_i, final_i = i+1, i+horizon
    if final_i >= len(df):
        return None
    entry = float(df.Open.iloc[entry_i])
    stop_level = signal["sweep_low"]-0.1*signal["atr_ref"]
    if stop and entry <= stop_level:
        return None
    exit_i, price, reason = final_i, float(df.Close.iloc[final_i]), "time"
    if stop:
        for j in range(entry_i, final_i+1):
            if df.Open.iloc[j] <= stop_level:
                exit_i, price, reason = j, float(df.Open.iloc[j]), "gap_stop"
                break
            if df.Low.iloc[j] <= stop_level:
                exit_i, price, reason = j, stop_level, "stop"
                break
    gross = price/entry-1
    result = dict(signal_date=df.index[i], entry_date=df.index[entry_i],
                  exit_date=df.index[exit_i], entry_i=entry_i, exit_i=exit_i,
                  entry=entry, exit=price, stop_level=stop_level, reason=reason,
                  gross=gross, net=gross-cost_bps/10000,
                  r_net=(price-entry-entry*cost_bps/10000)/(entry-stop_level) if stop else np.nan)
    return result


def evaluate(df, signals, horizon=10, stop=False):
    rows, last_exit = [], -1
    for s in signals:
        # Do not overlap positions within ticker/variant/exit policy.
        if s["signal_i"]+1 <= last_exit:
            continue
        trade = simulate(df, s, horizon, stop)
        if trade:
            rows.append({**s, **trade})
            last_exit = trade["exit_i"]
    return rows


def block_ci(frame, column, repetitions=2000):
    x = frame[["signal_date", column]].dropna()
    if x.empty:
        return [np.nan, np.nan]
    g = x.groupby(x.signal_date.dt.to_period("M"))[column].agg(["sum", "count"])
    if len(g) < 8:
        return [np.nan, np.nan]
    rng = np.random.default_rng(9202026)
    idx = rng.integers(0, len(g), size=(repetitions, len(g)))
    values = g["sum"].to_numpy()[idx].sum(axis=1)/g["count"].to_numpy()[idx].sum(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()
