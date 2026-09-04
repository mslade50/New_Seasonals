"""Price-island repair + segment-aware basis guard (2026-09-04, D13).

SOXS carried 2026-03-19..2026-05-22 at ~15x in master_prices.parquet after
yfinance mis-applied its 2026-03-05 1:20 split; the updater's median overlap
test cannot see a 12-row segment inside the ~83-row refresh window, so the
island was re-imported nightly over the 2026-07-17 repair. Two pieces:
scripts/repair_price_island.py (generic island repair, mirror-derived factor)
and detect_broken_segments in scripts/update_master_prices.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.repair_price_island import (  # noqa: E402
    apply_island_repair, derive_mirror_factor, max_abs_return,
    mirror_residual, rank_stats, write_parquet_atomic, main as repair_main,
)
from scripts.update_master_prices import (  # noqa: E402
    detect_basis_changes, detect_broken_segments, drop_broken_segment_rows,
)

RNG = np.random.default_rng(7)


def _dates(n, start="2026-01-05"):
    return pd.date_range(start, periods=n, freq="B")


def _pair(n=120, island=(40, 60), factor=15.0):
    """(true SOXS-like closes, SOXL-like mirror closes, cache closes with island).

    Mirror returns are exactly the negated ticker returns plus small noise, so
    the mirror-derived factor is recoverable to well under 1%.
    """
    r = RNG.normal(0, 0.03, n)
    r[0] = 0.0
    close = 100.0 * np.cumprod(1.0 + r)
    r_m = -r + RNG.normal(0, 0.0005, n)
    r_m[0] = 0.0
    mirror = 50.0 * np.cumprod(1.0 + r_m)
    cache = close.copy()
    a, b = island
    cache[a:b + 1] *= factor
    return close, mirror, cache


def _frame(ticker, closes, dates, vol=1_000_000.0):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "Open": (closes * 0.99).astype("float32"),
        "High": (closes * 1.01).astype("float32"),
        "Low": (closes * 0.98).astype("float32"),
        "Close": closes.astype("float32"),
        "Volume": np.full(len(closes), vol, dtype="float64"),
    })


# ---------------------------------------------------------------- repair ----

def test_mirror_factor_recovers_planted_island():
    close, mirror, cache = _pair(factor=15.0)
    d = _dates(len(close))
    fac = derive_mirror_factor(pd.Series(cache, index=d), pd.Series(mirror, index=d), d[40], d[60])
    assert abs(fac["factor"] / 15.0 - 1.0) < 0.01
    assert abs(fac["factor_entry"] / 15.0 - 1.0) < 0.01
    assert abs(fac["factor_exit"] / 15.0 - 1.0) < 0.01
    assert fac["n_days"] == 21
    assert fac["entry_date"] == str(d[40].date()) and fac["exit_date"] == str(d[61].date())


def test_mirror_factor_is_one_without_island():
    close, mirror, _ = _pair(factor=1.0)
    d = _dates(len(close))
    fac = derive_mirror_factor(pd.Series(close, index=d), pd.Series(mirror, index=d), d[40], d[60])
    assert abs(fac["factor"] - 1.0) < 0.005


def test_mirror_factor_needs_anchors_on_both_sides():
    close, mirror, cache = _pair()
    d = _dates(len(close))
    with pytest.raises(ValueError):
        derive_mirror_factor(pd.Series(cache, index=d), pd.Series(mirror, index=d), d[0], d[10])
    with pytest.raises(ValueError):
        derive_mirror_factor(pd.Series(cache, index=d), pd.Series(mirror, index=d), d[100], d[-1])


def test_factor_mode_scales_only_the_island_rows():
    close, mirror, cache = _pair(factor=15.0)
    d = _dates(len(close))
    island_vol = np.full(len(close), 3_000_000.0)
    island_vol[40:61] = 200_000.0
    df = pd.concat([
        _frame("SOXS", cache, d).assign(Volume=island_vol),
        _frame("SOXL", mirror, d),
        _frame("SPY", np.linspace(500, 520, len(close)), d),
    ], ignore_index=True)
    out = apply_island_repair(df, "SOXS", d[40], d[60], 15.0)

    mask = (out["ticker"] == "SOXS") & (out["date"] >= d[40]) & (out["date"] <= d[60])
    assert mask.sum() == 21
    # every other row byte-identical
    assert out.loc[~mask].equals(df.loc[~mask])
    # OHLC divided, Volume multiplied, dtypes kept
    for c in ["Open", "High", "Low", "Close"]:
        np.testing.assert_allclose(out.loc[mask, c].to_numpy(), df.loc[mask, c].to_numpy() / 15.0, rtol=1e-6)
        assert out[c].dtype == df[c].dtype == np.float32
    np.testing.assert_allclose(out.loc[mask, "Volume"].to_numpy(), 3_000_000.0)
    assert out["Volume"].dtype == np.float64
    # the repaired closes match the true path
    np.testing.assert_allclose(out.loc[out["ticker"] == "SOXS", "Close"].to_numpy(), close, rtol=1e-5)
    # input frame untouched
    assert float(df.loc[mask, "Close"].iloc[0]) == pytest.approx(cache[40], rel=1e-6)


def test_factor_mode_rejects_bad_inputs():
    close, mirror, cache = _pair()
    d = _dates(len(close))
    df = _frame("SOXS", cache, d)
    with pytest.raises(ValueError):
        apply_island_repair(df, "SOXS", d[40], d[60], 0.0)
    with pytest.raises(ValueError):
        apply_island_repair(df, "NOPE", d[40], d[60], 15.0)


def test_window_stats_before_and_after():
    close, mirror, cache = _pair(factor=15.0)
    d = _dates(len(close))
    ct, cm = pd.Series(cache, index=d), pd.Series(mirror, index=d)
    before = max_abs_return(ct, d[40], d[60], 5)
    assert before["max_abs_ret"] > 10.0 and before["date"] == str(d[40].date())
    res_before = mirror_residual(ct, cm, d[40], d[60], 5)
    assert res_before["max"] > 10.0
    fixed = apply_island_repair(_frame("SOXS", cache, d), "SOXS", d[40], d[60], 15.0)
    ca = fixed.set_index("date")["Close"].astype(float)
    after = max_abs_return(ca, d[40], d[60], 5)
    assert after["max_abs_ret"] < 0.2
    res_after = mirror_residual(ca, cm, d[40], d[60], 5)
    assert res_after["max"] < 0.01
    assert res_after["median"] < 0.01


def test_rank_stats_reports_both_definitions():
    close, _, _ = _pair(n=600)
    st = rank_stats(pd.Series(close, index=_dates(600)))
    for k in ("prod_rank_126d", "prod_rank_252d", "recon_rank_126d", "recon_rank_252d", "sma200"):
        assert k in st and np.isfinite(st[k])
    assert 0.0 <= st["prod_rank_252d"] <= 100.0


def test_atomic_write_takes_backup_first_and_refuses_existing(tmp_path):
    close, mirror, cache = _pair()
    d = _dates(len(close))
    df = _frame("SOXS", cache, d)
    path = tmp_path / "master.parquet"
    df.to_parquet(path, index=False)
    backup = tmp_path / "master.parquet.bak"
    fixed = apply_island_repair(df, "SOXS", d[40], d[60], 15.0)
    write_parquet_atomic(fixed, str(path), str(backup))
    assert not (tmp_path / "master.parquet.tmp").exists()
    assert pd.read_parquet(backup).equals(df)
    assert pd.read_parquet(path).equals(fixed)
    with pytest.raises(FileExistsError):
        write_parquet_atomic(fixed, str(path), str(backup))


def test_cli_mirror_mode_end_to_end(tmp_path):
    close, mirror, cache = _pair(factor=15.0)
    d = _dates(len(close))
    df = pd.concat([_frame("SOXS", cache, d), _frame("SOXL", mirror, d),
                    _frame("SPY", np.linspace(500, 520, len(close)), d)], ignore_index=True)
    path = tmp_path / "master.parquet"
    df.to_parquet(path, index=False)
    backup = tmp_path / "master.parquet.bak_test"
    report = tmp_path / "report.json"
    args = ["--ticker", "SOXS", "--start", str(d[40].date()), "--end", str(d[60].date()),
            "--mirror", "SOXL", "--path", str(path), "--backup", str(backup), "--report", str(report)]
    assert repair_main(args + ["--dry-run"]) == 0
    assert not backup.exists()
    assert pd.read_parquet(path).equals(df)
    assert repair_main(args) == 0
    assert backup.exists()
    out = pd.read_parquet(path)
    soxs = out[out["ticker"] == "SOXS"].set_index("date")["Close"].astype(float)
    np.testing.assert_allclose(soxs.to_numpy(), close, rtol=0.01)
    assert out[out["ticker"] != "SOXS"].reset_index(drop=True).equals(
        df[df["ticker"] != "SOXS"].reset_index(drop=True))
    assert report.exists()
    # second run refuses to clobber the backup
    with pytest.raises(SystemExit):
        repair_main(args)


# ----------------------------------------------------------------- guard ----

def _overlap(master_closes, fresh_closes, ticker="SOXS", start="2026-05-07"):
    d = _dates(len(master_closes), start)
    m = pd.DataFrame({"ticker": ticker, "date": d, "Close": master_closes})
    f = pd.DataFrame({"ticker": ticker, "date": d[:len(fresh_closes)], "Close": fresh_closes})
    return m, f


def test_guard_rejects_12_row_15x_segment_the_median_cannot_see():
    n = 83
    base = 60.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[:12] *= 15.0  # island rows 05-07..05-22 still served inflated
    m, f = _overlap(base, fresh)
    assert detect_basis_changes(m, f) == []          # today's guard is blind
    broken = detect_broken_segments(m, f)
    assert list(broken) == ["SOXS"]
    seg = broken["SOXS"][0]
    assert seg["n"] == 12
    assert seg["start"] == m["date"].iloc[0] and seg["end"] == m["date"].iloc[11]
    assert abs(seg["max_ratio"] - 15.0) < 1e-6
    assert "12 consecutive" in seg["reason"]


def test_guard_accepts_uniform_dividend_shift():
    n = 83
    base = 100.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    m, f = _overlap(base, base * 0.985, ticker="SPY")   # 1.5% re-adjustment
    assert detect_basis_changes(m, f) == []
    assert detect_broken_segments(m, f) == {}


def test_guard_accepts_isolated_bad_bars():
    n = 83
    base = 100.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[10] *= 1.5
    fresh[30] *= 0.5
    fresh[50] *= 1.1
    m, f = _overlap(base, fresh, ticker="XLE")
    assert detect_broken_segments(m, f) == {}


def test_guard_rejects_short_island_tail_bounded_by_novel_cliff():
    # 2026-09-16: the fetch window starts 05-19, only 4 island rows remain
    # inside it, followed by the vendor's -94% cliff on 05-26 that the
    # repaired cache does not have.
    n = 80
    base = 60.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[:4] *= 15.0
    m, f = _overlap(base, fresh, start="2026-05-19")
    broken = detect_broken_segments(m, f)
    assert list(broken) == ["SOXS"]
    seg = broken["SOXS"][0]
    assert seg["n"] == 4 and "novel >50% cliff" in seg["reason"]
    # ...and a 4-row divergence NOT bounded by a cliff (mild, 3%) is tolerated
    fresh2 = base.copy()
    fresh2[:4] *= 1.03
    m2, f2 = _overlap(base, fresh2, start="2026-05-19")
    assert detect_broken_segments(m2, f2) == {}
    # ...and the same 4-row 15x blip in MID-window (not at the edge) is below
    # the 5-session rule and merges as today - the edge rule is deliberately
    # narrow so an isolated bogus bar never blanks a ticker for a night.
    fresh3 = base.copy()
    fresh3[20:24] *= 15.0
    m3, f3 = _overlap(base, fresh3, start="2026-05-19")
    assert detect_broken_segments(m3, f3) == {}
    # a single 1-row tail (2026-09-18) is still caught
    fresh4 = base.copy()
    fresh4[:1] *= 15.0
    m4, f4 = _overlap(base, fresh4, start="2026-05-22")
    assert detect_broken_segments(m4, f4)["SOXS"][0]["n"] == 1


def test_guard_leaves_median_flagged_tickers_to_the_basis_path():
    n = 83
    base = 80.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    m, f = _overlap(base, base / 4.0, ticker="TQQQ")   # clean 1:4 reverse split
    flagged = detect_basis_changes(m, f)
    assert flagged == ["TQQQ"]
    assert detect_broken_segments(m, f, exclude=flagged) == {}


def _with_new_bars(fresh_frame, n_new):
    last = fresh_frame["date"].max()
    extra = pd.DataFrame({"ticker": fresh_frame["ticker"].iloc[0],
                          "date": pd.bdate_range(last, periods=n_new + 1)[1:],
                          "Close": np.full(n_new, float(fresh_frame["Close"].iloc[-1]))})
    return pd.concat([fresh_frame, extra], ignore_index=True)


def test_segment_only_drop_keeps_agreeing_rows_and_new_bars():
    # 12-row island in an 83-row overlap plus 2 brand-new vendor bars:
    # exactly the 12 drop, the other 71 overlap rows AND the 2 new bars merge.
    n = 83
    base = 60.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[:12] *= 15.0
    m, f = _overlap(base, fresh)
    f = _with_new_bars(f, 2)
    assert len(f) == 85
    broken = detect_broken_segments(m, f)
    kept, stats = drop_broken_segment_rows(f, broken)
    assert stats == {"SOXS": {"dropped": 12, "merged": 73}}
    assert len(kept) == 73
    seg = broken["SOXS"][0]
    assert not ((kept["date"] >= seg["start"]) & (kept["date"] <= seg["end"])).any()
    assert kept["date"].max() == f["date"].max()          # new bars survive
    assert (kept["date"] > m["date"].max()).sum() == 2
    # the merged overlap rows are the agreeing ones (ratio 1.0)
    j = kept.merge(m, on=["ticker", "date"], suffixes=("_new", "_old"))
    assert len(j) == 71 and np.allclose(j["Close_new"], j["Close_old"])
    # no broken segments -> frame returned untouched
    same, none = drop_broken_segment_rows(f, {})
    assert same is f and none == {}


def test_segment_only_drop_window_edge_tail():
    # 2026-09-16: 4 island rows at the window edge then the novel cliff;
    # only those 4 drop, the remaining overlap rows and the new bar merge.
    n = 80
    base = 60.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[:4] *= 15.0
    m, f = _overlap(base, fresh, start="2026-05-19")
    f = _with_new_bars(f, 1)
    broken = detect_broken_segments(m, f)
    assert broken["SOXS"][0]["n"] == 4
    kept, stats = drop_broken_segment_rows(f, broken)
    assert stats == {"SOXS": {"dropped": 4, "merged": 77}}
    assert kept["date"].min() == m["date"].iloc[4]
    assert kept["date"].max() > m["date"].max()


def test_segment_only_drop_touches_only_the_named_ticker():
    n = 83
    base = 60.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[:12] *= 15.0
    m1, f1 = _overlap(base, fresh, ticker="SOXS")
    m2, f2 = _overlap(base, base, ticker="SOXL")
    m, f = pd.concat([m1, m2], ignore_index=True), pd.concat([f1, f2], ignore_index=True)
    kept, stats = drop_broken_segment_rows(f, detect_broken_segments(m, f))
    assert stats == {"SOXS": {"dropped": 12, "merged": 71}}
    assert kept[kept["ticker"] == "SOXL"].reset_index(drop=True).equals(f2)


# ---- verifier round-2 battery (artifacts/verify_2026-09-04/soxs_repair/02_guard_cases.py) ----
# Two tolerances: the LENGTH rule (5+ sessions) fires beyond 10% so a >2%
# dividend going ex mid-window (a genuine partial re-adjustment) is merged;
# the WINDOW-EDGE rule keeps 2%.

_SESS = pd.bdate_range("2026-05-07", periods=83)
_HIST = pd.bdate_range("2024-01-02", "2026-05-06")


def _vbuild(ticker, seed, n_overlap=83):
    sess = _SESS[:n_overlap]
    all_dates = _HIST.append(sess)
    r = np.random.default_rng(seed)
    c = 100.0 * np.cumprod(1 + r.normal(0, 0.01, len(all_dates)))
    master = pd.DataFrame({"ticker": ticker, "date": all_dates, "Close": c})
    new = pd.DataFrame({"ticker": ticker, "date": sess, "Close": c[-len(sess):]})
    return master, new, sess


def _vrun(master, new):
    basis = detect_basis_changes(master, new)
    broken = detect_broken_segments(master, new, exclude=basis)
    kept, stats = drop_broken_segment_rows(new, broken)
    return basis, broken, len(new) - len(kept)


def test_v_c7_soxs_shaped_12row_island_rejected():
    m, n, sess = _vbuild("ISL", 8)
    n.loc[n["date"].isin(sess[:12]), "Close"] *= 15.0139
    basis, broken, dropped = _vrun(m, n)
    assert basis == [] and dropped == 12
    assert broken["ISL"][0]["n"] == 12 and abs(broken["ISL"][0]["max_ratio"] - 15.0139) < 1e-6


def test_v_c4_two_segments_one_ticker_both_dropped():
    m, n, sess = _vbuild("TWOSEG", 5)
    n.loc[n["date"].isin(sess[5:13]), "Close"] *= 15.0
    n.loc[n["date"].isin(sess[40:47]), "Close"] *= 0.5
    basis, broken, dropped = _vrun(m, n)
    assert basis == [] and dropped == 15
    assert [s["n"] for s in broken["TWOSEG"]] == [8, 7]
    assert broken["TWOSEG"][0]["start"] == sess[5] and broken["TWOSEG"][1]["end"] == sess[46]


def test_v_c5a_4row_edge_tail_with_novel_cliff_rejected():
    m, n, sess = _vbuild("EDGE4", 6)
    n.loc[n["date"].isin(sess[:4]), "Close"] *= 15.0
    basis, broken, dropped = _vrun(m, n)
    assert basis == [] and dropped == 4
    assert broken["EDGE4"][0]["n"] == 4 and "window edge" in broken["EDGE4"][0]["reason"]


def test_v_c2b_5row_1p5x_blip_rejected_at_min_run():
    m, n, sess = _vbuild("BLIP5", 2)
    n.loc[n["date"].isin(sess[40:45]), "Close"] *= 1.5
    basis, broken, dropped = _vrun(m, n)
    assert basis == [] and dropped == 5 and broken["BLIP5"][0]["n"] == 5


@pytest.mark.parametrize("div,k", [(0.025, 25), (0.025, 45), (0.05, 20)])
def test_v_c3c_exdividend_partial_readjustment_accepted(div, k):
    # yfinance rescales only the block BEFORE the ex-date: a 2.5-5% step over
    # 20-45 sessions is genuine and must merge (or go to the basis re-pull
    # path when the block is the majority), never be dropped as a segment.
    m, n, sess = _vbuild(f"EXDIV{k}", 4)
    n.loc[n["date"].isin(sess[:k]), "Close"] *= (1 - div)
    basis, broken, dropped = _vrun(m, n)
    assert broken == {} and dropped == 0


def test_v_c5b_edge_run_without_cliff_merges():
    m, n, sess = _vbuild("EDGE4NC", 6)
    n.loc[n["date"].isin(sess[:4]), "Close"] *= 1.10
    assert _vrun(m, n)[1] == {}


def test_v_c1_uniform_split_left_to_basis_path():
    m, n, sess = _vbuild("SPLIT", 1)
    n["Close"] /= 2.0
    basis, broken, dropped = _vrun(m, n)
    assert basis == ["SPLIT"] and broken == {} and dropped == 0


# ---- main() on the production path with yfinance + cache_io stubbed ----
# (mirrors artifacts/verify_2026-09-04/soxs_repair/03_prod_path.py)

def _run_main(monkeypatch, tmp_path, argv):
    import sys as _sys
    import types
    import scripts.update_master_prices as U

    dates = pd.bdate_range("2024-01-02", "2026-09-03")
    r = np.random.default_rng(0)
    rows = []
    for t in ["T00", "T01", "T05"]:
        d = dates[dates <= pd.Timestamp("2026-01-02")] if t == "T05" else dates   # stale name -> 120d floor
        c = 100 * np.cumprod(1 + r.normal(0, 0.01, len(d)))
        rows.append(pd.DataFrame({"ticker": t, "date": d, "Open": c, "High": c * 1.01,
                                  "Low": c * 0.99, "Close": c, "Volume": 1e6}))
    master = pd.concat(rows, ignore_index=True)
    cache = tmp_path / "synthetic_master.parquet"
    master.to_parquet(cache, index=False)
    isl = pd.bdate_range("2026-05-07", periods=12)

    def fake_download(tickers, start=None, **kw):
        tk = list(tickers) if not isinstance(tickers, str) else [tickers]
        sub = master[master["date"] >= pd.Timestamp(start)]
        frames = {}
        for t in tk:
            d = sub[sub["ticker"] == t].set_index("date")[["Open", "High", "Low", "Close", "Volume"]].copy()
            if d.empty:
                continue
            if t == "T00":
                mm = d.index.isin(isl)
                for c in ["Open", "High", "Low", "Close"]:
                    d.loc[mm, c] *= 15.0
            frames[t] = d
        out = pd.concat(frames, axis=1)
        out.columns = pd.MultiIndex.from_tuples(list(out.columns))
        return out

    calls = []
    fake = types.ModuleType("cache_io")
    fake.upload_from_local = lambda local, key: calls.append((local, key)) or True
    fake.head = lambda key: {"ContentLength": 0}
    monkeypatch.setitem(_sys.modules, "cache_io", fake)
    monkeypatch.setattr(U, "yf", types.SimpleNamespace(download=fake_download))
    monkeypatch.setattr(U, "PATH", str(cache))
    monkeypatch.setattr(U, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(_sys, "argv", ["update_master_prices.py"] + argv)
    mtime = os.path.getmtime(cache)
    rc = U.main()
    return rc, calls, cache, mtime, master


def test_main_only_tickers_never_uploads_and_out_leaves_canonical_alone(tmp_path, monkeypatch, capsys):
    # --only-tickers ALONE: no upload call, canonical (temp) path rewritten
    rc, calls, cache, mtime, master = _run_main(monkeypatch, tmp_path, ["--only-tickers", "T00"])
    out = capsys.readouterr().out
    assert rc == 0 and calls == []
    assert "--only-tickers implies --no-upload" in out and "[r2 upload] skipped (--no-upload)" in out
    assert "[SEGMENT] T00:" in out and "rows dropped 12" in out
    assert os.path.getmtime(cache) >= mtime and len(pd.read_parquet(cache)) >= len(master)
    # --only-tickers with --out: canonical path byte-untouched, --out written
    outp = tmp_path / "elsewhere.parquet"
    rc, calls, cache, mtime, master = _run_main(monkeypatch, tmp_path, ["--only-tickers", "T00", "--out", str(outp)])
    assert rc == 0 and calls == [] and outp.exists()
    assert pd.read_parquet(cache).equals(master)


def test_main_no_flags_uploads_via_cache_io(tmp_path, monkeypatch, capsys):
    # the production invocation still pushes (through the stub), so the
    # no-upload paths above are real guards, not a broken upload
    rc, calls, cache, mtime, master = _run_main(monkeypatch, tmp_path, [])
    out = capsys.readouterr().out
    assert rc == 0 and len(calls) == 1 and calls[0][1] == "master_prices.parquet"
    assert "[SEGMENT] T00:" in out


def test_guard_multi_ticker_only_broken_named():
    n = 83
    base = 60.0 + RNG.normal(0, 1.0, n).cumsum() * 0.1
    fresh = base.copy()
    fresh[:12] *= 15.0
    m1, f1 = _overlap(base, fresh, ticker="SOXS")
    m2, f2 = _overlap(base, base * 0.99, ticker="SOXL")
    broken = detect_broken_segments(pd.concat([m1, m2]), pd.concat([f1, f2]))
    assert list(broken) == ["SOXS"]
