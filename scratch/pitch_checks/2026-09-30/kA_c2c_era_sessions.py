"""C2 round 2b (ungated parent): (a) which post-ME session carries the excess,
BY ERA (the registry 1765-1792 moving-part test); excess = session mean minus
the era's all-days one-session mean; (b) is the run-in dose ME-specific or just
generic 5d TLT reversal (same regression on all days); (c) ISM-services day
check: ME+3 is the 3rd business day."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa
from kA_c2b_sessions import session_table  # noqa

ERAS = (("2002", "2013"), ("2013", "2020"), ("2020", "2027"))

if __name__ == "__main__":
    tlt = own_series("TLT")
    nfp = set(nfp_dates())
    me = month_ends(tlt.index)
    base1 = -(tlt.pct_change().dropna())
    S = session_table(tlt, me, nfp, K=7)
    out = []
    for lo, hi in ERAS:
        b = base1[(base1.index >= lo) & (base1.index < hi)].mean()
        s = S[(S.me >= lo) & (S.me < hi)]
        row = {"era": f"{lo}-{int(hi)-1}", "n_me": s.me.nunique(), "base_bp": round(1e4 * b, 2)}
        tot = 0.0
        for k in range(1, 8):
            x = s[s.k == k].r
            ex = 1e4 * (x.mean() - b)
            row[f"ME+{k}"] = f"{ex:+.1f} (t{(x.mean()-b)/(x.std()/np.sqrt(len(x))):+.1f})"
            if k <= 5:
                tot += ex
        row["sum_1_5_bp"] = round(tot, 1)
        out.append(row)
    print("\n=== excess SHORT bp per post-ME session vs era all-days (t) ===")
    print(pd.DataFrame(out).to_string(index=False))

    # same, ex-NFP sessions only
    out = []
    for lo, hi in ERAS:
        b = base1[(base1.index >= lo) & (base1.index < hi)].mean()
        s = S[(S.me >= lo) & (S.me < hi) & (~S.nfp)]
        row = {"era": f"{lo}-{int(hi)-1}"}
        for k in range(1, 8):
            x = s[s.k == k].r
            row[f"ME+{k}"] = f"{1e4*(x.mean()-b):+.1f} (t{(x.mean()-b)/(x.std()/np.sqrt(len(x))):+.1f})"
        out.append(row)
    print("\n=== same, NON-NFP sessions only ===")
    print(pd.DataFrame(out).to_string(index=False))

    # (b) generic 5d reversal control
    r5 = tlt / tlt.shift(5) - 1
    f5 = -(tlt.shift(-5) / tlt - 1)
    df = pd.DataFrame({"run": r5, "fw": f5}).dropna()
    b_all = np.polyfit(df.run, df.fw, 1)[0]
    is_me = df.index.isin(me)
    b_me = np.polyfit(df.run[is_me], df.fw[is_me], 1)[0]
    b_non = np.polyfit(df.run[~is_me], df.fw[~is_me], 1)[0]
    print(f"\nDOSE control: slope of fwd-5d SHORT on trailing-5d TLT return: all days {b_all:+.4f}, "
          f"non-ME days {b_non:+.4f}, ME days {b_me:+.4f}")
    q_hi = df.run.quantile(2 / 3)
    hi_non = df.fw[(~is_me) & (df.run >= q_hi)]
    hi_me = df.fw[is_me & (df.run >= q_hi)]
    print(f"top-tercile run-in (>= {100*q_hi:.2f}%): non-ME days fwd short {100*hi_non.mean():+.3f}% "
          f"(n {len(hi_non)}), ME days {100*hi_me.mean():+.3f}% (n {len(hi_me)})")
    lo_me = df.fw[is_me & (df.run < q_hi)]
    lo_non = df.fw[(~is_me) & (df.run < q_hi)]
    print(f"rest: non-ME {100*lo_non.mean():+.3f}%  ME {100*lo_me.mean():+.3f}%  -> ME premium "
          f"hi {100*(hi_me.mean()-hi_non.mean()):+.3f}pp, rest {100*(lo_me.mean()-lo_non.mean()):+.3f}pp")
