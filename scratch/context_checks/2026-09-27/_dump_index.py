import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
d = json.loads((ROOT / "data/context_state.json").read_text())
for g in d["cells_index"]:
    print(f"\n== {g['trigger_id']} | {g['lane']} | {g['cell']} | {g.get('anchor_rule','')}")
    extra = {k: v for k, v in g.items() if k not in ("trigger_id", "lane", "cell", "anchor_rule", "subjects")}
    if extra:
        print("   ", json.dumps(extra)[:400])
    for s in g["subjects"]:
        keys = [k for k in s if k not in ("subject", "fp")]
        core = ("tag_hint", "n", "h1_mean_pct", "h1_hit", "h1_t", "h1_edge_pct", "h1_record",
                "h1_sign_p", "h5_mean_pct", "era_stable", "bh_pass")
        line = " ".join(f"{k}={s.get(k)}" for k in core if k in s)
        rest = {k: s[k] for k in keys if k not in core}
        print(f"  {s['fp']:<44} {line} {json.dumps(rest) if rest else ''}")

print("\n== novelty")
nov = d["novelty"]
print(json.dumps({k: v for k, v in nov.items() if k != "flags"})[:800])
flags = nov.get("flags", {})
for fp, f in flags.items():
    if f.get("repeat_blocked") or f.get("delta_suppressed") or f.get("last_published"):
        print(" ", fp, json.dumps(f))
