"""Create a local immutable review package; never install or deploy it.

The package records the clean source revision, changed-file digests, reviewed
external-source hashes and candidate hashes. External candidates retain private
configuration and must stay under ignored artifacts, outside the deployed path.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def git(*args):
    return subprocess.check_output(["git","-c",f"safe.directory={ROOT.as_posix()}",*args],cwd=ROOT)


def prepare(output, executor_source, baseline):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        raise ValueError("release output must be a new directory under this worktree's artifacts")
    if git("status","--porcelain","--untracked-files=normal").strip():
        raise ValueError("source must be committed and clean before preparing a release")
    head=git("rev-parse","HEAD").decode().strip()
    base=git("rev-parse",baseline).decode().strip()
    # Git blobs, not mutable checkout bytes, define the release. Renames are
    # recorded as delete+add so the manifest also captures removed paths.
    changes=git("diff","--name-status","--no-renames","-z",base,head).decode().rstrip("\0").split("\0")
    changed={}
    for status,name in zip(changes[::2],changes[1::2]):
        changed[name]={"status":status,"sha256":None if status=="D" else
                       hashlib.sha256(git("show",f"{head}:{name}")).hexdigest()}
    source=output / "reviewed-source"
    helpers=git("ls-tree","-r","--name-only","-z",head,"broker_runtime").decode().rstrip("\0").split("\0")
    for name in helpers:
        if not name or not name.startswith("broker_runtime/") or ".." in Path(name).parts:
            raise ValueError("invalid executor preparation path")
        target=source/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(git("show",f"{head}:{name}"))
    # Run the exact committed preparer against its immutable helper snapshot.
    # It verifies every external source before writing any candidate.
    prepared=subprocess.run([sys.executable,str(source/"broker_runtime/prepare.py"),
        "--source",str(Path(executor_source).resolve()),"--output",str(output/"executor-candidate")],
        cwd=source,capture_output=True,text=True,check=False)
    if prepared.returncode:
        raise RuntimeError("committed executor preparation failed")
    candidates={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((output / "executor-candidate").iterdir()) if path.is_file()}
    original=json.loads((source / "broker_runtime/source_hashes.json").read_text())
    if git("rev-parse","HEAD").decode().strip()!=head or git("status","--porcelain","--untracked-files=normal").strip():
        raise RuntimeError("source changed during preparation; no release manifest published")
    manifest={"schema_version":1,"prepared_at":dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_sha":head,"audit_baseline_sha":base,"source_changes":changed,
              "external_original_sha256":original,"external_candidate_sha256":candidates,
              "production_changed":False,
              "promotion_requires":["reviewed Primary inventory seed and continuous fill history",
                 "paper/TWS verification of native bracket and owner-client behavior",
                 "one pinned runtime/executor/broker/site release with captured rollback state",
                 "explicit approval before financially consequential activation"],
              "site_build":"GitHub Actions deploy_site.yml from canonical R2 only",
              "rollback":"Restore captured prior source/runtime/executor versions; reconcile outstanding orders before any retry. Code rollback does not undo trades."}
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--executor-source",type=Path,required=True)
    parser.add_argument("--baseline",default="e53c478e232bc37016baed6c285fb8f3b6c6688b")
    args=parser.parse_args(argv)
    try:
        manifest=prepare(args.output,args.executor_source,args.baseline)
        print(f"Release prepared at {args.output}: {manifest['source_sha']}; production unchanged")
        return 0
    except Exception as exc:
        print(f"Release preparation failed ({type(exc).__name__}); no activation performed",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
