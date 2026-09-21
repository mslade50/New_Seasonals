"""Prepare only the four reconciliation modules from a verified live baseline."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("broker_reconciliation.py", "position_actions.py", "position_action_agent.py", "manual_order_actions.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    expected = json.loads((HERE / "reconciliation_source_hashes.json").read_text())
    for name, checksum in expected.items():
        if digest(source / name) != checksum:
            raise ValueError("Runtime changed since review: " + name)
    if (source / "broker_reconciliation.py").exists():
        raise ValueError("A reconciliation module already exists; review it before installing")
    if output.exists():
        raise ValueError("Candidate directory must be new")
    contents = {name: (HERE / name).read_bytes() for name in MODULES}
    for name, content in contents.items():
        compile(content, name, "exec")
    output.mkdir(parents=True)
    for name, content in contents.items():
        (output / name).write_bytes(content)
    manifest = dict(source=expected, candidate={name: digest(output / name) for name in MODULES})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
