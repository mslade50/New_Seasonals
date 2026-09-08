"""Explicit, backed-up removal of the obsolete preview-only Pages allow rule.

Ordinary deployment verification remains read-only. This repair requires --apply
and never deletes the reusable Denali Team policy or changes production access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.configure_shared_access import (
    CloudflareAccessClient, PREVIEW_DOMAIN, TARGET_DOMAIN,
    app_for_domain, configure_access, verify_domain,
)

LEGACY_NAME = "Allow Members - Cloudflare Pages"


def private_backup(payload):
    from cache_io import _client, _r2_creds
    client, creds = _client(), _r2_creds()
    if client is None or creds is None:
        raise RuntimeError("Private R2 backup storage is required before repair")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    key = f"operations/access-policy-backups/shared-preview/{digest}.json"
    try:
        client.put_object(Bucket=creds["R2_BUCKET"], Key=key, Body=body,
                          ContentType="application/json", IfNoneMatch="*")
    except Exception:
        # An identical prior backup is acceptable; anything else fails closed.
        existing = client.get_object(Bucket=creds["R2_BUCKET"], Key=key)["Body"].read()
        if existing != body:
            raise RuntimeError("Existing access backup differs")
    saved = client.get_object(Bucket=creds["R2_BUCKET"], Key=key)["Body"].read()
    if saved != body:
        raise RuntimeError("Access backup verification failed")
    return key


def repair(client, *, apply=False, backup=private_backup):
    apps = client.list_apps()
    verify_domain(client, apps, TARGET_DOMAIN)
    previews = [app for app in apps if app_for_domain([app], PREVIEW_DOMAIN)]
    if len(previews) != 1:
        raise RuntimeError("Expected exactly one shared preview application")
    preview = previews[0]
    policies = client.list_policies(preview["id"])
    legacy = [p for p in policies if p.get("name") == LEGACY_NAME]
    if not legacy:
        configure_access(client)
        return {"status": "already_correct", "domain": PREVIEW_DOMAIN}
    if len(legacy) != 1 or legacy[0].get("decision") != "allow" or not legacy[0].get("id"):
        raise RuntimeError("Obsolete preview rule does not match the approved target")
    target = legacy[0]

    class RemainingPolicies:
        def list_policies(self, app_id):
            return [p for p in policies if p.get("id") != target["id"]]

    verify_domain(RemainingPolicies(), apps, PREVIEW_DOMAIN)
    result = {"status": "planned", "domain": PREVIEW_DOMAIN, "policy": LEGACY_NAME}
    if not apply:
        return result
    result["backup_key"] = backup({
        "schema_version": "access-policy-backup.v1", "account_id": client.account_id,
        "app_id": preview["id"], "domain": PREVIEW_DOMAIN, "policy": target,
        "restore_method": "POST", "restore_path":
            f"/accounts/{client.account_id}/access/apps/{preview['id']}/policies",
    })
    # Do not delete a same-ID rule that an operator changed after the review.
    current = client.list_policies(preview["id"])
    if current != policies:
        raise RuntimeError("Preview policies changed during repair; nothing deleted")
    client.request("DELETE", f"/accounts/{client.account_id}/access/apps/{preview['id']}/policies/{target['id']}")
    configure_access(client)
    return dict(result, status="repaired")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    client = CloudflareAccessClient(os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"])
    print(json.dumps(repair(client, apply=args.apply), sort_keys=True))


if __name__ == "__main__":
    main()
