"""Disable or verify Cloudflare Pages' automatic Git deployments.

The private site is deployed by GitHub Actions after R2 provenance checks.
Cloudflare's independent Git integration must therefore have production and
preview auto-deployments disabled, otherwise a normal data commit can bypass
the cloud-only pipeline and publish the repository checkout directly.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Callable


PROJECT_NAME = "seasonals-mslade"


def _request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Cloudflare API {method} failed ({exc.code}): {detail}") from exc
    if not payload.get("success"):
        raise RuntimeError(f"Cloudflare API {method} failed: {payload.get('errors')}")
    return payload["result"]


def _source_config(project: dict) -> dict:
    return ((project.get("source") or {}).get("config") or {})


def git_deployments_disabled(project: dict) -> bool:
    # A direct-upload Pages project has no Git source at all. That is already
    # the strongest possible lock for this workflow: Cloudflare has no branch
    # integration that could bypass the GitHub Actions + R2 deployment path.
    # Wrangler reports this state as ``Git Provider: No`` and the API returns
    # a missing/null source, so treat it as safely disabled instead of trying
    # to attach a GitHub source merely to turn that source off.
    if not project.get("source"):
        return True
    config = _source_config(project)
    return (
        config.get("production_deployments_enabled") is False
        and config.get("preview_deployment_setting") == "none"
    )


def enforce_disabled(
    account_id: str,
    token: str,
    *,
    request_fn: Callable[[str, str, str, dict | None], dict] = _request,
) -> dict:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects/{PROJECT_NAME}"
    current = request_fn("GET", url, token, None)
    if not git_deployments_disabled(current):
        source_type = (current.get("source") or {}).get("type") or "github"
        request_fn(
            "PATCH",
            url,
            token,
            {
                "source": {
                    "type": source_type,
                    "config": {
                        "deployments_enabled": False,
                        "production_deployments_enabled": False,
                        "preview_deployment_setting": "none",
                    },
                }
            },
        )
    verified = request_fn("GET", url, token, None)
    if not git_deployments_disabled(verified):
        raise RuntimeError("Cloudflare Git auto-deployments are still enabled")
    return verified


def check_disabled(
    account_id: str,
    token: str,
    *,
    request_fn: Callable[[str, str, str, dict | None], dict] = _request,
) -> dict:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects/{PROJECT_NAME}"
    project = request_fn("GET", url, token, None)
    if not git_deployments_disabled(project):
        raise RuntimeError(
            "Cloudflare Git auto-deployments are enabled; refusing the official deploy "
            "until branch control is locked"
        )
    return project


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("disable", "check"))
    args = parser.parse_args()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id or not token:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")
    if args.command == "disable":
        enforce_disabled(account_id, token)
        print("Cloudflare Pages Git production and preview auto-deployments are disabled.")
    else:
        check_disabled(account_id, token)
        print("Cloudflare Pages Git auto-deployment guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
