"""Verify the shared seasonality domain is protected by Denali Team.

The reusable policy is managed in Cloudflare Zero Trust and attached to the
application there. This pre-deploy check is intentionally read-only: a missing,
empty, or wrong policy fails closed before any site files are published.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request


API_ROOT = "https://api.cloudflare.com/client/v4"
TARGET_DOMAIN = "denali-seasonality.pages.dev"
PREVIEW_DOMAIN = "*.denali-seasonality.pages.dev"
TARGET_APP_NAME = "Denali shared seasonality"
TARGET_POLICY_NAME = "Denali Team"


class CloudflareAccessClient:
    def __init__(self, account_id: str, token: str):
        self.account_id = account_id
        self.token = token

    def request(self, method: str, path: str, payload: dict | None = None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{API_ROOT}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8"))
                messages = detail.get("errors") or detail.get("messages") or []
            except Exception:
                messages = []
            safe_detail = "; ".join(
                str(item.get("message", item)) if isinstance(item, dict) else str(item)
                for item in messages
            )
            raise RuntimeError(
                f"Cloudflare Access API returned HTTP {error.code}"
                + (f": {safe_detail}" if safe_detail else "")
            ) from error
        if not result.get("success"):
            raise RuntimeError("Cloudflare Access API reported an unsuccessful response")
        return result.get("result")

    def list_apps(self) -> list[dict]:
        account = urllib.parse.quote(self.account_id, safe="")
        return self.request("GET", f"/accounts/{account}/access/apps?per_page=100") or []

    def list_policies(self, app_id: str) -> list[dict]:
        account = urllib.parse.quote(self.account_id, safe="")
        app = urllib.parse.quote(app_id, safe="")
        return self.request(
            "GET", f"/accounts/{account}/access/apps/{app}/policies?per_page=100"
        ) or []

def app_for_domain(apps: list[dict], domain: str) -> dict | None:
    wanted = domain.lower().rstrip("/")
    for app in apps:
        candidates = [app.get("domain")]
        for item in app.get("self_hosted_domains") or []:
            candidates.append(item.get("hostname") if isinstance(item, dict) else item)
        for item in app.get("destinations") or []:
            if isinstance(item, dict):
                candidates.append(item.get("uri") or item.get("hostname"))
        for candidate in candidates:
            normalized = str(candidate or "").lower().rstrip("/")
            normalized = normalized.removeprefix("https://").removeprefix("http://")
            if normalized == wanted:
                return app
    return None


def verify_domain(client: CloudflareAccessClient, apps: list[dict], domain: str) -> dict:
    target = app_for_domain(apps, domain)
    if target is None:
        raise ValueError(f"Access application was not found for {domain}")
    verified_allow = next(
        (
            policy
            for policy in client.list_policies(target["id"])
            if policy.get("decision") == "allow"
            and str(policy.get("name", "")).lower() == TARGET_POLICY_NAME.lower()
        ),
        None,
    )
    if verified_allow is None or not verified_allow.get("include"):
        raise RuntimeError(
            f"{domain} is not protected by a populated {TARGET_POLICY_NAME} allow policy"
        )
    return {
        "app": target.get("name") or TARGET_APP_NAME,
        "domain": domain,
        "include_rule_count": len(verified_allow["include"]),
    }


def configure_access(client: CloudflareAccessClient) -> dict:
    apps = client.list_apps()
    verified = [
        verify_domain(client, apps, domain)
        for domain in (TARGET_DOMAIN, PREVIEW_DOMAIN)
    ]
    return {
        "domains": [item["domain"] for item in verified],
        "include_rule_count": min(item["include_rule_count"] for item in verified),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", default=os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
    args = parser.parse_args()
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not args.account_id or not token:
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")

    result = configure_access(CloudflareAccessClient(args.account_id, token))
    print(
        f"Verified {', '.join(result['domains'])} are protected by Denali Team "
        f"({result['include_rule_count']} include rules)."
    )


if __name__ == "__main__":
    main()
