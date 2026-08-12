from scripts.configure_shared_access import (
    PREVIEW_DOMAIN,
    TARGET_DOMAIN,
    app_for_domain,
    configure_access,
)


class FakeClient:
    def __init__(self):
        self.apps = [
            {
                "id": "target-app",
                "name": "Denali shared seasonality",
                "domain": TARGET_DOMAIN,
            },
            {
                "id": "preview-app",
                "name": "denali-seasonality - Cloudflare Pages",
                "destinations": [{"uri": PREVIEW_DOMAIN}],
            },
        ]
        denali_team = {
            "id": "denali-team",
            "name": "Denali Team",
            "decision": "allow",
            "precedence": 1,
            "include": [
                {"email": {"email": "one@example.com"}},
                {"email": {"email": "two@example.com"}},
            ],
            "exclude": [],
            "require": [],
        }
        self.policies = {
            "target-app": [denali_team],
            "preview-app": [denali_team],
        }

    def list_apps(self):
        return self.apps

    def list_policies(self, app_id):
        return self.policies.get(app_id, [])

def test_app_lookup():
    apps = [
        {"id": "a", "domain": TARGET_DOMAIN},
        {"id": "b", "destinations": [{"uri": PREVIEW_DOMAIN}]},
    ]
    assert app_for_domain(apps, TARGET_DOMAIN)["id"] == "a"
    assert app_for_domain(apps, PREVIEW_DOMAIN)["id"] == "b"
    assert app_for_domain(apps, "other.pages.dev") is None


def test_configure_access_verifies_denali_team():
    client = FakeClient()
    result = configure_access(client)

    target = app_for_domain(client.apps, TARGET_DOMAIN)
    assert target is not None
    assert result["domains"] == [TARGET_DOMAIN, PREVIEW_DOMAIN]
    assert result["include_rule_count"] == 2
