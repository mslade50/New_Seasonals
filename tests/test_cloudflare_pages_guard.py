import pytest

from scripts.cloudflare_pages_guard import check_disabled, enforce_disabled


def _project(production=True, preview="all"):
    return {
        "source": {
            "type": "github",
            "config": {
                "production_deployments_enabled": production,
                "preview_deployment_setting": preview,
            },
        }
    }


def test_enforce_disables_both_production_and_preview_git_deploys():
    state = _project()
    calls = []

    def request(method, _url, _token, body=None):
        calls.append((method, body))
        if method == "PATCH":
            state["source"]["config"].update(body["source"]["config"])
        return state

    verified = enforce_disabled("acct", "token", request_fn=request)
    assert verified["source"]["config"]["production_deployments_enabled"] is False
    assert verified["source"]["config"]["preview_deployment_setting"] == "none"
    assert [method for method, _ in calls] == ["GET", "PATCH", "GET"]


def test_check_fails_closed_when_cloudflare_git_path_is_enabled():
    with pytest.raises(RuntimeError, match="auto-deployments are enabled"):
        check_disabled("acct", "token", request_fn=lambda *_args: _project())


@pytest.mark.parametrize("source", [None, {}])
def test_no_git_provider_is_already_locked(source):
    project = {"source": source}
    calls = []

    def request(method, _url, _token, body=None):
        calls.append((method, body))
        return project

    verified = enforce_disabled("acct", "token", request_fn=request)
    assert verified == project
    assert calls == [("GET", None), ("GET", None)]
    assert check_disabled("acct", "token", request_fn=request) == project
