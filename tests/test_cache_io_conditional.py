from pathlib import Path

import cache_io


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or {"ETag": '"etag-1"'}
        self.error = error
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        assert kwargs["Body"].read() == b"receipt"
        return self.response


class PreconditionFailed(Exception):
    response = {
        "Error": {"Code": "PreconditionFailed"},
        "ResponseMetadata": {"HTTPStatusCode": 412},
    }


def configure(monkeypatch, client):
    monkeypatch.setattr(cache_io, "_client", lambda: client)
    monkeypatch.setattr(cache_io, "_r2_creds",
                        lambda: {"R2_BUCKET": "bucket"})


def test_conditional_create_uses_if_none_match(tmp_path, monkeypatch):
    source = tmp_path / "receipt.json"
    source.write_bytes(b"receipt")
    client = FakeClient()
    configure(monkeypatch, client)

    result, etag = cache_io.conditional_upload_from_local(
        str(source), "receipt.json", create_only=True)

    assert (result, etag) == ("uploaded", '"etag-1"')
    assert client.calls[0]["IfNoneMatch"] == "*"
    assert "IfMatch" not in client.calls[0]


def test_conditional_update_uses_expected_etag(tmp_path, monkeypatch):
    source = tmp_path / "receipt.json"
    source.write_bytes(b"receipt")
    client = FakeClient()
    configure(monkeypatch, client)

    result, _ = cache_io.conditional_upload_from_local(
        str(source), "receipt.json", expected_etag='"old-etag"')

    assert result == "uploaded"
    assert client.calls[0]["IfMatch"] == '"old-etag"'
    assert "IfNoneMatch" not in client.calls[0]


def test_conditional_precondition_failure_is_distinct_from_io_error(
        tmp_path, monkeypatch):
    source = tmp_path / "receipt.json"
    source.write_bytes(b"receipt")
    configure(monkeypatch, FakeClient(error=PreconditionFailed()))

    assert cache_io.conditional_upload_from_local(
        str(source), "receipt.json", create_only=True) == (
            "precondition_failed", None)
