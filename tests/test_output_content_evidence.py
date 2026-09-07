import datetime as dt
import hashlib
import io

import pytest

from scripts import automation_supervisor as sup


def test_equal_size_different_remote_content_is_not_success(tmp_path):
    path = tmp_path / "producer.json"
    path.write_bytes(b"good")
    class Backend:
        def head(self, key): return {"ContentLength": 4, "ETag": '"v1"'}
        def content_hash(self, key, etag):
            assert key == "producer.json" and etag == '"v1"'
            return hashlib.sha256(b"evil").hexdigest()
    validator = sup.OutputValidator(Backend())
    with sup.RunLogger(tmp_path / "run.log", echo=False) as logger:
        with pytest.raises(sup.ValidationError, match="content mismatch"):
            validator.validate((sup.OutputSpec("producer.json", r2_key="producer.json"),),
                repo_root=tmp_path, started_at_utc=dt.datetime.now(dt.timezone.utc), logger=logger)
    assert validator.evidence == []


def test_remote_hash_is_bound_to_exact_head_generation_and_closes_body():
    class Client:
        def __init__(self): self.etag = '"v1"'
        def get_object(self, **kwargs):
            assert kwargs["IfMatch"] == '"v1"'
            self.body = io.BytesIO(b"good")
            return {"Body": self.body, "ETag": self.etag}
    client = Client()
    backend = sup.R2Backend({"R2_ACCOUNT_ID": "fixture", "R2_ACCESS_KEY_ID": "fixture",
        "R2_SECRET_ACCESS_KEY": "fixture", "R2_BUCKET": "fixture"}, client=client)
    assert backend.content_hash("key", '"v1"') == hashlib.sha256(b"good").hexdigest()
    assert client.body.closed
    client.etag = '"v2"'
    with pytest.raises(sup.ValidationError, match="generation changed"):
        backend.content_hash("key", '"v1"')
    assert client.body.closed


def test_success_retains_sha_and_generation_evidence(tmp_path):
    path = tmp_path / "producer.json"; path.write_bytes(b"good")
    digest = hashlib.sha256(b"good").hexdigest()
    class Backend:
        def head(self, key): return {"ContentLength": 4, "ETag": '"v1"'}
        def content_hash(self, key, etag): return digest
    validator = sup.OutputValidator(Backend())
    with sup.RunLogger(tmp_path / "run.log", echo=False) as logger:
        validator.validate((sup.OutputSpec("producer.json", r2_key="producer.json"),),
            repo_root=tmp_path, started_at_utc=dt.datetime.now(dt.timezone.utc), logger=logger)
    assert validator.evidence == [{"key": "producer.json", "size": 4, "sha256": digest, "etag": '"v1"'}]
