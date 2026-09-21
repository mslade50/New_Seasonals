from unittest.mock import Mock

import pytest

from scripts.validate_ep_runtime import validate_runtime

PIN = "a" * 40


@pytest.mark.parametrize(
    "tracked,untracked,passes",
    [
        (b"", b"", True),
        (b"", b"debug.log\0", True),
        (b"episodic_pivot/config.py\n", b"", False),
        (b"", b"unexpected.py\0", False),
        (b"", b"nested/debug.log\0", False),
    ],
)
def test_runtime_preserves_known_log_but_rejects_source_drift(
    tmp_path, monkeypatch, tracked, untracked, passes
):
    log = tmp_path / "debug.log"
    log.write_text("preserve diagnostic evidence")
    mock = Mock(side_effect=[PIN.encode(), tracked, untracked])
    monkeypatch.setattr("scripts.validate_ep_runtime.subprocess.check_output", mock)
    if passes:
        validate_runtime(tmp_path, PIN)
    else:
        with pytest.raises(ValueError):
            validate_runtime(tmp_path, PIN)
    assert log.read_text() == "preserve diagnostic evidence"
    assert all(call.args[0][0] == "git" for call in mock.call_args_list)


def test_wrong_commit_fails_before_any_content_check(tmp_path, monkeypatch):
    mock = Mock(return_value=b"b" * 40)
    monkeypatch.setattr("scripts.validate_ep_runtime.subprocess.check_output", mock)
    with pytest.raises(ValueError, match="pin"):
        validate_runtime(tmp_path, PIN)
    assert mock.call_count == 1


def test_debug_directory_is_not_whitelisted(tmp_path, monkeypatch):
    (tmp_path / "debug.log").mkdir()
    monkeypatch.setattr(
        "scripts.validate_ep_runtime.subprocess.check_output",
        Mock(side_effect=[PIN.encode(), b"", b"debug.log\0"]),
    )
    with pytest.raises(ValueError, match="untracked"):
        validate_runtime(tmp_path, PIN)
