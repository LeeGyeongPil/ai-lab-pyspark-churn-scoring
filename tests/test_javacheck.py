import pytest

from churn_scoring import javacheck


def test_missing_java_message(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no java here
    monkeypatch.setattr(javacheck, "candidate_homes", lambda: [])
    with pytest.raises(SystemExit) as exit_info:
        javacheck.ensure_java()
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "JAVA_HOME" in message and "17" in message
    assert "JAVA_GATEWAY_EXITED" not in message


def test_bad_java_home_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(javacheck, "candidate_homes", lambda: [])
    assert javacheck.find_java_home() is None


def test_version_parsing(tmp_path):
    fake = tmp_path / "java"
    fake.write_text("#!/bin/sh\necho 'openjdk version \"21.0.10\" 2026-01-20' >&2\n")
    fake.chmod(0o755)
    assert javacheck.java_major_version(str(fake)) == 21
    old = tmp_path / "java8"
    old.write_text("#!/bin/sh\necho 'java version \"1.8.0_402\"' >&2\n")
    old.chmod(0o755)
    assert javacheck.java_major_version(str(old)) == 8
