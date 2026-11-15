import os
import string
import subprocess
import sys

import pytest

from churn_scoring import cli, i18n
from churn_scoring.messages_en import MESSAGES as EN
from churn_scoring.messages_ko import MESSAGES as KO


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_message_keys_match():
    assert set(EN) == set(KO)
    assert all(EN[k].strip() and KO[k].strip() for k in EN)
    for key in EN:  # same {placeholders} in both languages
        assert placeholders(EN[key]) == placeholders(KO[key]), key


def test_lang_resolution_order():
    env = {"CHURN_LANG": "ko", "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    assert i18n.resolve_lang("en", env) == "en"  # --lang beats everything
    assert i18n.resolve_lang(None, env) == "ko"  # CHURN_LANG beats LC_ALL
    assert i18n.resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LC_MESSAGES": "en", "LANG": "en"}) == "ko"
    assert i18n.resolve_lang(None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "fr_FR.UTF-8"}) == "en"  # not Korean -> English
    assert i18n.resolve_lang(None, {}) == "en"


def test_switching_changes_output():
    i18n.set_lang("ko")
    ko = i18n.t("step.evaluate")
    i18n.set_lang("en")
    assert ko != i18n.t("step.evaluate")
    assert i18n.t("table.header") != KO["table.header"]


def run_help(lang_args, env=None):
    out = subprocess.run(
        [sys.executable, "-m", "churn_scoring.cli", *lang_args, "evaluate", "--help"],
        capture_output=True, text=True, env={**os.environ, "LANG": "C", "LC_ALL": "C", **(env or {})},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_help_in_each_language():
    assert "평가" in run_help(["--lang", "ko"])
    assert "Evaluate the models" in run_help(["--lang", "en"])
    assert "평가" in run_help([], {"CHURN_LANG": "ko"})  # environment variable also works


def test_bad_values_give_fix_hint(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--customers", "0"])
    assert "Use a value like 3000" in capsys.readouterr().err
