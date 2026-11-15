"""Language choice and message lookup (English / Korean)."""

from __future__ import annotations

import os
from typing import Mapping

from .messages_en import MESSAGES as EN
from .messages_ko import MESSAGES as KO

LANGS = ("en", "ko")
ENV_VAR = "CHURN_LANG"
_CATALOGS = {"en": EN, "ko": KO}
_current = "en"


def _normalize(value: str | None) -> str | None:
    if not value:
        return None
    return "ko" if value.strip().lower().startswith("ko") else "en"


def resolve_lang(cli_value: str | None, environ: Mapping[str, str] | None = None) -> str:
    """Order: --lang > CHURN_LANG > LC_ALL > LC_MESSAGES > LANG. Not Korean -> English."""
    env = os.environ if environ is None else environ
    if cli_value:
        return _normalize(cli_value) or "en"
    for name in (ENV_VAR, "LC_ALL", "LC_MESSAGES", "LANG"):
        if env.get(name):
            return _normalize(env[name]) or "en"
    return "en"


def set_lang(lang: str) -> None:
    global _current
    _current = lang if lang in _CATALOGS else "en"


def get_lang() -> str:
    return _current


def t(key: str, **values: object) -> str:
    """Look up a message in the current language and fill in `{placeholders}`."""
    template = _CATALOGS[_current].get(key) or EN[key]
    return template.format(**values) if values else template
