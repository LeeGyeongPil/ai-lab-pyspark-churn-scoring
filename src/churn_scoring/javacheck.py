"""Find a usable Java (17+) before Spark starts, with a helpful message if none."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from .i18n import t

MIN_JAVA = 17


def candidate_homes() -> list[str]:
    """Typical install locations, newest first."""
    homes: list[str] = []
    for prefix in ("/opt/homebrew/opt", "/usr/local/opt"):
        for name in ("openjdk@21", "openjdk@17", "openjdk"):
            homes.append(f"{prefix}/{name}")
            homes.append(f"{prefix}/{name}/libexec/openjdk.jdk/Contents/Home")
    homes.extend(sorted(glob.glob("/Library/Java/JavaVirtualMachines/*/Contents/Home"), reverse=True))
    homes.extend(sorted(glob.glob("/usr/lib/jvm/*"), reverse=True))
    return homes


def _java_binary(home: str) -> Path | None:
    exe = Path(home) / "bin" / "java"
    return exe if exe.is_file() and os.access(exe, os.X_OK) else None


def java_major_version(java_exe: str) -> int | None:
    """Return the major version printed by `java -version`, or None."""
    try:
        out = subprocess.run(
            [java_exe, "-version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'version "(\d+)(?:\.(\d+))?', out.stdout + out.stderr)
    if not match:
        return None
    major = int(match.group(1))
    return int(match.group(2) or 0) if major == 1 else major


def find_java_home(environ: Mapping[str, str] | None = None) -> str | None:
    """Return a JAVA_HOME with Java >= 17, or None. Order: JAVA_HOME, PATH, known folders."""
    env = os.environ if environ is None else environ
    homes: list[str] = []
    if env.get("JAVA_HOME"):
        homes.append(env["JAVA_HOME"])
    on_path = shutil.which("java", path=env.get("PATH"))
    if on_path:
        homes.append(str(Path(on_path).resolve().parent.parent))
    homes.extend(candidate_homes())
    for home in homes:
        exe = _java_binary(home)
        if exe is None:
            continue
        major = java_major_version(str(exe))
        if major is not None and major >= MIN_JAVA:
            return home
    return None


def ensure_java() -> str:
    """Set JAVA_HOME for this process or exit with code 2 and install instructions."""
    home = find_java_home()
    if home is None:
        print(t("java.missing", min=MIN_JAVA), file=sys.stderr)
        raise SystemExit(2)
    os.environ["JAVA_HOME"] = home
    return home
