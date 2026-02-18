import os
import shutil
import sys
from typing import Optional


def _candidate_paths(*paths: str) -> list[str]:
    return [p for p in paths if p]


def _in_pyinstaller_bundle() -> bool:
    # In PyInstaller, sys.frozen is truthy.
    return bool(getattr(sys, "frozen", False))


def _bundle_dir() -> str:
    # One-folder: binaries + vendor/ sit next to sys.executable.
    if _in_pyinstaller_bundle():
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


def find_vendor_exe(exe_name: str) -> Optional[str]:
    # vendor/<exe_name>
    p = os.path.join(_bundle_dir(), "vendor", exe_name)
    if os.path.exists(p):
        return p
    return None


def resolve_ffmpeg() -> str:
    # Allow override
    env = os.environ.get("HOLY_FFMPEG")
    if env and os.path.exists(env):
        return env

    # Prefer vendored binary (Windows offline installer)
    vend = find_vendor_exe("ffmpeg.exe") or find_vendor_exe("ffmpeg")
    if vend:
        return vend

    # Common macOS Homebrew path
    hb = "/opt/homebrew/bin/ffmpeg"
    if os.path.exists(hb):
        return hb

    # Fall back to PATH
    w = shutil.which("ffmpeg")
    if w:
        return w

    return "ffmpeg"  # last resort


def resolve_rubberband() -> Optional[str]:
    env = os.environ.get("HOLY_RUBBERBAND")
    if env and os.path.exists(env):
        return env

    vend = find_vendor_exe("rubberband.exe") or find_vendor_exe("rubberband")
    if vend:
        return vend

    hb = "/opt/homebrew/bin/rubberband"
    if os.path.exists(hb):
        return hb

    w = shutil.which("rubberband")
    if w:
        return w

    return None
