"""Resolve the studio credential without relying on a refreshed Windows shell."""
from __future__ import annotations

import os


def _windows_user_variable(name: str) -> str:
    if os.name != "nt":
        return ""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, kind = winreg.QueryValueEx(key, name)
        if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str):
            return value.strip()
    except OSError:
        pass
    return ""


def credential_value(name: str) -> str:
    # Explicit process values (including empty ones) take precedence. Read only
    # our dedicated saved credential; do not import unrelated Windows settings.
    if name in os.environ:
        return os.environ[name].strip()
    if name == "SSSTOKEN_API_KEY":
        return _windows_user_variable(name)
    return ""
