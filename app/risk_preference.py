"""
app/risk_preference.py

Reads user_risk_preference from app/data/user_style.json.
Written by the website API; read by the trading engine each block.
Falls back to USER_STYLE env var if the file is missing or corrupt.
"""

import os
import json
import pathlib

_PREF_FILE = pathlib.Path(__file__).parent / "data" / "user_style.json"
_VALID = {"high_risk_short_term", "medium_risk_swing", "low_risk_long_term"}


def read_risk_preference() -> str:
    try:
        data = json.loads(_PREF_FILE.read_text())
        val = str(data.get("style", "")).strip()
        if val in _VALID:
            return val
    except Exception:
        pass
    return os.environ.get("USER_STYLE", "high_risk_short_term")


def write_risk_preference(style: str) -> bool:
    if style not in _VALID:
        return False
    _PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PREF_FILE.write_text(json.dumps({"style": style}))
    return True
