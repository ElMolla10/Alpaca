"""
app/risk_preference.py

Stores the active risk profile for the local Project Command Center.
The dashboard writes the selected style to app/data/user_style.json and
the trading engine reads it each block. Falls back to USER_STYLE env var
if the file is missing or corrupt.
"""

import os
import json
import pathlib

_PREF_FILE = pathlib.Path(__file__).parent / "data" / "user_style.json"
VALID_RISK_STYLES = (
    "high_risk_short_term",
    "medium_risk_swing",
    "low_risk_long_term",
)

RISK_STYLE_LABELS = {
    "high_risk_short_term": "High risk, short term",
    "medium_risk_swing": "Medium risk, swing",
    "low_risk_long_term": "Low risk, long term",
}


def read_risk_preference() -> str:
    try:
        data = json.loads(_PREF_FILE.read_text())
        val = str(data.get("style", "")).strip()
        if val in VALID_RISK_STYLES:
            return val
    except Exception:
        pass
    return os.environ.get("USER_STYLE", "high_risk_short_term")


def write_risk_preference(style: str) -> bool:
    if style not in VALID_RISK_STYLES:
        return False
    _PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PREF_FILE.write_text(json.dumps({"style": style}))
    return True
