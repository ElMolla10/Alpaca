# app/__init__.py
from .agent.arl_agent import (
    ARLAgent,
    UserStyle,
    default_style,
    BlockContext,
    Decision,
)

# Optional: expose BlockLedger at app.BlockLedger (correct path is app/execution.py)
try:
    from .execution import BlockLedger  # correct location
except Exception:
    BlockLedger = None

__all__ = [
    "ARLAgent",
    "UserStyle",
    "default_style",
    "BlockContext",
    "Decision",
    "BlockLedger",
]
