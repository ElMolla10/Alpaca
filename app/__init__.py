# app/__init__.py

# Keep top-level package lightweight; re-export from the real subpackages.
from .agent.arl_agent import (
    ARLAgent,
    UserStyle,
    default_style,
    BlockContext,
    Decision,
)

# Optional export if you want BlockLedger available at app.BlockLedger
try:
    from .model.execution import BlockLedger  # noqa: F401
except Exception:
    BlockLedger = None  # not critical for import

__all__ = [
    "ARLAgent",
    "UserStyle",
    "default_style",
    "BlockContext",
    "Decision",
    "BlockLedger",
]
