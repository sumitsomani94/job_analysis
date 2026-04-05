"""In-memory session storage for full analysis results."""

from typing import Any
from uuid import UUID, uuid4

_sessions: dict[str, dict[str, Any]] = {}


def save_session(data: dict[str, Any], session_id: str | None = None) -> str:
    """Store analysis result and return session id string."""
    sid = session_id or str(uuid4())
    _sessions[sid] = data
    return sid


def get_session(session_id: str | UUID) -> dict[str, Any] | None:
    """Retrieve stored analysis by session id."""
    key = str(session_id)
    return _sessions.get(key)
