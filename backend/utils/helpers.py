"""Shared helpers for AI response normalization."""


def normalize_ai_text_field(value: str | list | None) -> str:
    """If model returns a list for a string field, join with '. '."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(p).strip() for p in value if p is not None and str(p).strip()]
        return ". ".join(parts)
    return str(value).strip()
