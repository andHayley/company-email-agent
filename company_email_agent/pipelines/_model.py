"""Parsing helpers for the course's plain-text model interface."""

import json
import re


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def read_object(reply: str) -> dict | None:
    """Read a JSON object, allowing a surrounding Markdown code fence."""
    reply = reply.strip()
    fence = _JSON_FENCE.fullmatch(reply)
    if fence:
        reply = fence[1]
    try:
        value = json.loads(reply)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None
