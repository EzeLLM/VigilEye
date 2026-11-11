"""Token counting with tiktoken when available, a fast heuristic otherwise."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # not installed / offline
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # ~4 chars/token for English; be slightly conservative.
    return max(1, int(len(text) / 3.6))


def truncate(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    enc = _encoder()
    if enc is not None:
        return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens]) + " …[truncated]"
    return text[: int(max_tokens * 3.6)] + " …[truncated]"


__all__ = ["count_tokens", "truncate"]
