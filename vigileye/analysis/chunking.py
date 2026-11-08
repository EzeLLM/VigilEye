"""Token-aware packing of documents into prompt-sized chunks."""
from __future__ import annotations

from typing import Iterable

from ..models import Document
from ..tokens import count_tokens, truncate

#: Overhead (tokens) assumed for the separator between rendered documents.
_SEP_TOKENS = 4
DOC_SEPARATOR = "\n\n---\n\n"


def doc_ref(doc: Document) -> str:
    """Citation handle for a document (``platform:id``)."""
    ref = getattr(doc, "ref", None)
    if isinstance(ref, str) and ref:
        return ref
    return doc.id if doc.id.startswith(f"{doc.platform}:") else f"{doc.platform}:{doc.id}"


def render_doc(doc: Document, max_tokens: int | None = None) -> str:
    """Render a document for a prompt, truncated to ``max_tokens`` if given."""
    text = doc.render()
    if max_tokens is not None and count_tokens(text) > max_tokens:
        text = truncate(text, max_tokens)
    return text


def chunk_documents(docs: Iterable[Document], max_tokens: int) -> list[list[Document]]:
    """Greedily pack documents (in order) into chunks of at most ``max_tokens``.

    A single document whose rendering exceeds ``max_tokens`` is placed alone in
    its own chunk; :func:`render_chunk` truncates it so the chunk still fits.
    """
    max_tokens = max(int(max_tokens), 64)
    chunks: list[list[Document]] = []
    current: list[Document] = []
    used = 0
    for doc in docs:
        cost = min(count_tokens(doc.render()), max_tokens) + _SEP_TOKENS
        if current and used + cost > max_tokens:
            chunks.append(current)
            current, used = [], 0
        current.append(doc)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def render_chunk(chunk: list[Document], max_tokens: int | None = None) -> str:
    """Render a chunk to a single prompt-ready string, truncating oversized docs."""
    per_doc = None
    if max_tokens is not None:
        per_doc = max(32, max_tokens - _SEP_TOKENS) if len(chunk) == 1 else max_tokens
    return DOC_SEPARATOR.join(render_doc(d, per_doc) for d in chunk)


def chunk_tokens(chunk: list[Document]) -> int:
    return sum(count_tokens(d.render()) + _SEP_TOKENS for d in chunk)


__all__ = ["chunk_documents", "render_chunk", "render_doc", "doc_ref", "chunk_tokens", "DOC_SEPARATOR"]
