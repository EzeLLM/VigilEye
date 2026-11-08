"""Deterministic offline provider (base_url ``mock://``) so the full pipeline runs with no network."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter

_REF_RE = re.compile(r"\[([a-z0-9_.-]+:[^\]\s|]+)\]")
_CAP_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,})?)\b")
_STOP = {"The", "This", "That", "Title", "You", "Your", "JSON", "Return", "Reply", "Main", "Document", "Documents",
         "Analyst", "Key", "BLUF", "Lens", "Focus", "Use", "Only", "Each", "For", "And", "With", "Not", "Describe"}


def _text(messages) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


#: An evidence block of a ``vigileye ask`` prompt (ask.build_messages): nonce, ref, body.
_ASK_BLOCK = re.compile(r"<<<EVIDENCE-([0-9a-f]{6,32}) E\d+ \| \[([^\]\s]+)\][^\n]*\n(.*?)\nEVIDENCE-\1>>>", re.S)
_SENTENCE = re.compile(r"(.+?[.!?])(?=\s|$)", re.S)


def _mock_ask_answer(prompt: str, max_blocks: int = 3) -> str | None:
    """Extractive answer for an ask prompt: the first sentence of the top evidence blocks, each cited.

    It quotes the sources verbatim and makes no assessment, so the citation check can verify it.
    """
    lines = []
    for _, ref, body in _ASK_BLOCK.findall(prompt):
        text = "\n".join(ln for ln in body.splitlines() if ln.strip() and not ln.startswith(("Title: ", "[")))
        text = " ".join(text.split())
        if not text or text == "(no text)":
            continue
        m = _SENTENCE.match(text)
        sentence = (m.group(1) if m else text)[:300].rstrip(" .!?")
        if sentence:
            lines.append(f"{sentence} [{ref}].")
        if len(lines) >= max_blocks:
            break
    if not lines:
        return None
    return "\n".join(lines) + "\n\nConfidence: low - offline mock provider, verbatim source extracts only"


def mock_completion(messages, model: str = "mock", json_mode: bool = False):
    prompt = _text(messages)
    last = _text(messages[-1:]) if messages else ""
    usage = {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": 0}

    if any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict))
           for m in messages):
        text = "Mock image description: a screenshot containing text and a logo; no identifiable location."
    elif "pong" in last.lower():
        text = "pong"
    elif not json_mode and "<<<EVIDENCE-" in last and (answer := _mock_ask_answer(last)):
        text = answer
    elif json_mode or "json" in last.lower():
        refs = list(dict.fromkeys(_REF_RE.findall(prompt)))[:6]
        names = [n for n, _ in Counter(n for n in _CAP_RE.findall(prompt) if n.split()[0] not in _STOP).most_common(8)]
        entities = [{"name": n, "type": "unknown", "mentions": 1, "context": []} for n in names]
        finding = {
            "title": "Mock finding",
            "summary": "Offline mock provider output; configure a real OpenAI-compatible endpoint for analysis.",
            "confidence": "low",
            "likelihood": "roughly even chance",
            "evidence": refs[:3],
        }
        payload = {
            "summary": "Mock summary of the supplied documents.",
            "findings": [finding],
            "key_judgments": [finding],
            "entities": entities,
            "notable_sources": refs[:3],
            "signals": ["mock signal"],
            "bluf": "MOCK BLUF: pipeline executed end-to-end on the offline provider; no real judgments were made.",
            "assumptions": ["The collected sample is representative (mock)."],
            "gaps": ["No real model was consulted (mock provider)."],
            "indicators_to_watch": ["Configure a real provider"],
            "body_markdown": "## Discussion\n\nThis report was produced by the **mock** provider.",
            "hypotheses": [
                {"hypothesis": "H1: Organic activity", "likelihood": "likely", "consistent": refs[:2], "inconsistent": [], "notes": "mock"},
                {"hypothesis": "H2: Coordinated activity", "likelihood": "unlikely", "consistent": [], "inconsistent": refs[:1], "notes": "mock"},
            ],
        }
        text = json.dumps(payload)
    else:
        text = "Mock analysis: offline provider response."
    usage["completion_tokens"] = max(1, len(text) // 4)
    return text, usage


# ---------------------------------------------------------------------------- embeddings
MOCK_EMBED_DIM = 256
_EMBED_WORD = re.compile(r"[^\W_]+")
_EMBED_STOP = frozenset("""
the and for with that this from are was were has have had not but its into than then them they their there
what which who whom whose how why when where will would can could should about over under also been being
""".split())


def _embed_tokens(text: str) -> list[str]:
    out = []
    for t in _EMBED_WORD.findall(unicodedata.normalize("NFKC", text or "").casefold()):
        if len(t) < 3 or t in _EMBED_STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def mock_embedding(texts, model: str = "mock", dim: int = MOCK_EMBED_DIM):
    """Deterministic hashed bag-of-words unit vectors (signed feature hashing, 1 + log tf weights).

    Texts sharing words get positive cosine similarity; the result never depends on Python's
    randomised ``hash()``. Returns ``(vectors, usage)`` like :func:`mock_completion`.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        counts = Counter(_embed_tokens(str(text))) or Counter({"\x00empty": 1})
        for tok, n in sorted(counts.items()):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(h[:4], "big") % dim] += (1.0 if h[4] & 1 else -1.0) * (1.0 + math.log(n))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # every feature cancelled out: fall back to a fixed unit vector
            vec[0], norm = 1.0, 1.0
        vectors.append([v / norm for v in vec])
    usage = {"prompt_tokens": max(1, sum(len(str(t)) for t in texts) // 4), "completion_tokens": 0}
    return vectors, usage
