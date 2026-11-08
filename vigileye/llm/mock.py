"""Deterministic offline provider (base_url ``mock://``) so the full pipeline runs with no network."""

from __future__ import annotations

import json
import re
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


def mock_completion(messages, model: str = "mock", json_mode: bool = False):
    prompt = _text(messages)
    last = _text(messages[-1:]) if messages else ""
    usage = {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": 0}

    if any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict))
           for m in messages):
        text = "Mock image description: a screenshot containing text and a logo; no identifiable location."
    elif "pong" in last.lower():
        text = "pong"
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
