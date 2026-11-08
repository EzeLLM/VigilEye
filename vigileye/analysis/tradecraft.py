"""Intelligence tradecraft primitives.

Implements the vocabulary professional analysts use so that every VigilEye
product speaks the same language:

* Words of Estimative Probability (WEP) -- ICD 203 / ODNI scale.
* Analytic confidence levels (low / moderate / high).
* Admiralty / NATO (STANAG 2511) source grading: reliability A-F and
  information credibility 1-6.
* The ICD 203 analytic standards, rendered as a block for system prompts.
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Words of Estimative Probability (ICD 203)
# ---------------------------------------------------------------------------

#: (term, low, high) -- probabilities in [0, 1].
WEP_SCALE: list[tuple[str, float, float]] = [
    ("almost no chance", 0.01, 0.05),
    ("very unlikely", 0.05, 0.20),
    ("unlikely", 0.20, 0.45),
    ("roughly even chance", 0.45, 0.55),
    ("likely", 0.55, 0.80),
    ("very likely", 0.80, 0.95),
    ("almost certain", 0.95, 0.99),
]

WEP_TERMS: list[str] = [t for t, _, _ in WEP_SCALE]

# Common synonyms analysts (and LLMs) use, mapped onto the canonical scale.
_WEP_SYNONYMS: dict[str, str] = {
    "remote": "almost no chance",
    "almost no chance": "almost no chance",
    "highly improbable": "very unlikely",
    "very unlikely": "very unlikely",
    "highly unlikely": "very unlikely",
    "improbable": "unlikely",
    "unlikely": "unlikely",
    "probably not": "unlikely",
    "roughly even chance": "roughly even chance",
    "roughly even odds": "roughly even chance",
    "even chance": "roughly even chance",
    "even odds": "roughly even chance",
    "possible": "roughly even chance",
    "probable": "likely",
    "probably": "likely",
    "likely": "likely",
    "very likely": "very likely",
    "highly likely": "very likely",
    "highly probable": "very likely",
    "almost certain": "almost certain",
    "almost certainly": "almost certain",
    "nearly certain": "almost certain",
    "certain": "almost certain",
}


def wep_for_probability(p: float) -> str:
    """Return the WEP term for a probability ``p`` (0-1, or 0-100 as percent)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "roughly even chance"
    if p > 1.0:
        p = p / 100.0
    p = min(max(p, 0.0), 1.0)
    for term, low, high in WEP_SCALE:
        if p <= high:
            return term
    return "almost certain"


def wep_range(term: str) -> tuple[float, float] | None:
    """Return the (low, high) probability band for a WEP term or synonym."""
    canon = normalize_wep(term)
    if canon is None:
        return None
    for t, low, high in WEP_SCALE:
        if t == canon:
            return (low, high)
    return None


def normalize_wep(value: object) -> str | None:
    """Map free text (or a number) onto a canonical WEP term; ``None`` if unknown."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return wep_for_probability(value)
    s = str(value).strip().lower()
    if not s:
        return None
    pct = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)\s*%?", s)
    if pct:
        return wep_for_probability(float(pct.group(1)))  # >1 is treated as percent
    s = re.sub(r"[^a-z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _WEP_SYNONYMS:
        return _WEP_SYNONYMS[s]
    # Longest synonym contained in the string wins ("very likely" beats "likely").
    for syn in sorted(_WEP_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(syn)}\b", s):
            return _WEP_SYNONYMS[syn]
    return None


# ---------------------------------------------------------------------------
# Analytic confidence
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS: dict[str, str] = {
    "low": (
        "Information is scant, questionable, fragmented, or poorly corroborated; "
        "solid analytic inferences are difficult, or sources raise significant concerns."
    ),
    "moderate": (
        "Information is credibly sourced and plausible but not of sufficient quality "
        "or corroboration to warrant a higher level of confidence."
    ),
    "high": (
        "Judgments are based on high-quality information from multiple sources, and/or "
        "the nature of the issue makes a solid judgment possible. High confidence is not "
        "a fact or certainty; such judgments still carry a risk of being wrong."
    ),
}


def normalize_confidence(value: object) -> str:
    """Normalize free-form confidence (text or number) to low|moderate|high."""
    if value is None:
        return "moderate"
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1.0:
            v /= 100.0
        return "low" if v < 0.4 else ("high" if v >= 0.75 else "moderate")
    s = str(value).strip().lower()
    if not s:
        return "moderate"
    try:
        return normalize_confidence(float(s.rstrip("%")) / (100.0 if s.endswith("%") else 1.0))
    except ValueError:
        pass
    if any(k in s for k in ("high", "strong", "confident")):
        return "high"
    if any(k in s for k in ("low", "weak", "limited", "poor", "minimal")):
        return "low"
    if any(k in s for k in ("moderate", "medium", "mid", "fair")):
        return "moderate"
    return "moderate"


# ---------------------------------------------------------------------------
# Admiralty / NATO source grading (STANAG 2511)
# ---------------------------------------------------------------------------

ADMIRALTY_RELIABILITY: dict[str, str] = {
    "A": "Completely reliable - no doubt of authenticity, trustworthiness, or competency; history of complete reliability.",
    "B": "Usually reliable - minor doubt about authenticity, trustworthiness, or competency; history of valid information most of the time.",
    "C": "Fairly reliable - doubt of authenticity, trustworthiness, or competency, but has provided valid information in the past.",
    "D": "Not usually reliable - significant doubt about authenticity, trustworthiness, or competency; has provided valid information in the past.",
    "E": "Unreliable - lacking in authenticity, trustworthiness, and competency; history of invalid information.",
    "F": "Reliability cannot be judged - no basis exists for evaluating the reliability of the source.",
}

ADMIRALTY_CREDIBILITY: dict[str, str] = {
    "1": "Confirmed by other sources - confirmed by other independent sources; logical in itself; consistent with other information on the subject.",
    "2": "Probably true - not confirmed; logical in itself; consistent with other information on the subject.",
    "3": "Possibly true - not confirmed; reasonably logical in itself; agrees with some other information on the subject.",
    "4": "Doubtful - not confirmed; possible but not logical; no other information on the subject.",
    "5": "Improbable - not confirmed; not logical in itself; contradicted by other information on the subject.",
    "6": "Truth cannot be judged - no basis exists for evaluating the validity of the information.",
}

#: Default (reliability, credibility) by platform. Unverified, user-generated
#: social media starts at "cannot be judged"; curated or institutional feeds
#: earn a modest presumption of reliability.
_PLATFORM_DEFAULT_GRADES: dict[str, str] = {
    "reddit": "F6",
    "twitter": "F6",
    "x": "F6",
    "mastodon": "F6",
    "bluesky": "F6",
    "telegram": "F6",
    "tiktok": "F6",
    "youtube": "F6",
    "instagram": "F6",
    "facebook": "F6",
    "forum": "F6",
    "4chan": "E6",
    "pastebin": "F6",
    "hackernews": "F3",
    "github": "C3",
    "rss": "C3",
    "news": "C3",
    "web": "F6",
    "http": "F6",
    "file": "F6",
    "local": "F6",
    "gov": "B2",
    "cisa": "B2",
    "nvd": "B2",
    "cve": "B2",
}


def default_source_grade(platform: str | None, source: str | None = None) -> str:
    """Return a conservative Admiralty grade (e.g. ``"F6"``) for a platform.

    Analysts should upgrade grades only on the basis of a track record or
    corroboration; this function never assumes reliability for user-generated
    content.
    """
    p = (platform or "").strip().lower()
    if p in _PLATFORM_DEFAULT_GRADES:
        return _PLATFORM_DEFAULT_GRADES[p]
    s = (source or "").lower()
    if s.endswith(".gov") or ".gov/" in s or ".mil" in s:
        return "B2"
    if any(k in p for k in ("rss", "news", "feed")):
        return "C3"
    return "F6"


def parse_grade(grade: str | None) -> tuple[str, str]:
    """Split ``"B2"`` into ``("B", "2")``; falls back to ``("F", "6")``."""
    if not grade:
        return ("F", "6")
    g = str(grade).strip().upper()
    m = re.fullmatch(r"([A-F])\s*[-/]?\s*([1-6])", g)
    if m:
        return (m.group(1), m.group(2))
    rel = g[0] if g and g[0] in ADMIRALTY_RELIABILITY else "F"
    cred = next((c for c in g if c in ADMIRALTY_CREDIBILITY), "6")
    return (rel, cred)


def describe_grade(grade: str) -> str:
    rel, cred = parse_grade(grade)
    return f"{rel}{cred}: {ADMIRALTY_RELIABILITY[rel].split(' - ')[0]} / {ADMIRALTY_CREDIBILITY[cred].split(' - ')[0]}"


def grade_summary(grades: Iterable[str]) -> dict[str, int]:
    """Count occurrences of each grade (useful for the reduce prompt / stats)."""
    out: dict[str, int] = {}
    for g in grades:
        rel, cred = parse_grade(g)
        key = f"{rel}{cred}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# ICD 203 analytic standards
# ---------------------------------------------------------------------------

ICD203_STANDARDS: list[tuple[str, str]] = [
    ("Objectivity", "Perform analysis free of bias, advocacy, or a preferred outcome; consider alternative perspectives and contrary information."),
    ("Independence of political considerations", "Do not distort analysis to support any agenda or audience."),
    ("Timeliness", "Note the time span covered by the evidence and flag stale or rapidly changing information."),
    ("All available sources", "Base judgments on all the material provided; never fabricate, never import unstated facts as evidence."),
]

ICD203_TRADECRAFT: list[tuple[str, str]] = [
    ("Sourcing", "Properly describe the quality and credibility of underlying sources; cite document ids for every claim."),
    ("Uncertainty", "Express and explain uncertainties using calibrated likelihood (WEP) and confidence levels."),
    ("Distinguish", "Clearly separate underlying intelligence (what sources say) from analyst assumptions and judgments."),
    ("Alternatives", "Incorporate analysis of alternative explanations and hypotheses."),
    ("Relevance", "Demonstrate customer relevance and address implications."),
    ("Logical argumentation", "Use clear, logical argumentation; state the reasoning chain from evidence to judgment."),
    ("Change", "Explain change to or consistency of judgments over time / compared with prior assessments."),
    ("Accuracy", "Make accurate judgments and assessments; do not overstate what the evidence supports."),
    ("Visualization", "Use tables or structured lists where they make the analysis clearer."),
]


def wep_table() -> str:
    return "\n".join(f"- {t}: {int(lo * 100)}-{int(hi * 100)}%" for t, lo, hi in WEP_SCALE)


def icd203_block() -> str:
    """Return the ICD 203 standards + WEP + confidence + Admiralty text block."""
    lines = ["ANALYTIC STANDARDS (ICD 203):"]
    lines += [f"- {name}: {desc}" for name, desc in ICD203_STANDARDS]
    lines.append("ANALYTIC TRADECRAFT STANDARDS:")
    lines += [f"{i}. {name}: {desc}" for i, (name, desc) in enumerate(ICD203_TRADECRAFT, 1)]
    lines.append("WORDS OF ESTIMATIVE PROBABILITY (use exactly these terms for likelihood):")
    lines.append(wep_table())
    lines.append("CONFIDENCE LEVELS (about the evidence base, not the likelihood):")
    lines += [f"- {k}: {v}" for k, v in CONFIDENCE_LEVELS.items()]
    lines.append("SOURCE GRADING (Admiralty/NATO): reliability A-F x credibility 1-6, e.g. B2, F6.")
    lines += [f"- {k}: {v.split(' - ')[0]}" for k, v in ADMIRALTY_RELIABILITY.items()]
    lines += [f"- {k}: {v.split(' - ')[0]}" for k, v in ADMIRALTY_CREDIBILITY.items()]
    return "\n".join(lines)


ICD203_BLOCK: str = icd203_block()

__all__ = [
    "WEP_SCALE",
    "WEP_TERMS",
    "wep_for_probability",
    "wep_range",
    "normalize_wep",
    "CONFIDENCE_LEVELS",
    "normalize_confidence",
    "ADMIRALTY_RELIABILITY",
    "ADMIRALTY_CREDIBILITY",
    "default_source_grade",
    "parse_grade",
    "describe_grade",
    "grade_summary",
    "ICD203_STANDARDS",
    "ICD203_TRADECRAFT",
    "ICD203_BLOCK",
    "icd203_block",
    "wep_table",
]
