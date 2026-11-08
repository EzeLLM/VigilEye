"""MITRE ATT&CK Navigator layer (format 4.5) built from a report's technique indicators.

Each ``attack_technique`` indicator (T-id, sub-techniques included) becomes a scored technique:
the score is how often the T-id was mentioned, the comment and metadata list the citing document
refs (at most 10). Tactic ids (TA-ids) are ignored because Navigator scores techniques only. A
sub-technique's parent gets ``showSubtechniques`` so it is visible without manual expansion.
Pure stdlib; the output is plain JSON-serialisable data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..models import Report

LAYER_VERSION = "4.5"
NAVIGATOR_VERSION = "5.1.0"
MAX_DOC_METADATA = 10
_TECHNIQUE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def _slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "x"


def report_to_navigator_layer(report: Report, *, domain: str = "enterprise-attack", attack_version: str = "20",
                              name: str | None = None) -> dict:
    """Return a Navigator layer dict scoring every ATT&CK technique the report's indicators mention."""
    agg: dict[str, dict] = {}
    for rec in report.indicators or []:
        if not isinstance(rec, dict) or str(rec.get("type") or "").lower() != "attack_technique":
            continue
        tid = str(rec.get("value") or "").strip().upper()
        if not _TECHNIQUE.match(tid):
            continue
        slot = agg.setdefault(tid, {"count": 0, "docs": []})
        count = rec.get("count")
        slot["count"] += count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 1
        slot["docs"] += [str(d) for d in rec.get("docs") or [] if str(d) not in slot["docs"]]

    techniques: list[dict] = []
    for tid in sorted(agg):
        docs = agg[tid]["docs"]
        shown = docs[:MAX_DOC_METADATA]
        comment = "cited in: " + (", ".join(shown) if shown else "n/a")
        if len(docs) > len(shown):
            comment += f" (+{len(docs) - len(shown)} more)"
        techniques.append({"techniqueID": tid, "score": agg[tid]["count"], "comment": comment, "enabled": True,
                           "metadata": [{"name": "doc", "value": d} for d in shown]})
    parents = {t.split(".")[0] for t in agg if "." in t}
    for tech in techniques:
        if tech["techniqueID"] in parents:
            tech["showSubtechniques"] = True
    techniques += [{"techniqueID": p, "enabled": True, "showSubtechniques": True}
                   for p in sorted(parents - agg.keys())]

    max_score = max((t["score"] for t in techniques if "score" in t), default=1)
    return {
        "name": name or f"{report.case}: {report.lens} techniques",
        "versions": {"layer": LAYER_VERSION, "attack": str(attack_version), "navigator": NAVIGATOR_VERSION},
        "domain": domain,
        "description": (report.bluf or "")[:500],
        "sorting": 3,
        "layout": {"layout": "side", "showID": True, "showName": True},
        "hideDisabled": False,
        # maxValue must exceed minValue or Navigator's gradient divides by zero.
        "gradient": {"colors": ["#ffe766ff", "#ff6666ff"], "minValue": 1, "maxValue": max(2, max_score)},
        "legendItems": [],
        "metadata": [{"name": "case", "value": str(report.case)}, {"name": "generated_by", "value": "VigilEye"}],
        "techniques": techniques,
    }


def write_navigator(report: Report, path, **kw: Any) -> Path:
    """Write the layer as JSON. A directory ``path`` gets ``<case>-<lens>-<ts>.layer.json``."""
    p = Path(path)
    if p.is_dir():
        p = p / f"{_slug(report.case)}-{_slug(report.lens)}-{report.created_at:%Y%m%d-%H%M%S}.layer.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report_to_navigator_layer(report, **kw), indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


__all__ = ["report_to_navigator_layer", "write_navigator", "LAYER_VERSION", "NAVIGATOR_VERSION"]
