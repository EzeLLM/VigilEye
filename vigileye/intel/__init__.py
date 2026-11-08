"""Reference intelligence: authoritative vulnerability grounding (CISA KEV, FIRST EPSS) and warninglists.

* :mod:`.refdata` grounds CVE indicators in KEV and EPSS, renders the AUTHORITATIVE REFERENCE FACTS
  prompt block and corroborates or flags exploitation claims in a finished report.
* :mod:`.warninglists` annotates benign infrastructure (MISP warninglist format) without dropping it.

Both sources are keyless and read-only. KEV is a bulk download; EPSS only ever receives public CVE ids.
Neither module touches anything about people.
"""
from __future__ import annotations

from .refdata import (
    EPSS_REF,
    EPSS_URL,
    EXPLOIT_RE,
    KEV_REF,
    KEV_URL,
    RefCache,
    apply_grounding,
    epss_scores,
    exploitation_claim_check,
    ground_cves,
    grounding_summary,
    load_kev,
    lookup_cve,
    reference_facts_block,
    reference_section,
    refresh,
)
from .warninglists import (
    BUILTIN_LISTS,
    KIND_TO_MISP,
    Warninglist,
    annotate_indicators,
    load_misp_warninglists,
    match,
)

__all__ = [
    "KEV_URL", "EPSS_URL", "KEV_REF", "EPSS_REF", "EXPLOIT_RE", "RefCache", "load_kev", "epss_scores",
    "ground_cves", "reference_facts_block", "exploitation_claim_check", "grounding_summary", "reference_section",
    "apply_grounding", "refresh", "lookup_cve",
    "Warninglist", "BUILTIN_LISTS", "KIND_TO_MISP", "load_misp_warninglists", "match", "annotate_indicators",
]
