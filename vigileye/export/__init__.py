"""Interop exports: STIX 2.1 bundles (OpenCTI, MISP via misp-stix) and ATT&CK Navigator layers.

Both are pure functions of a finished :class:`~vigileye.models.Report`, deterministic (same report,
same output) and stdlib-only. Neither emits people: see :mod:`.stix` for the scope guard.
"""
from __future__ import annotations

from .navigator import report_to_navigator_layer, write_navigator
from .stix import (
    CONFIDENCE_TO_STIX,
    ENTITY_MAP,
    EXCLUDED_ENTITY_TYPES,
    STATEMENT,
    STIX_NS,
    TLP,
    TLP_OBJECTS,
    WEP_TO_STIX,
    canonical_json,
    deterministic_id,
    report_to_stix,
    sco_for_indicator,
    stix_ts,
    validate_bundle,
    write_stix,
)

__all__ = [
    "STIX_NS", "TLP", "TLP_OBJECTS", "WEP_TO_STIX", "CONFIDENCE_TO_STIX", "ENTITY_MAP", "EXCLUDED_ENTITY_TYPES",
    "STATEMENT", "stix_ts", "canonical_json", "deterministic_id", "sco_for_indicator", "report_to_stix",
    "validate_bundle", "write_stix", "report_to_navigator_layer", "write_navigator",
]
