"""VigilEye analysis engine: tradecraft, lenses, enrichment and map-reduce synthesis."""
from __future__ import annotations

from .chunking import chunk_documents, render_chunk
from .critic import review_report
from .engine import DEFAULT_SETTINGS, AnalysisEngine, analyze
from .enrich import (
    defang,
    describe_media,
    extract_entities,
    extract_indicators,
    merge_entities,
    refang,
    summarize_indicators,
)
from .lenses import Lens, get_lens, list_lenses, load_lens, register_lens, template_env
from .signals import compute_signals, signals_markdown
from .temporal import detect_bursts, temporal_profile
from .tradecraft import (
    ADMIRALTY_CREDIBILITY,
    ADMIRALTY_RELIABILITY,
    ICD203_BLOCK,
    WEP_SCALE,
    default_source_grade,
    normalize_confidence,
    normalize_wep,
    wep_for_probability,
)
from .verify import verify_report

__all__ = [
    "AnalysisEngine", "analyze", "DEFAULT_SETTINGS",
    "chunk_documents", "render_chunk",
    "extract_indicators", "summarize_indicators", "refang", "defang",
    "describe_media", "extract_entities", "merge_entities",
    "Lens", "get_lens", "list_lenses", "load_lens", "register_lens", "template_env",
    "WEP_SCALE", "wep_for_probability", "normalize_wep", "normalize_confidence",
    "ADMIRALTY_RELIABILITY", "ADMIRALTY_CREDIBILITY", "default_source_grade", "ICD203_BLOCK",
    "compute_signals", "signals_markdown", "detect_bursts", "temporal_profile", "verify_report", "review_report",
]
