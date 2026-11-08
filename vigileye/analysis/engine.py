"""The VigilEye analysis engine: map-reduce intelligence production.

Pipeline::

    dedupe -> computed signals (copypasta, bursts, coordination; no LLM) -> [vision]
           -> IOC regex + warninglists [+ KEV/EPSS grounding] -> collapse near-dups -> select & chunk
           -> MAP (parallel, analyst) -> entity merge -> REDUCE (synthesizer) -> [ACH (analyst)]
           -> Report -> [citation verification] -> [reference-fact checks] -> [tradecraft critic]

Every LLM step degrades gracefully: failures are recorded in
``report.meta["errors"]`` and ``report.gaps`` and the engine keeps whatever
partial results it has instead of raising.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .. import __version__
from ..models import Document, Entity, Finding, Report
from ..tokens import count_tokens
from . import tradecraft as tc
from .chunking import chunk_documents, chunk_tokens, doc_ref, fence_untrusted, render_chunk
from .enrich import (
    coerce_entities,
    describe_media,
    extract_entities as _extract_entities,
    extract_indicators,
    merge_entities,
    summarize_indicators,
)
from ..intel import (
    BUILTIN_LISTS,
    annotate_indicators,
    apply_grounding,
    ground_cves as _ground_cves,
    load_misp_warninglists,
    reference_facts_block,
)
from .critic import review_report
from .lenses import Lens, get_lens, template_env
from .signals import (
    collapse_near_duplicates,
    compute_signals,
    render_signals_block,
    signals_for_refs,
    signals_markdown,
)
from .temporal import describe_profile, temporal_profile
from .verify import verify_report

log = logging.getLogger("vigileye.analysis")

DEFAULT_SETTINGS: dict[str, Any] = {
    "lens": "general",
    "chunk_tokens": 12000,
    "max_chunks": 24,
    "extract_entities": True,
    "entity_pass": False,          # extra dedicated extractor-role pass per chunk
    "use_vision": False,
    "max_images": 12,
    "ach": False,
    "max_concurrency": 4,
    "max_findings_per_chunk": 6,
    "max_entities": 100,
    "reduce_entities": 40,
    "template_dir": None,
    # computed evidence signals (analysis/signals.py, temporal.py, coordination.py): deterministic, no LLM
    "signals": True,
    "collapse_near_dups": True,    # the map phase sees one representative per copypasta cluster
    "near_dup_threshold": 0.8,
    "near_dup_min_size": 3,
    "bursts": {"z_threshold": 3.5, "min_events": 5},          # temporal.detect_bursts kwargs (False disables)
    "coordination": {"window_s": 60, "min_weight": 2, "n_iter": 200, "public_accounts": []},  # (False disables)
    # claim-level citation verification (analysis/verify.py)
    "verify": True,
    "verify_llm": True,            # lexical pre-filter + verifier-role LLM with a verbatim-quote guard
    "verify_max_pairs": 60,
    # tradecraft critic (analysis/critic.py): rule-based by default, LLM red-team with critic_llm
    "critique": True,
    "critic_llm": False,
    "stale_days": 30,
    # authoritative vulnerability grounding (intel/): CISA KEV + FIRST EPSS; network unless intel.offline
    "ground_cves": False,
    "warninglists_dir": None,      # optional MISP misp-warninglists clone
    # when max_chunks forces a cut and a query is set, rank documents by relevance to it (retrieval.py:
    # BM25/FTS5 + the optional embedder role, fused 0.6) before engagement/recency (0.4)
    "relevance_selection": True,
}

_REF_IN_TEXT = re.compile(r"\[([a-z0-9_.-]+:[^\]\s|,]+)\]", re.I)


def _utc_iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


class AnalysisEngine:
    """Turn a pile of public documents into a tradecraft-grade :class:`Report`."""

    def __init__(self, router: Any, config: Any = None):
        self.router = router
        self.config = config
        section: dict = {}
        if config is not None:
            if hasattr(config, "section"):
                section = config.section("analysis") or {}
            elif isinstance(config, dict):
                section = config.get("analysis") or {}
        self.settings: dict[str, Any] = {**DEFAULT_SETTINGS, **{k: v for k, v in section.items() if v is not None}}
        self.env = template_env(self.settings.get("template_dir"))
        self._warninglists = list(BUILTIN_LISTS)
        if self.settings.get("warninglists_dir"):
            self._warninglists += load_misp_warninglists(self.settings["warninglists_dir"])

    # ------------------------------------------------------------------ helpers
    def _render(self, name: str, **ctx: Any) -> str:
        base = {"icd203": tc.ICD203_BLOCK, "wep_terms": tc.WEP_TERMS, "wep_scale": tc.WEP_SCALE}
        base.update(ctx)
        return self.env.get_template(name).render(**base)

    def _messages(self, lens: Lens, query: str, task: str, template: str, **ctx: Any) -> list[dict]:
        system = self._render("system.j2", lens=lens, query=query, task=task)
        user = self._render(template, lens=lens, query=query, **ctx)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _emit(progress: Callable[[str], None] | None, msg: str) -> None:
        log.info(msg)
        if progress:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- selection
    @staticmethod
    def _priority(docs: list[Document]) -> list[Document]:
        """Rank documents by a blend of engagement and recency (rank-normalised)."""
        if len(docs) < 2:
            return list(docs)
        n = len(docs) - 1
        by_eng = sorted(range(len(docs)), key=lambda i: docs[i].engagement())
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        by_time = sorted(range(len(docs)), key=lambda i: docs[i].created_at or epoch)
        score = [0.0] * len(docs)
        for rank, i in enumerate(by_eng):
            score[i] += 0.6 * rank / n
        for rank, i in enumerate(by_time):
            score[i] += 0.4 * rank / n
        return [docs[i] for i in sorted(range(len(docs)), key=lambda i: -score[i])]

    def _select_and_chunk(self, docs: list[Document], budget: int, max_chunks: int | None, query: str = ""):
        def chrono(ds: Iterable[Document]) -> list[Document]:
            epoch = datetime.min.replace(tzinfo=timezone.utc)
            return sorted(ds, key=lambda d: d.created_at or epoch)

        chunks = chunk_documents(chrono(docs), budget)
        if not max_chunks or max_chunks <= 0 or len(chunks) <= max_chunks:
            return chunks, []
        ranked = self._priority(docs)
        if query and query.strip() and self.settings.get("relevance_selection", True):
            try:
                from ..retrieval import rank_for_query

                ranked = rank_for_query(docs, query, self.router)  # 0.6 relevance (RRF) + 0.4 engagement/recency
            except Exception as e:  # noqa: BLE001 - selection must never fail a run
                log.warning("relevance-aware selection failed, using engagement/recency: %s", e)
        kept_ids = {d.fingerprint() for chunk in chunk_documents(ranked, budget)[:max_chunks] for d in chunk}
        kept = [d for d in ranked if d.fingerprint() in kept_ids]  # in priority order
        chunks = chunk_documents(chrono(kept), budget)
        # Repacking in time order can need more chunks than the ranked packing (long documents
        # scattered in time each close a partly filled chunk). Drop the lowest-priority documents
        # until it fits, never the chronological tail (that would silently lose the newest).
        while len(chunks) > max_chunks and len(kept) > 1:
            kept.pop()
            chunks = chunk_documents(chrono(kept), budget)
        final_ids = {d.fingerprint() for c in chunks for d in c}
        dropped = [d for d in docs if d.fingerprint() not in final_ids]
        return chunks, dropped

    # ------------------------------------------------------------- normalising
    @staticmethod
    def _resolve_ref(raw: Any, known: set[str], by_id: dict[str, str]) -> str | None:
        if raw is None:
            return None
        s = str(raw).strip().strip("[]()").strip()
        if not s:
            return None
        if s in known:
            return s
        if s in by_id:
            return by_id[s]
        low = s.lower()
        for k in known:
            if k.lower() == low:
                return k
        return None

    def _finding(self, d: Any, known: set[str], by_id: dict[str, str], stats: Counter) -> Finding | None:
        if isinstance(d, Finding):
            d = d.to_dict()
        if isinstance(d, str):
            d = {"title": d[:120], "summary": d}
        if not isinstance(d, dict):
            return None
        title = str(d.get("title") or d.get("judgment") or d.get("headline") or "").strip()
        summary = str(d.get("summary") or d.get("text") or d.get("description") or d.get("detail") or "").strip()
        if not title and not summary:
            return None
        if not title:
            title = summary.split(". ")[0][:120]
        ev_raw = d.get("evidence") or d.get("sources") or d.get("citations") or []
        if isinstance(ev_raw, str):
            ev_raw = re.split(r"[,\s]+", ev_raw)
        evidence: list[str] = []
        for e in list(ev_raw) + _REF_IN_TEXT.findall(summary):
            ref = self._resolve_ref(e, known, by_id)
            if ref and ref not in evidence:
                evidence.append(ref)
            elif not ref and str(e).strip():
                stats["unverifiable_citations"] += 1
        tags = d.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        likelihood = tc.normalize_wep(d.get("likelihood") or d.get("probability"))
        return Finding(
            title=title,
            summary=summary or title,
            confidence=tc.normalize_confidence(d.get("confidence") or "low"),  # unstated is not "moderate"
            likelihood=likelihood,
            evidence=evidence[:15],
            tags=[str(t) for t in tags][:8],
        )

    @staticmethod
    def _str_list(value: Any, limit: int = 30) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [v.strip(" -*\t") for v in value.splitlines() if v.strip()]
        out: list[str] = []
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, dict):
                v = v.get("text") or v.get("indicator") or v.get("gap") or v.get("assumption") or json.dumps(v, default=str)
            s = str(v).strip()
            if s and s not in out:
                out.append(s)
        return out[:limit]

    # --------------------------------------------------------------------- MAP
    def _map_chunk(self, idx: int, total: int, chunk: list[Document], lens: Lens, query: str, budget: int,
                   indicators: list[dict], want_entities: bool, known: set[str], by_id: dict[str, str],
                   computed_signals: list[dict] | None = None, reference_facts: str = "",
                   citation_check: bool = False, author_key: bytes | None = None) -> dict:
        refs = {doc_ref(d) for d in chunk}
        times = [d.created_at for d in chunk if d.created_at]
        time_range = f"{min(times):%Y-%m-%d} to {max(times):%Y-%m-%d}" if times else ""
        chunk_iocs = summarize_indicators([r for r in indicators if refs.intersection(r["docs"])], top_per_type=5)
        result: dict[str, Any] = {"index": idx, "doc_ids": sorted(refs), "summary": "", "findings": [],
                                  "entities": [], "notable_sources": [], "signals": [], "error": None,
                                  "tokens": chunk_tokens(chunk)}
        local_stats: Counter = Counter()
        try:
            documents, nonce = fence_untrusted(render_chunk(chunk, budget, key=author_key))
            messages = self._messages(
                lens, query, f"chunk analysis {idx}/{total}", "map.j2",
                chunk_index=idx, chunk_count=total, doc_count=len(chunk), time_range=time_range,
                indicators=chunk_iocs if chunk_iocs.get("counts") else None,
                documents=documents, nonce=nonce,
                entity_types=lens.entity_types,
                max_findings=self.settings["max_findings_per_chunk"],
                computed_signals=render_signals_block(signals_for_refs(computed_signals or [], refs)),
                reference_facts=reference_facts,
                citation_check=citation_check,
            )
            payload = self.router.chat_json(messages, role="analyst")
            if isinstance(payload, list):
                payload = {"findings": payload}
            if not isinstance(payload, dict):
                raise ValueError(f"unexpected map payload type {type(payload).__name__}")
            result["summary"] = str(payload.get("summary") or "").strip()
            findings = payload.get("findings") or payload.get("key_findings") or []
            result["findings"] = [f for f in (self._finding(x, known, by_id, local_stats)
                                              for x in (findings if isinstance(findings, list) else [findings])) if f]
            result["findings"] = result["findings"][: int(self.settings["max_findings_per_chunk"]) + 2]
            if want_entities:
                result["entities"] = coerce_entities(payload.get("entities") or [])
            result["notable_sources"] = [r for r in (self._resolve_ref(x, known, by_id)
                                                     for x in self._str_list(payload.get("notable_sources")))
                                         if r]
            result["signals"] = self._str_list(payload.get("signals"), limit=15)
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            log.warning("map chunk %d failed: %s", idx, e)
            result["error"] = f"map chunk {idx}: {type(e).__name__}: {e}"
        if want_entities and (self.settings.get("entity_pass") or (not result["entities"] and result["error"] is None)):
            try:
                result["entities"] = merge_entities(
                    result["entities"] + _extract_entities(chunk, self.router, lens=lens, env=self.env, max_tokens=budget,
                                                           author_key=author_key)
                )
            except Exception as e:  # noqa: BLE001
                log.warning("entity extraction chunk %d failed: %s", idx, e)
                result.setdefault("entity_error", f"entities chunk {idx}: {type(e).__name__}: {e}")
        result["unverifiable_citations"] = local_stats["unverifiable_citations"]
        return result

    # ------------------------------------------------------------------ REDUCE
    def _reduce(self, lens: Lens, query: str, chunk_results: list[dict], entities: list[Entity],
                ioc_summary: dict, stats: dict, grades: dict, computed_signals: list[dict] | None = None,
                reference_facts: str = "", citation_check: bool = False) -> dict:
        ctx_window = int(self.router.context_window("synthesizer"))
        limit = max(2000, int(ctx_window * 0.7))
        max_f, max_chars = int(self.settings["max_findings_per_chunk"]), 1200
        n_ent = int(self.settings["reduce_entities"])
        messages: list[dict] = []
        for _ in range(6):
            summaries = []
            for c in chunk_results:
                if c.get("error") and not c.get("findings"):
                    continue
                summaries.append({
                    "index": c["index"],
                    "summary": c["summary"][:max_chars],
                    "findings": [{**f.to_dict(), "summary": f.summary[:max_chars]} for f in c["findings"][:max_f]],
                    "signals": c["signals"][:8],
                })
            messages = self._messages(
                lens, query, "final synthesis", "reduce.j2",
                chunk_summaries=summaries, entities=entities[:n_ent], indicators=ioc_summary if ioc_summary.get("counts") else None,
                stats=stats, source_grades=grades,
                computed_signals=render_signals_block(computed_signals or []),
                reference_facts=reference_facts,
                citation_check=citation_check,
            )
            if sum(count_tokens(m["content"]) for m in messages) <= limit:
                break
            max_f, max_chars, n_ent = max(1, max_f - 1), max(200, int(max_chars * 0.6)), max(10, n_ent // 2)
        payload = self.router.chat_json(messages, role="synthesizer")
        if isinstance(payload, list):
            payload = {"key_judgments": payload}
        if not isinstance(payload, dict):
            raise ValueError(f"unexpected reduce payload type {type(payload).__name__}")
        return payload

    # --------------------------------------------------------------------- ACH
    def _ach(self, lens: Lens, query: str, judgments: list[Finding], chunk_results: list[dict],
             known: set[str], by_id: dict[str, str],
             computed_signals: list[dict] | None = None) -> tuple[list[dict], dict]:
        # computed signals go first so the 40-item cap can never drop them; chunk items are then taken
        # round-robin across chunks (chunks are in time order: a sequential fill would keep only the
        # oldest material)
        evidence: list[str] = [f"{x['id']}: {x['summary']}" for x in (computed_signals or [])[:10]]
        per_chunk: list[list[str]] = []
        for c in chunk_results:
            items = [f"{f.title}: {f.summary[:300]} ({', '.join(f.evidence[:4])})" for f in c["findings"]]
            items += [f"signal: {s}" for s in c["signals"][:3]]
            per_chunk.append(items)
        for k in range(max((len(x) for x in per_chunk), default=0)):
            evidence += [items[k] for items in per_chunk if k < len(items)]
        evidence = list(dict.fromkeys(evidence))[:40]
        messages = self._messages(lens, query, "analysis of competing hypotheses", "ach.j2",
                                  key_judgments=judgments, evidence=evidence,
                                  min_hypotheses=3, max_hypotheses=5)
        payload = self.router.chat_json(messages, role="analyst")
        if isinstance(payload, list):
            payload = {"hypotheses": payload}
        hyps: list[dict] = []
        matrix = payload.get("matrix") if isinstance(payload.get("matrix"), dict) else {}
        for i, h in enumerate(payload.get("hypotheses") or [], 1):
            if isinstance(h, str):
                h = {"hypothesis": h}
            if not isinstance(h, dict):
                continue
            text = str(h.get("hypothesis") or h.get("text") or h.get("title") or "").strip()
            if not text:
                continue

            def refs(items: Any) -> list[str]:
                out = []
                for x in self._str_list(items, limit=40):
                    out.append(self._resolve_ref(x, known, by_id) or x)
                return out

            consistent, inconsistent = refs(h.get("consistent")), refs(h.get("inconsistent"))
            hid = str(h.get("id") or f"H{i}")
            # Heuer: rank by the evidence the model itself marked inconsistent (matrix "I" cells, else
            # its inconsistent list), never by a free-floating number it may have copied from the example.
            cells = [str(row.get(hid) or "").strip().upper()[:1] for row in matrix.values() if isinstance(row, dict)]
            own = float(sum(c == "I" for c in cells)) if any(cells) else float(len(inconsistent))
            try:
                stated = float(h["inconsistency_score"]) if h.get("inconsistency_score") is not None else None
            except (TypeError, ValueError):
                stated = None
            hyps.append({
                "id": hid,
                "hypothesis": text,
                "likelihood": tc.normalize_wep(h.get("likelihood")),  # None when the model gave none
                "confidence": tc.normalize_confidence(h.get("confidence") or "low"),
                "consistent": consistent,
                "inconsistent": inconsistent,
                "inconsistency_score": own,
                **({"model_inconsistency_score": stated} if stated is not None and stated != own else {}),
                "notes": str(h.get("notes") or h.get("rationale") or "").strip(),
            })
        hyps.sort(key=lambda h: h["inconsistency_score"])
        extra = {
            "matrix": matrix,
            "most_diagnostic_evidence": self._str_list(payload.get("most_diagnostic_evidence")),
            "conclusion": str(payload.get("conclusion") or "").strip(),
            "evidence_items": {f"E{i}": e for i, e in enumerate(evidence, 1)},
        }
        bands = [tc.wep_range(h["likelihood"]) for h in hyps if h["likelihood"]]
        if len(bands) >= 2 and sum(lo for lo, _ in bands) > 1.0:
            extra["likelihood_warning"] = (
                "ACH: the likelihoods given to mutually exclusive hypotheses ("
                + ", ".join(f"{h['id']} {h['likelihood']}" for h in hyps if h["likelihood"])
                + ") cannot all hold (their lower bounds sum above 100%); treat them as a ranking, not probabilities.")
        return hyps, extra

    # --------------------------------------------------------------------- RUN
    def run(
        self,
        docs: Iterable[Document],
        *,
        lens: "str | Lens" = "general",
        query: str = "",
        case: str = "default",
        use_vision: bool | None = None,
        extract_entities: bool | None = None,
        ach: bool | None = None,
        max_chunks: int | None = None,
        verify: bool | None = None,
        critic: bool | None = None,
        ground_cves: bool | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Report:
        t0 = time.monotonic()
        s = self.settings
        lens_obj = get_lens(lens or s.get("lens") or "general")
        use_vision = bool(s["use_vision"] if use_vision is None else use_vision)
        want_entities = bool(s["extract_entities"] if extract_entities is None else extract_entities)
        want_ach = bool(s["ach"] if ach is None else ach)
        want_verify = bool(s["verify"] if verify is None else verify)
        want_critic_llm = bool(s["critic_llm"] if critic is None else critic)
        want_grounding = bool(s["ground_cves"] if ground_cves is None else ground_cves)
        max_chunks = int(s["max_chunks"] or 0) if max_chunks is None else int(max_chunks)
        errors: list[str] = []
        gaps: list[str] = []
        cite_stats: Counter = Counter()

        # 1. dedupe -------------------------------------------------------------
        incoming = list(docs or [])
        seen: set[str] = set()
        unique: list[Document] = []
        for d in incoming:
            fp = d.fingerprint()
            if fp not in seen:
                seen.add(fp)
                unique.append(d)
        self._emit(progress, f"[{case}] {len(unique)} unique documents ({len(incoming) - len(unique)} duplicates removed)")

        if not unique:
            return Report(
                case=case, lens=lens_obj.name, query=query,
                bluf="No documents were available for analysis; no judgments can be made.",
                key_judgments=[], body_markdown="_No documents were collected._",
                gaps=["No collection: the source returned no documents for this tasking."],
                stats={"documents": 0, "unique_documents": 0, "chunks": 0},
                meta={"errors": [], "engine_version": __version__, "lens_title": lens_obj.title},
            )

        # 1b. computed evidence signals (deterministic, no LLM) ---------------------------
        sig: dict[str, Any] = {"signals": [], "clusters": [], "errors": [], "detectors": {}}
        if s.get("signals", True):
            try:
                sig = compute_signals(unique, settings=s, case=case)
            except Exception as e:  # noqa: BLE001 - compute_signals already isolates detectors
                errors.append(f"signals: {type(e).__name__}: {e}")
        errors += sig["errors"]
        run_signals: list[dict] = sig["signals"]
        if run_signals:
            self._emit(progress, f"computed {len(run_signals)} evidence signal(s): {sig['detectors']}")

        # 2. vision -------------------------------------------------------------
        images_described = 0
        if use_vision:
            self._emit(progress, "describing imagery (vision role)")
            try:
                images_described = describe_media(unique, self.router, max_images=int(s["max_images"]),
                                                  max_workers=int(s["max_concurrency"]))
            except Exception as e:  # noqa: BLE001
                errors.append(f"vision: {type(e).__name__}: {e}")
            vision_errs = sum(len(d.meta.get("vision_errors", [])) for d in unique)
            if vision_errs:
                errors.append(f"vision: {vision_errs} image(s) could not be described")
                gaps.append(f"{vision_errs} image(s) could not be analysed; imagery-borne content may be missed.")

        # 3. indicators (regex, no LLM) ----------------------------------------
        indicators = extract_indicators(unique)
        self._emit(progress, f"extracted {len(indicators)} indicators")
        try:
            annotate_indicators(indicators, self._warninglists)  # benign-infrastructure warninglists (no drops)
        except Exception as e:  # noqa: BLE001
            errors.append(f"warninglists: {type(e).__name__}: {e}")
        grounding: dict | None = None
        if want_grounding and any(r.get("type") == "cve" for r in indicators):
            self._emit(progress, "grounding CVEs in CISA KEV / FIRST EPSS")
            try:
                grounding = _ground_cves(indicators, config=self.config)
                errors += [f"refdata: {e}" for e in grounding.get("errors") or []]
            except Exception as e:  # noqa: BLE001 - grounding is best-effort
                errors.append(f"refdata: {type(e).__name__}: {e}")
        reference_facts = reference_facts_block(grounding) if grounding else ""
        ioc_summary = summarize_indicators(indicators)

        # 4. chunk ----------------------------------------------------------------
        try:
            ctx = int(self.router.context_window("analyst"))
        except Exception:  # noqa: BLE001
            ctx = 32000
        budget = max(512, min(int(s["chunk_tokens"]), int(ctx * 0.6)))
        to_chunk, n_collapsed = unique, 0
        if s.get("collapse_near_dups", True) and sig["clusters"]:
            try:
                to_chunk, n_collapsed = collapse_near_duplicates(unique, sig["clusters"],
                                                                 min_size=int(s["near_dup_min_size"]))
            except Exception as e:  # noqa: BLE001
                errors.append(f"signals: collapse failed: {type(e).__name__}: {e}")
                to_chunk, n_collapsed = unique, 0
        if n_collapsed:
            self._emit(progress, f"collapsed {n_collapsed} near-duplicate copies into their cluster representatives")
        chunks, dropped = self._select_and_chunk(to_chunk, budget, max_chunks, query=query)
        analysed = [d for c in chunks for d in c]
        if dropped:
            relevance = bool(query and query.strip() and s.get("relevance_selection", True))
            msg = (f"{len(dropped)} lower-priority documents were not analysed (max_chunks={max_chunks}); "
                   + ("selection favoured content relevant to the query, then the most engaged and most recent."
                      if relevance else "selection favoured the most engaged and most recent content."))
            log.warning(msg)
            gaps.append(msg)
            self._emit(progress, msg)
        known = {doc_ref(d) for d in unique}
        by_id: dict[str, str] = {}
        for d in unique:
            by_id.setdefault(d.id, doc_ref(d))
            by_id.setdefault(d.ref if hasattr(d, "ref") else doc_ref(d), doc_ref(d))
        known |= {x["id"] for x in run_signals}
        if grounding:
            known |= {src["id"] for src in grounding.get("sources") or []}  # ref:cisa-kev / ref:first-epss

        # 5. MAP (parallel) ---------------------------------------------------------
        total = len(chunks)
        self._emit(progress, f"map: {total} chunk(s), budget {budget} tokens each")
        done = 0

        from .coordination import case_key, pseudonymize

        author_key = case_key(case)

        def job(i_chunk: tuple[int, list[Document]]) -> dict:
            i, chunk = i_chunk
            return self._map_chunk(i, total, chunk, lens_obj, query, budget, indicators, want_entities, known, by_id,
                                   computed_signals=run_signals, reference_facts=reference_facts,
                                   citation_check=want_verify, author_key=author_key)

        chunk_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=max(1, min(int(s["max_concurrency"]), total))) as pool:
            for res in pool.map(job, enumerate(chunks, 1)):
                done += 1
                chunk_results.append(res)
                if res.get("error"):
                    errors.append(res["error"])
                if res.get("entity_error"):
                    errors.append(res["entity_error"])
                cite_stats["unverifiable_citations"] += res.get("unverifiable_citations", 0)
                self._emit(progress, f"map {done}/{total} done" + (" (failed)" if res.get("error") else ""))
        failed_chunks = [c["index"] for c in chunk_results if c.get("error")]
        if failed_chunks:
            gaps.append(f"Analysis failed for {len(failed_chunks)} of {total} chunk(s) "
                        f"({', '.join(map(str, failed_chunks))}); judgments rest on partial coverage.")

        # 6. entities -----------------------------------------------------------------
        entities: list[Entity] = []
        if want_entities:
            entities = merge_entities(e for c in chunk_results for e in c["entities"])[: int(s["max_entities"])]

        # 7. source grading & sources ------------------------------------------------------
        cited: Counter = Counter()
        for c in chunk_results:
            for f in c["findings"]:
                cited.update(f.evidence)
            cited.update(c["notable_sources"])
        grades_by_ref: dict[str, str] = {}
        for d in analysed:
            g = d.meta.get("grade") or d.meta.get("admiralty") or tc.default_source_grade(d.platform, d.source, d.url)
            rel, cred = tc.parse_grade(g)
            grades_by_ref[doc_ref(d)] = f"{rel}{cred}"
        grade_counts = tc.grade_summary(grades_by_ref.values())

        times = [d.created_at for d in analysed if d.created_at]
        stats: dict[str, Any] = {
            "documents": len(incoming),
            "unique_documents": len(unique),
            "duplicates_removed": len(incoming) - len(unique),
            "analyzed_documents": len(analysed),
            "dropped_docs": len(dropped),
            "chunks": total,
            "failed_chunks": len(failed_chunks),
            "chunk_token_budget": budget,
            "tokens_in": sum(c.get("tokens", 0) for c in chunk_results),
            "time_range": {"start": _utc_iso(min(times)) if times else None,
                           "end": _utc_iso(max(times)) if times else None},
            "platforms": dict(Counter(d.platform for d in analysed).most_common()),
            "top_sources": dict(Counter(f"{d.platform}/{d.source}" for d in analysed).most_common(20)),
            "distinct_authors": len({(d.platform, d.author) for d in analysed if d.author}),
            "media_items": sum(len(d.media) for d in analysed),
            "images_described": images_described,
            "indicators": ioc_summary.get("counts", {}),
            "entities": len(entities),
            "source_grades": grade_counts,
            "collapsed_docs": n_collapsed,
            "signals": sig["detectors"],
        }
        profile: dict | None = None
        try:
            profile = temporal_profile(unique)
            stats["volume_profile"] = describe_profile(profile)
        except Exception as e:  # noqa: BLE001
            errors.append(f"temporal_profile: {type(e).__name__}: {e}")

        # 8. REDUCE ---------------------------------------------------------------------
        all_findings = [f for c in chunk_results for f in c["findings"]]
        reduce_payload: dict = {}
        prompt_stats = {k: stats[k] for k in ("unique_documents", "analyzed_documents", "dropped_docs",
                                              "collapsed_docs", "chunks", "failed_chunks", "time_range",
                                              "platforms", "top_sources", "distinct_authors",
                                              "images_described", "volume_profile") if k in stats}
        if all_findings or any(c["summary"] for c in chunk_results):
            self._emit(progress, "reduce: synthesising final assessment")
            try:
                reduce_payload = self._reduce(lens_obj, query, chunk_results, entities, ioc_summary,
                                              prompt_stats, grade_counts, computed_signals=run_signals,
                                              reference_facts=reference_facts, citation_check=want_verify)
            except Exception as e:  # noqa: BLE001
                log.warning("reduce failed: %s", e)
                errors.append(f"reduce: {type(e).__name__}: {e}")
                gaps.append("Final synthesis step failed; key judgments are the highest-confidence chunk-level "
                            "findings and have not been reconciled across chunks.")
        else:
            gaps.append("No chunk produced usable analysis; no key judgments can be offered.")

        key_judgments = [f for f in (self._finding(x, known, by_id, cite_stats)
                                     for x in (reduce_payload.get("key_judgments") or [])) if f]
        if not key_judgments and all_findings:
            rank = {"high": 0, "moderate": 1, "low": 2}
            key_judgments = sorted(all_findings, key=lambda f: (rank.get(f.confidence, 3), -len(f.evidence)))[:7]
        bluf = str(reduce_payload.get("bluf") or "").strip()
        if not bluf:
            if key_judgments:
                bluf = ("Automated synthesis was unavailable; the most strongly supported chunk-level findings are: "
                        + "; ".join(f.title for f in key_judgments[:3]) + ". Overall confidence: low.")
            else:
                bluf = "The analysis could not produce judgments from the collected material (see gaps)."
        body = str(reduce_payload.get("body_markdown") or "").strip()
        if not body:
            parts = ["## Chunk-level analysis", ""]
            for c in sorted(chunk_results, key=lambda c: c["index"]):
                if c["summary"]:
                    parts += [f"### Chunk {c['index']}", "", c["summary"], ""]
                for f in c["findings"]:
                    ev = " ".join(f"[{e}]" for e in f.evidence)
                    lk = f"; {f.likelihood}" if f.likelihood else ""
                    parts.append(f"- **{f.title}** ({f.confidence} confidence{lk}): {f.summary} {ev}".rstrip())
                parts.append("")
            body = "\n".join(parts).strip()
        assumptions = self._str_list(reduce_payload.get("assumptions"))
        gaps = self._str_list(reduce_payload.get("gaps")) + gaps
        watch = self._str_list(reduce_payload.get("indicators_to_watch") or reduce_payload.get("indicators"))

        # 9. ACH ------------------------------------------------------------------------
        hypotheses: list[dict] = []
        ach_meta: dict = {}
        if want_ach:
            self._emit(progress, "ACH: evaluating competing hypotheses")
            try:
                hypotheses, ach_meta = self._ach(lens_obj, query, key_judgments, chunk_results, known, by_id,
                                                 computed_signals=run_signals)
                if ach_meta.get("likelihood_warning"):
                    gaps.append(ach_meta["likelihood_warning"])
            except Exception as e:  # noqa: BLE001
                log.warning("ACH failed: %s", e)
                errors.append(f"ach: {type(e).__name__}: {e}")
                gaps.append("Analysis of Competing Hypotheses could not be completed.")

        # 10. assemble ------------------------------------------------------------------
        for f in key_judgments:
            cited.update(f.evidence)
        by_ref = {doc_ref(d): d for d in analysed}
        sources = []
        for ref, d in by_ref.items():
            sources.append({
                "id": ref,
                "platform": d.platform,
                "source": d.source,
                "url": d.url,
                # keyed pseudonym, never the handle: sources[] would otherwise map every
                # pseudonymised coordination network (sig: doc_refs) back to account names
                "author": pseudonymize(d.platform, d.author, author_key) if d.author else None,
                "created_at": _utc_iso(d.created_at),
                "title": (d.title or d.text or "")[:140],
                "grade": grades_by_ref.get(ref, "F6"),
                "cited": cited.get(ref, 0),
            })
        sources.sort(key=lambda r: (-r["cited"], -by_ref[r["id"]].engagement()))

        try:
            usage = self.router.usage_summary()
        except Exception:  # noqa: BLE001
            usage = {}
        stats["llm_usage"] = usage
        stats["unverifiable_citations"] = cite_stats["unverifiable_citations"]
        stats["elapsed_s"] = round(time.monotonic() - t0, 3)

        signals = Counter(sig.strip() for c in chunk_results for sig in c["signals"] if sig.strip())
        meta: dict[str, Any] = {
            "engine_version": __version__,
            "lens_title": lens_obj.title,
            "lens_description": lens_obj.description,
            "key_questions": list(lens_obj.key_questions),
            "errors": errors,
            "indicators_to_watch": watch,
            "signals": [sig for sig, _ in signals.most_common(25)],
            "notable_sources": [r for r, _ in Counter(r for c in chunk_results for r in c["notable_sources"]).most_common(15)],
            "chunks": [{"index": c["index"], "docs": len(c["doc_ids"]), "tokens": c.get("tokens", 0),
                        "summary": c["summary"], "findings": len(c["findings"]), "error": c.get("error")}
                       for c in sorted(chunk_results, key=lambda c: c["index"])],
            "settings": {"use_vision": use_vision, "extract_entities": want_entities, "ach": want_ach,
                         "max_chunks": max_chunks, "chunk_tokens": budget},
            "dropped_doc_ids": [doc_ref(d) for d in dropped][:500],
            "degraded": bool(errors),
        }
        if ach_meta:
            meta["ach"] = ach_meta
        meta["computed_signals"] = run_signals
        meta["near_dup_clusters"] = sig["clusters"][:200]
        if run_signals:
            meta.setdefault("extra_sections", []).append(
                {"id": "evidence_signals", "title": "Evidence Signals", "markdown": signals_markdown(run_signals)})
        if profile is not None:
            meta["temporal_profile"] = profile
        if lens_obj.name in ("influence", "narrative", "community") and s.get("coordination") is not False:
            try:
                from .coordination import coordination_graph
                from .signals import detector_options

                meta["coordination_graph"] = coordination_graph(
                    unique, clusters=sig["clusters"], case=case, **detector_options(s.get("coordination")))
            except Exception as e:  # noqa: BLE001
                errors.append(f"coordination_graph: {type(e).__name__}: {e}")

        report = Report(
            case=case,
            lens=lens_obj.name,
            query=query,
            bluf=bluf,
            key_judgments=key_judgments,
            body_markdown=body,
            entities=entities,
            indicators=indicators,
            assumptions=assumptions,
            gaps=list(dict.fromkeys(gaps)),
            hypotheses=hypotheses,
            sources=sources,
            stats=stats,
            meta=meta,
        )

        # 11. post-synthesis checks (each mutates the report in place and degrades gracefully) ------------------
        if want_verify:  # WS1: does each cited source actually support the judgment?
            self._emit(progress, "verifying key-judgment citations against their sources")
            try:
                verify_report(report, unique, router=self.router, use_llm=bool(s["verify_llm"]),
                              max_pairs=int(s["verify_max_pairs"]), max_workers=int(s["max_concurrency"]))
            except Exception as e:  # noqa: BLE001
                log.warning("verify failed: %s", e)
                report.meta["errors"].append(f"verify: {type(e).__name__}: {e}")
        if grounding:  # WS5: exploitation claims vs KEV/EPSS, reference sources, Reference facts section
            try:
                apply_grounding(report, grounding, cited=cited)
            except Exception as e:  # noqa: BLE001
                log.warning("reference grounding failed: %s", e)
                report.meta["errors"].append(f"refdata: {type(e).__name__}: {e}")
        if s["critique"]:  # WS4: tradecraft lint (+ optional LLM red team); revisions only ever lower confidence
            self._emit(progress, "tradecraft critic: reviewing the assessment")
            try:
                review_report(report, unique, router=self.router, use_llm=want_critic_llm,
                              stale_days=s["stale_days"])
            except Exception as e:  # noqa: BLE001
                log.warning("critic failed: %s", e)
                report.meta["errors"].append(f"critic: {type(e).__name__}: {e}")
        report.meta["degraded"] = bool(report.meta.get("errors"))
        report.meta["settings"].update({"signals": bool(s.get("signals", True)), "verify": want_verify,
                                        "critique": bool(s["critique"]), "critic_llm": want_critic_llm,
                                        "ground_cves": want_grounding})
        try:
            report.stats["llm_usage"] = self.router.usage_summary()  # now includes verifier/critic calls
        except Exception:  # noqa: BLE001
            pass
        report.stats["elapsed_s"] = round(time.monotonic() - t0, 3)
        self._emit(progress, f"done in {report.stats['elapsed_s']}s ({len(report.key_judgments)} key judgments, "
                             f"{len(report.meta['errors'])} error(s))")
        return report


def analyze(docs: Iterable[Document], config: Any = None, router: Any = None, **kwargs: Any) -> Report:
    """One-shot convenience: build router/engine from config and run."""
    if config is None:
        from ..config import load_config

        config = load_config()
    if router is None:
        from ..llm import LLMRouter

        router = LLMRouter.from_config(config)
    return AnalysisEngine(router, config).run(docs, **kwargs)


__all__ = ["AnalysisEngine", "analyze", "DEFAULT_SETTINGS"]
