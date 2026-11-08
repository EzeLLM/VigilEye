# VigilEye and the open-source intelligence tooling landscape

_Compiled 2026-11-08 against VigilEye 2.0.0 (commit 6fe6b36)._

## Scope

VigilEye analyses public information about topics, communities, narratives, events, public organisations and published threat reporting.

This survey leaves out features that target, identify, locate or profile private individuals. It also leaves out active scanning or probing of systems. Where a surveyed tool centres on either, it is marked **OOS** (out of scope) and only its other ideas are taken.

## Verification legend

| Mark | Meaning |
|---|---|
| **W** | Checked against the tool's current README, docs or live endpoint by the survey agents (2026-11). |
| **W+** | Checked again in this session (2026-11-08). The live fetch is listed in the verification log at the end. |
| **K** | From prior knowledge, not verified on the web in this survey. Confirm before relying on exact identifiers. |

---

## 1. Capability matrix

`Y` means first-class. `P` means partial, manual or indirect. `-` means absent. `N` means planned for the next VigilEye wave, after this plan.

| Capability | MISP | OpenCTI | YETI | IntelOwl | 4CAT | CNT / CooRTweet | OSoMe (Hoaxy, Coordiscope) | Auto Archiver | GPT Researcher | STORM / Co-STORM | Open Deep Research | PaperQA2 | MiniCheck / FACT | **VigilEye now** | **VigilEye + plan** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Finished analytic product (BLUF, key judgments, WEP, confidence) | P (Event Reports, estimative-language tags) | P (Note likelihood/assessment) | - | - | - | - | - | - | P (prose, no WEP) | - (encyclopedic) | P (prose) | P (cited answer) | - | **Y** | **Y** |
| Source reliability and credibility grading | P (admiralty-scale tags) | Y (reliability A-F, confidence 0-100) | - | - | - | - | - | - | - | - | - | P (retraction check) | - | P (platform defaults) | **Y** (computed from independent origins) |
| Alternative hypotheses or dissent | P (MISPOpinion) | P (opinion SDO) | - | - | - | - | - | - | P (reviewer agent) | P (perspectives) | P (think_tool) | - | - | Y (ACH) | **Y** (ACH and red-team critic) |
| Claim-level citation verification | - | - | - | - | - | - | - | - | - | - | - | P (RCS relevance, ContraCrow) | **Y** | P (checks only that the ref exists) | **Y** |
| Near-duplicate and copypasta detection | - | - | - | - | P (exact only) | Y (co_tweet, co_similar_tweet) | P (hashtag-sequence traces) | P (PDQ hash) | - | - | - | - | - | P (fingerprint dedupe) | **Y** |
| Coordination detection with a null model | - | - | - | - | P (co-tag networks) | P (no null model) | Y (Coordiscope) | - | - | - | - | - | - | - (LLM guesses from text) | **Y** (permutation baseline) |
| Burst and temporal anomaly detection | - | - | - | - | P (histograms, RankFlow) | P (flag_speed_share) | P (Hoaxy trends) | - | - | - | - | - | - | - | **Y** |
| Cross-case correlation | Y (correlation engine) | Y (23 inference rules) | Y (link and tag graphs) | P (pivots) | - | - | - | - | - | - | - | - | - | - | N |
| False-positive warninglists | Y | - | - | - | - | - | - | - | - | - | - | - | - | P (hard-coded benign domains) | **Y** (built-in lists and MISP format) |
| Authoritative vulnerability grounding (KEV, EPSS) | P (modules) | Y (x_opencti_cisa_kev, epss) | - | P (passive analyzers) | - | - | - | - | - | - | - | - | - | P (NVD link only) | **Y** |
| ATT&CK and DISARM vocabulary | Y (galaxies) | Y | P | - | - | - | - | - | - | - | - | - | - | P (regex T-IDs) | P (Navigator layer); validation N |
| STIX 2.1 / TAXII 2.1 interop | Y (misp-stix) | Y (native) | P | P (push connectors) | - | - | - | - | - | - | - | - | - | - | **Y** (STIX export); TAXII N |
| Evidence preservation, hashing, chain of custody | - | - | - | - | P (stored datasets) | - | - | Y (hash, RFC 3161, WACZ) | - | P (saved artifacts, K) | - | - | - | P (raw JSON kept, unhashed) | N |
| Iterative deep-research loop | - | - | - | P (playbooks) | - | - | - | - | Y | Y | Y | Y | - | - (collect, then analyse) | N |
| Hybrid retrieval over the case corpus | - | P (search) | - | - | P (search) | - | - | - | P (BM25 filter) | P (VectorRM) | - | Y | - | - (engagement plus recency) | N |
| Runs offline or on local models | Y (no LLM) | Y | Y | P | Y | Y | - (hosted) | Y | P | P (LiteLLM) | P | P | Y | **Y** (mock://, any /v1) | **Y** |
| Lightweight deploy (one process plus SQLite) | - (PHP/MySQL) | - (ES, Redis, RabbitMQ, MinIO) | P (ArangoDB) | - (Django stack) | - (Docker, Postgres) | Y | - | P | Y | Y | P | P | Y | **Y** | **Y** |

All cells come from W-verified features in the survey, except the cells marked K.

---

## 2. What each tool does best, and what VigilEye takes from it

### 2.1 CTI platforms and interop standards

| Tool | Best at | What VigilEye takes | Scope | Verified |
|---|---|---|---|---|
| **MISP** 2.5 / PyMISP 3.0.0 | Sharing at scale. Its correlation engine does exact, CIDR and ssdeep matching with an over-correlation cap. misp-warninglists (string, substring, hostname, cidr, regex) flag false positives without deleting them. Machine-tag taxonomies cover admiralty-scale, estimative-language and information-origin. It publishes static feeds (manifest.json, `<uuid>`.json, hashes.csv). | Warninglist annotation (this wave). Later: cross-case correlation with an over-correlation cap, and MISP event and feed export. | In scope | W, W+ (warninglist format) |
| **OpenCTI** (Community Edition, pycti) | A native STIX 2.1 knowledge graph. Deterministic UUIDv5 ids make re-imports merge instead of duplicate. Reliability and confidence are kept as separate dimensions. The rule that a lower confidence wins blocks overwrites. 23 materialised inference rules are kept apart from asserted facts. Vulnerability fields: `x_opencti_cisa_kev`, `x_opencti_epss_score`, CVSS v3/v4. | Deterministic ids, Note `likelihood`/`note_types` mapping, and KEV/EPSS fields in the STIX export (this wave). Later: inference rules marked 'inferred, not stated'. | In scope | W |
| **YETI** v2 | Tag ontology with `produces`/`replaces` and tag expiry (a lightweight decay model). Indicators run as matchers. DFIQ stores investigative questions. | Tag-expiry idea for watch cases (next wave). | In scope (aimed at DFIR) | W |
| **IntelOwl** | A per-plugin `maximum_tlp` ceiling. Playbooks and pivots. DataModel normalisation across analyzers. | TLP- and PAP-aware model routing (next wave: endpoint `max_tlp`). | **Partly OOS**: its active analyzers (Nuclei, scanners, honeyclients, detonation) are excluded | W |
| **TheHive 6 + Cortex** | Promoting alerts to cases. Per-observable TLP and PAP. | The PAP concept, and an optional alert-JSON export target (next wave). | **Mostly OOS**: responders are active response, and TheHive 5/6 is no longer open source | W |
| **STIX 2.1 / TAXII 2.1** | Graph standard. report, note and opinion containers. A sighting SRO. Appendix A confidence mappings (WEP 5..95, L/M/H 15/50/85). Deterministic SCO ids under namespace `00abedb4-aa42-466c-9c01-fed23315a9b7`. | STIX 2.1 bundle export (this wave). TAXII read endpoint (next wave). | In scope | W. TLP 2.0 marking ids: K |
| **MITRE attack-stix-data** | Versioned ATT&CK STIX 2.1 bundles (enterprise, mobile, ICS) with `index.json`. Revocation and deprecation instead of deletion. Group aliases resolve to G-IDs. | ATT&CK Navigator layer output (this wave). Validating T-IDs against the bundle and following `revoked-by` (next wave). | In scope | W (v20.1 counts), W+ (paths, index.json) |
| **CISA KEV** | Authoritative list of CVEs exploited in the wild. About 1.8 MB of JSON, diffable by `catalogVersion`. | Grounding and corroborating exploitation claims (this wave). | In scope (keyless) | W+ (live feed, fields) |
| **NVD 2.0 + FIRST EPSS + cvelistV5** | CVSS v3.1/v4 and SSVC from NVD. Exploitation probability from EPSS. Upstream CNA data from cvelistV5 when NVD lags. | EPSS grounding (this wave). NVD incremental cache (next wave). | In scope (EPSS keyless; NVD keyless at about 5 requests per 30 s) | EPSS W+, NVD W (rate limits from third-party sources) |

### 2.2 Disinformation and online-research toolkits

| Tool | Best at | What VigilEye takes | Scope | Verified |
|---|---|---|---|---|
| **4CAT** (DMI) | Chainable processors whose outputs form a provenance tree. GEXF network processors (co-word, co-tag, URL co-occurrence). RankFlow and diachronic embeddings. Zeeschuimer and CrowdTangle importers. | The signal record keeps its parameters and tool version (this wave). GEXF export and 4CAT/Zeeschuimer importers (next wave). | In scope | W |
| **coordination-network-toolkit / CooRTweet / CooRnet** | Platform-neutral co-sharing detection within a time window (co_link, co_tweet, co_similar_tweet). Multimodal object ids (URL:, TEXT:, PHASH:). Edge weight, avg_time_delta and symmetry. | Content-level coordination detection, plus the **null model** these tools lack (this wave). | In scope at the network-behaviour level, with pseudonymised accounts | W |
| **OSoMe**: Hoaxy, Coordiscope, OSoMeNet | Replayable diffusion cascades. Pacheco et al. bipartite TF-IDF coordination. Hashtag-sequence traces. | Hashtag-sequence keys for copypasta (this wave). Cascades (next wave). | Botometer-style per-account scoring is **OOS** | W |
| **Junkipedia** | Curated cross-platform monitoring lists. Video transcripts feed the same text pipeline. | Concept only: watch cases (next wave). | Closed platform, gated access | W |
| **GDELT 2.0** | Global news volume, tone and themes every 15 minutes. DOC 2.0 API covers 3 months, keyless. | Baseline for bursts and checks for mainstream pickup (next wave). | In scope | W. DOC API returned HTTP 503 on recheck on 2026-11-08 |
| **Bellingcat Auto Archiver** | Evidentiary enrichers: hash, RFC 3161 timestamp, OpenTimestamps, WACZ, Wayback, PDQ, Whisper. | Hash-chained custody log (next wave). | In scope | W |
| **InVID / WeVerify / vera.ai plugin** | Splitting video into keyframes, reverse-search links, a database of known fakes. | pHash reuse detection (next wave). | Face matching and person identification are **OOS** | W (feature list partly K) |
| **DFRLab FIAT + DISARM** | Attribution scored separately from impact (Breakout Scale). DISARM supplies influence-operation TTP identifiers. | Rule in the critic: coordination claims need a computed signal (this wave). Attribution rubric and DISARM tagging (next wave). | In scope | W, W+ (DISARM ships xlsx/CSV/SQLite. The repo states the licence inconsistently, CC-BY-4.0 in the README and CC-BY-SA-4.0 in the sidebar) |
| **Berkeley Protocol** | Evidentiary standard: hash at collection, keep originals apart from working copies, three-part verification (source, technical, content). The verification step includes 'first copy found online' and exculpatory information. | Independent-origin counting in the critic (this wave). Custody log and Annex IV manifest (next wave). | In scope | W |

### 2.3 LLM research agents and verification

| Tool | Best at | What VigilEye takes | Scope | Verified |
|---|---|---|---|---|
| **GPT Researcher** (+ gptr-mcp) | Planner, then parallel executors, then publisher. Deep mode with breadth x depth x concurrency. A reviewer/revisor loop. MCP on both client and server side. | Red-team critic loop (this wave). Investigate loop and MCP server (next wave). | In scope | W |
| **STORM / Co-STORM** | Questioning guided by perspectives. A moderator raises evidence that was retrieved but not yet used. A mind map prunes nodes that lack support. | Multi-perspective KIQ generation (next wave). | In scope | W (saved intermediate artifacts: K) |
| **LangChain Open Deep Research** | A research brief that acts as a contract. Supervisor and worker subgraphs. Explicit budgets. Truncate-and-retry on token limits. | Budget governor and investigate loop (next wave). | In scope | W |
| **PaperQA2** | Retrieve, Contextual-Summarize, then score (RCS) as evidence objects. Hybrid sparse plus dense retrieval. ContraCrow contradiction search. | Searching passages inside the cited document for verification (this wave). RCS Evidence objects with E-ids (next wave). | In scope | W |
| **MiniCheck / LLM-AggreFact / DeepResearch Bench FACT** | Small grounding models score whether a document entails a claim. FACT measures `citation_accuracy` and `effective_citations`. | A verifier role (any OpenAI-compatible endpoint, for example Ollama `bespoke-minicheck`), the FACT metrics, and a deterministic quote guard (this wave). | In scope | W |
| **Perplexica / Vane, SearXNG, Local Deep Research** | Speed, balanced and quality compute modes. SearXNG keyless metasearch. Per-run egress scopes (Public only, Private only, Strict). | `searx:` collector, `ask` endpoint, and egress policy per case (next wave). | In scope | W (some SearXNG params: K) |
| **Hybrid RAG** (SQLite FTS5, bm25s, sqlite-vec, RRF) | Lexical plus dense retrieval in a single file, with a BM25-only degraded mode. | Hybrid retrieval replacing engagement-plus-recency selection (next wave). | In scope | W |

---

## 3. Where VigilEye stands

### Strengths no surveyed tool combines

1. **Tradecraft-native output.** Every run emits a BLUF, key judgments with ICD 203 estimative language and analytic confidence, Admiralty grades, assumptions, gaps and optional ACH. CTI platforms store facts but leave assessments as free text. Research agents write cited prose but give no calibrated likelihood or confidence.
2. **Model-agnostic routing.** Analyst, extractor, vision and synthesizer roles run as fallback chains over any OpenAI-compatible endpoint. A deterministic `mock://` provider runs the whole pipeline offline and in CI.
3. **Light footprint.** The core needs four dependencies (openai, httpx, PyYAML, Jinja2), one SQLite case store and a stdlib HTTP API. MISP, OpenCTI, IntelOwl and 4CAT each need a multi-service stack.
4. **Guardrails built in.** It does not extract e-mails or phone numbers. The vision prompt forbids identifying people by face. Collectors provide no de-anonymisation.

### Weaknesses this plan addresses (ranked)

1. **Citations are only checked structurally.** `AnalysisEngine._resolve_ref` checks that a cited ref exists, not that it supports the sentence (the FACT and MiniCheck gap).
2. **No computed evidence.** The influence, narrative and community lenses ask the LLM to *infer* copypasta, bursts and coordination by reading chunks. No measured signal backs a judgment, and the ACH matrix scores "organic vs coordinated" on impressions.
3. **No independence model.** Ten copies of one syndicated claim count as ten sources. Admiralty credibility is a platform default (`F6`, `C3`), never computed from corroboration.
4. **No self-critique.** Nothing checks for confidence overreach, single-source judgments, circular reporting or WEP/confidence mismatches before a report ships.
5. **No authoritative grounding.** CVE claims are not checked against CISA KEV or EPSS, so the model can invent exploitation status. Benign infrastructure is suppressed only by a 14-domain hard-coded list.
6. **No interop.** Nothing can be exported to OpenCTI, MISP or ATT&CK Navigator.
7. **Next-wave gaps.** These remain after this plan: no cross-case memory or correlation, no evidence preservation or custody, no iterative investigate loop, and selection by engagement and recency rather than relevance.

---

## 4. Build plan

Seven workstreams, ranked by value. Each one is new modules plus tests and can be built in parallel. The lead wires the named hook points.

| # | Workstream | New files | Inspired by |
|---|---|---|---|
| 1 | Claim-level citation verification | `vigileye/analysis/verify.py` | MiniCheck, FACT, PaperQA2 RCS |
| 2 | Computed signals core and copypasta clustering | `vigileye/analysis/signals.py` | 4CAT, CNT co_similar_tweet, OSoMe hashtag sequences |
| 3 | Coordination detection with a permutation null model | `vigileye/analysis/coordination.py` | CooRTweet, CNT, Pacheco et al., CooRnet |
| 4 | Tradecraft critic and independent-origin corroboration | `vigileye/analysis/critic.py` | GPT Researcher reviewer, Berkeley Protocol, DFRLab |
| 5 | Vulnerability grounding (KEV, EPSS) and warninglists | `vigileye/intel/{__init__,refdata,warninglists}.py` | CISA KEV, FIRST EPSS, MISP warninglists, OpenCTI |
| 6 | Temporal bursts | `vigileye/analysis/temporal.py` | Hoaxy, 4CAT RankFlow, Kleinberg |
| 7 | STIX 2.1 bundle and ATT&CK Navigator export | `vigileye/export/{__init__,stix,navigator}.py` | STIX 2.1, OpenCTI, attack-stix-data |

**Shared conventions** that let the workstreams merge without conflicts:

- **Signal record** (plain dict). Fields: `{"id": "sig:<kind>#<n>", "kind", "summary", "value", "baseline", "z", "p_value", "doc_refs": [...], "params": {...}, "detector": "<module>@<version>"}`.
  - The `sig:` ids are added to the engine's known-ref set, so judgments can cite them exactly like `[reddit:t3_abc]`.
  - The existing `_REF_IN_TEXT` regex already matches `[sig:copypasta#1]`.
- **Extra report sections.** Each module appends `{"id", "title", "markdown"}` to `report.meta["extra_sections"]`. One generic hook in `report.render_markdown` and `report.render_html` renders them before Methodology.
- **New SQLite tables**, if any, are created lazily by the owning module through `store._db` under `store._lock`. `store.py` is not edited.
- **New LLM roles** (`verifier`, `critic`) need no router change. `LLMRouter.chain()` already falls back to `analyst` for unknown roles.
- **Existing tests.** `tests/test_analysis.py::test_engine_end_to_end_mock` requires judgment evidence to be a subset of doc refs. `test_engine_max_chunks_drops_docs` requires `analyzed + dropped == 10`. Detectors therefore emit nothing for clusters or components smaller than 3. Near-duplicate collapse applies only to clusters of 3 or more, and is counted in a separate `collapsed_docs` stat.

### Next wave (not in this plan)

- Cross-case correlation index (observables, entity_mentions, edges) with an over-correlation cap.
- Berkeley Protocol custody log: hash-chained `custody_events` with an RFC 3161 timestamp on the manifest.
- Hybrid FTS5 and embedding retrieval with an `embedder` role, replacing engagement-plus-recency selection.
- Gap-driven `vigileye investigate` loop with a budget governor, and a `searx:` collector.
- MISP event and static-feed export. TAXII 2.1 read endpoint.
- ATT&CK and DISARM validation against the bundles. DFRLab-style attribution rubric and Breakout Scale.
- GDELT collector and burst baseline. Narrative clustering. Watch cases with delta reports. GEXF graph export.
- `vigileye eval` harness: FACT metrics, an ICD 203 rubric judge, and a Brier-scored forecast ledger.

---

## 5. Verification log (this session, 2026-11-08)

| Fact | Result |
|---|---|
| KEV feed `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` | Live. Top level: `title`, `catalogVersion` 2026.11.06, `dateReleased`, `count` 1769, `vulnerabilities[]`. Entry fields: `cveID`, `vendorProject`, `product`, `vulnerabilityName`, `dateAdded`, `shortDescription`, `requiredAction`, `dueDate`, `knownRansomwareCampaignUse`, `forensicTriage`, `notes`, `cwes`. |
| EPSS `https://api.first.org/data/v1/epss?cve=...` | Live, keyless. Top level: `status`, `status-code`, `version`, `access`, `total`, `offset`, `limit`, `data[]`. Entries are `{cve, epss, percentile, date}`, with **numbers encoded as strings**. |
| misp-warninglists format | Keys: `name`, `version`, `description`, `matching_attributes`, `list`, optional `type` (string, substring, hostname, cidr, regex; default string) and `category` (false_positive or known). |
| DISARMframeworks repo | Master data in `DISARM_MASTER_DATA/DISARM_FRAMEWORKS_MASTER.xlsx`, generated CSV/SQLite in `generated_files/`, STIX in the separate DISARM_CTI repo. Licence stated inconsistently (CC-BY-4.0 in the README, CC-BY-SA-4.0 in the sidebar). |
| attack-stix-data | STIX 2.1. `enterprise-attack/enterprise-attack.json` always holds the latest release; versioned files sit beside it; `index.json` at the repo root. |
| ATT&CK Navigator layer format 4.5 | Doc URL returned 404. Field names (`name`, `versions.layer`, `domain`, `techniques[].techniqueID/score/comment`) are **K**. |
| STIX TLP 2.0 marking ids | Not confirmed (**K**). The exporter uses the TLP 1.0 ids that ship in the STIX 2.1 spec and `stix2`, with CLEAR mapped onto WHITE as OpenCTI does. |
| GDELT DOC 2.0 | HTTP 503 on recheck. The survey's W verification stands, but the API's availability varies. |
