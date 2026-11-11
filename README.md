# VigilEye 👁️

**An open-source intelligence (OSINT) analysis framework that runs on any OpenAI-compatible model.**

VigilEye collects public content from many platforms and stores it in case files. It enriches the content with indicator extraction, entity resolution and image description, then runs it through a map-reduce analysis pipeline. The output is a finished intelligence product in the style of professional analytic tradecraft:

- a BLUF
- key judgments, each with calibrated estimative language and a confidence level
- evidence cited down to the document
- Admiralty-graded sources
- key assumptions, intelligence gaps and indicators
- optionally, an Analysis of Competing Hypotheses

Every model call goes through the OpenAI Chat Completions protocol, so you can use whichever backend you like: OpenAI, a vLLM server under your desk, Ollama, OpenRouter, Groq, or a private "stealth" model behind your own gateway. Mix them by role and chain fallbacks.

```
 targets ──► collectors ──► case store ──► signals ──► enrichment ──► map ──► reduce ──► ACH ──► checks ──► report
 reddit:r/x   reddit hn rss  (SQLite)      copypasta   IOCs + warn-   analyst  synthe-          verify      md / json / html
 hn:search:x  web mastodon   dedupe        bursts      inglists       per      sizer            citations   stix / navigator
 rss:https:…  bluesky files  llm cache     coordin-    [KEV/EPSS]     chunk,                    [KEV check] + stored
                                           ation       entities       cited                     critic
                                           (no LLM)    vision
                                     ▲
                 LLMRouter: role → [provider/model, fallback, …] — any OpenAI-compatible base_url
```

## Quickstart

```bash
pip install -e ".[reddit,tokens]"
cp .env.example .env            # add OPENAI_API_KEY, or point roles at your own endpoint

# fully offline demo (mock model, no keys needed)
vigileye --config config/offline.yaml run hn:search:ransomware --limit 30

# real analysis
vigileye run reddit:r/cybersecurity hn:search:"CVE-2024-3400" --case panos --lens cyber --ach \
  --query "Is CVE-2024-3400 being actively exploited, and by whom?"
```

Reports land in `export/` as Markdown, JSON and a self-contained HTML page (add `--formats stix,navigator` for a STIX 2.1 bundle and an ATT&CK Navigator layer). They're also saved in the case store (`vigileye reports --case panos`, `vigileye show 1`).

## Models: everything is an OpenAI endpoint

```yaml
providers:
  local:   {base_url: http://localhost:8000/v1, api_key: none}            # vLLM / SGLang / llama.cpp / LM Studio / TGI
  ollama:  {base_url: http://localhost:11434/v1, api_key: ollama}
  stealth:
    base_url: https://llm.internal.example/v1
    headers: {X-Gateway-Token: "${STEALTH_GATEWAY_TOKEN}"}
    verify_tls: /etc/ssl/internal-ca.pem
    proxy: socks5://127.0.0.1:9050
    supports_json_mode: false
roles:
  analyst:     [{provider: stealth, model: ghost-70b, context_window: 65536}, local/Qwen/Qwen2.5-32B-Instruct]
  extractor:   [ollama/qwen2.5:7b]
  vision:      [ollama/llava]
  synthesizer: [{provider: openai, model: gpt-4o, context_window: 128000}]
```

- **Roles.** `analyst` reads chunks in parallel, `extractor` does cheap structured extraction, `vision` describes images and `synthesizer` writes the final product. Two optional roles check the product: `verifier` (citation grounding; falls back to `extractor`, then `analyst`) and `critic` (red-team review with `--critic`; pick a different model family from the synthesizer). `planner` proposes collection targets for `vigileye investigate`. A role you leave out falls back to `analyst`.
- **Embeddings.** The optional `embedder` role is served through `/v1/embeddings` (OpenAI, vLLM `--task embed`, llama.cpp `--embeddings`, Ollama). It never falls back to a chat model; without it, retrieval is lexical only.
- **Fallback chains.** If an endpoint errors or times out, the next entry in the chain is tried automatically.
- **Compatibility shims.**
  - If a server rejects `response_format`, the request is retried without JSON mode.
  - Model output is parsed into JSON robustly, with one repair retry if that fails.
  - Images are sent inline as base64 data URLs to non-OpenAI servers.
- **Pass-through parameters.** Any extra key on a role entry (`top_p`, `seed`, `reasoning_effort`, …) goes straight to the API. Use `extra_body` for server-specific options such as vLLM `guided_json`.
- **Accounting.** Each report records tokens, latency and cost per model; set `pricing` on a provider to get costs. Identical calls are cached in the case store.
- **Supported out of the box** (`config/vigileye.yaml`): OpenAI, local vLLM/llama.cpp/SGLang/LM Studio, Ollama, OpenRouter, Groq, Together, DeepSeek, Mistral, Gemini (OpenAI surface), Azure OpenAI (custom headers and query), a stealth gateway and `mock://`.

Run `vigileye providers --test` to ping every role and print the latency and the model that actually served the request.

## Targets

| Spec | What it collects |
|---|---|
| `reddit:r/osint`, `reddit:u/name`, `reddit:search:query` (legacy `r/x`, `u/x`) | posts (+ comments with `--opt include_comments=true`); uses praw if credentials are set, public JSON otherwise |
| `hn:search:query`, `hn:user:pg`, `hn:item:123` | Hacker News via the Algolia API |
| `rss:https://…` | RSS 2.0 / Atom feeds |
| `web:https://…` | readable page text and images; same-domain crawl with `--opt crawl_depth=1`; respects robots.txt |
| `mastodon:#tag@mastodon.social`, `mastodon:@user@instance`, `mastodon:search:q@instance` | public Mastodon timelines |
| `bluesky:search:query`, `bluesky:@handle.bsky.social` | Bluesky public AppView |
| `file:data.jsonl` / `.json` / `.csv` / `.txt` / `.md` / directory | bring your own corpus |

`vigileye collectors` lists all collectors with examples.

## Lenses

A lens sets the analytic focus and the key questions for a run:

- `general`: situational awareness
- `threat`: security and threat assessment
- `influence`: information operations, coordinated inauthentic behavior and amplification
- `narrative`: narrative and sentiment mapping
- `cyber`: CTI (IOCs, TTPs, ATT&CK, CVEs)
- `market`: brand and competitive intelligence
- `crisis`: event monitoring and verification status
- `community`: profiling a forum or subreddit

To make your own, pass `--lens path/to/lens.yaml` with `name`, `title`, `description`, `focus` and `key_questions`. You can also override any prompt template through `analysis.template_dir`.

## Tradecraft built in

- **ICD 203 analytic standards** are written into every system prompt.
- **Words of Estimative Probability** cover the range from *almost no chance* (1–5%) to *almost certain* (95–99%). Likelihood and confidence are kept separate.
- **Admiralty / NATO source grading** rates reliability A–F and credibility 1–6. Unverified social media defaults to a conservative grade.
- **Evidence discipline.** Every judgment cites document IDs (`[reddit:t3_abc]`), and prompts forbid fabrication.
- **Indicator extraction** handles IPs, domains, URLs, hashes, CVEs, ATT&CK IDs, crypto wallets, hashtags and cashtags. It works on defanged input and defangs values in reports.
- **Analysis of Competing Hypotheses (ACH)** with `--ach`.
- **Graceful degradation.** If a model step fails, the run keeps its partial results and records the failure as a gap instead of crashing.

The layers below wrap the model with deterministic evidence and quality checks. Each analysis layer adds its own report section (just before Methodology) and its numbers to `report.stats`, and each can be turned off in `analysis:`.

### Evidence signals

Before any model call, VigilEye measures the whole case deterministically and turns each finding into a citable signal record, `[sig:<kind>#<n>]`, with its value, baseline, z-score, p-value and member documents.

- **Copypasta / near-duplicates** (`sig:copypasta#n`). Exact copies, MinHash near-duplicates (Jaccard >= `near_dup_threshold`, after stripping URLs, handles, emoji and case) and repeated hashtag runs. By default the map step sees one representative per cluster of `near_dup_min_size` or more, so a flood of copies counts as one voice; `stats.collapsed_docs` records how many were folded in.
- **Bursts** (`sig:burst#n`). Kleinberg burst detection plus a robust rolling z-score over overall volume, hashtags, linked domains and copypasta clusters. `stats.volume_profile` holds a sparkline of activity over the collection window.
- Signals go into the map, reduce and ACH prompts as COMPUTED EVIDENCE. The prompts require judgments about coordination, copypasta or bursts to cite a `sig:` id, or to say none exists and lower confidence.
- `vigileye signals --case NAME [--json]` prints the signals for a case without calling a model.

Settings: `analysis.signals`, `collapse_near_dups`, `near_dup_threshold`, `near_dup_min_size`, `bursts` (`false` disables).

### Coordination detection

`sig:coordination#n` flags groups of accounts that repeatedly share the same URLs, near-duplicate texts or hashtag runs within `window_s` seconds of each other.

- Each candidate network is tested against a permutation null: posts are reassigned to accounts within platform-and-day blocks, so each object's timing and each account's activity level are preserved. The p-value compares the network with the largest network found by chance in each shuffle. Networks with p >= 0.05 are still listed, marked "not significant".
- A single viral link never forms a network: a pair needs `min_weight` distinct co-share occasions, and objects shared more than `max_object_shares` times are skipped and listed.
- Accounts are HMAC-pseudonymised per case (`acct_…`). Set `VIGILEYE_PSEUDONYM_KEY`, because the built-in default key is public. Only handles in `coordination.public_accounts` appear verbatim.
- For the `influence`, `narrative` and `community` lenses, the pseudonymised co-share graph is stored in `meta.coordination_graph`.

Settings: `analysis.coordination` (`window_s` or `"auto"`, `min_weight`, `n_iter`, `public_accounts`; `false` disables).

### Citation verification

After synthesis, each (key judgment, cited document) pair is checked: does the cited source actually say this?

- The judgment is split into claims, and each claim is scored against overlapping passages of the source (BM25-style lexical support). With `verify_llm`, the `verifier` role then labels the pair `supported`, `partial`, `unsupported` or `contradicted`. A verdict counts only if the model quotes the source verbatim; an invented quote downgrades the pair to `unsupported`.
- Unsupported and contradicted citations are removed from the judgment and kept in `meta.citation_removed`. A judgment left with no support drops to low confidence and gains a gap line; partial-only support caps it at moderate.
- `sig:` and `ref:` citations are labelled `computed` and kept.
- `stats.citation_verification` reports `citation_accuracy` (supported ÷ checked) and `effective_citations` (distinct supporting sources).
- The prompts tell the model that every citation will be checked.

On by default. Settings: `analysis.verify`, `verify_llm`, `verify_max_pairs`. CLI: `--verify` / `--no-verify`. With the offline mock provider, only the lexical check runs.

### Tradecraft critic

A rule-based reviewer lints the finished product and records its critiques in an "Analytic review" section.

- **Rules:** `no_evidence`, `single_origin`, `circular_reporting`, `confidence_overreach`, `wep_confidence_mismatch`, `low_grade_sources`, `stale_evidence`, `coordination_claim_without_signal` and `unverified_citation`.
- **Independent origins.** Cited documents are grouped into origins by near-duplicate cluster, shared outbound link, thread and site; authors are never used. Ten reposts of one article count as one origin, and the computed Admiralty credibility follows the number of origins (0 = 6, 1 = 3, 2 = 2, 3 or more = 1).
- Revisions are deterministic and can only lower confidence. Each revision adds a gap line explaining why.
- `--critic` adds an LLM red-team pass (`critic` role) for devil's advocacy and a key-assumptions check. LLM critiques are reported but never applied automatically.

On by default (rule-based). Settings: `analysis.critique`, `critic_llm`, `stale_days`.

### Vulnerability grounding

With `--ground-cves` (or `analysis.ground_cves: true`), every CVE found in the collection is checked against the CISA Known Exploited Vulnerabilities catalog and FIRST EPSS scores.

- The facts go into the prompts as an AUTHORITATIVE REFERENCE FACTS block, citable as `[ref:cisa-kev]` / `[ref:first-epss]`.
- Exploitation claims in the BLUF and key judgments are then checked. A claim about a KEV-listed CVE is corroborated and cites `ref:cisa-kev`. A claim that a KEV-listed CVE is *not* exploited is flagged as contradicted. A claim about an unlisted, low-EPSS CVE adds a gap.
- The results appear in a "Reference facts" section and in `meta.reference_grounding`.
- Both feeds are keyless, and only public CVE ids are sent. Data is cached in `intel.cache_dir` for `intel.ttl_hours`; `intel.offline: true` uses the cache only.
- Off by default, because it downloads reference data. Manage the cache with `vigileye refdata update` and look up one CVE with `vigileye refdata cve CVE-2024-3400`.

Independently of grounding, every indicator is annotated against MISP-format warninglists: RFC 1918/bogon ranges, public DNS resolvers, reserved test domains, link shorteners, major platforms/CDNs and security-reference sites. Matches are marked benign rather than dropped. Point `analysis.warninglists_dir` at a [misp-warninglists](https://github.com/MISP/misp-warninglists) clone to load the full set.

### STIX and ATT&CK Navigator export

`--formats stix,navigator` writes two more files next to the report (`vigileye show ID --format stix|navigator` prints them for a stored report, and the HTTP API serves them at `/v1/reports/{id}/stix` and `/navigator`):

- **`*.stix.json`.** A STIX 2.1 bundle that imports into OpenCTI, or into MISP via misp-stix. It holds a `report` object, key judgments as `note` objects (STIX confidence from the confidence level, OpenCTI likelihood from the WEP term), vulnerabilities, attack-patterns, organisations/intrusion-sets/malware/tools from the entity list, and observables for non-benign indicators.
  - Ids are deterministic (OpenCTI-compatible), so re-exporting the same report updates objects instead of duplicating them.
  - Every domain object carries the TLP marking from `--tlp clear|green|amber|red` (default `output.tlp`).
  - Scope guard: no people, accounts or author strings are exported.
- **`*.layer.json`.** An ATT&CK Navigator layer, scored by how often each technique id appears.

`pip install -e ".[stix]"` installs `stix2`, which the test suite uses to check bundles.

## Evidence and chain of custody

VigilEye hashes every item when it is collected and logs every handling step, following the Berkeley Protocol on Digital Open Source Investigations.

- **Stamps.** At collection, each document gets `meta.evidence`: the SHA-256 of its evidentiary content (`platform, id, source, url, title, text, created_at, media` as canonical JSON, `vigileye-cjson/1`), the SHA-256 of the platform's raw record, `collected_at`, the collector, the target spec and the tool version. A stamp is never overwritten. Re-collecting an unchanged item keeps its first stamp.
- **Custody log.** `collect`, `analyze`, `export`, `amend` and `manifest` events go into an append-only, hash-chained `custody` table in the case database. Each entry hashes the previous one. Database triggers reject `UPDATE` and `DELETE`, and a unique index stops two writers from forking a chain. Each `collect` entry binds the hash of every item it collected. Each `analyze` entry binds the SHA-256 of the stored report.
- **Methodology.** Every report records the chain head at analysis time and how many analysed documents were stamped.
- **Verification.** `vigileye evidence verify --case NAME [--anchor HEAD]` recomputes the chain and re-hashes every stored document and report. It exits 1 and lists the fingerprints of documents whose content or raw record no longer matches its stamp (`mismatched`, `raw_mismatched`), of documents edited and then re-stamped (`unlogged`), and of reports changed without an `amend` entry. Documents collected before stamping are listed as `unstamped` but do not fail the check.
- **Manifest.** `vigileye evidence manifest --case NAME --out case.json` writes an inventory (ref, fingerprint, URL, collected_at, hashes and status per document, plus the chain head) and a `case.json.sha256` sidecar that `sha256sum -c` accepts. Account handles and profile-like URL segments are masked, and authors are left out.
- `vigileye evidence log --case NAME` prints the custody entries.

The chain is tamper-evident, not tamper-proof. Anyone who can write the database can rebuild every hash, and deleting the newest entries leaves a valid shorter chain. Keep the head somewhere else (hand over the manifest and its sidecar) and pass it back as `--anchor` to detect truncation.

Settings: `evidence.custody` (default `true`) and `evidence.actor` (recorded in every entry; default `vigileye`, or `$VIGILEYE_ACTOR`).

## Ask the case

`vigileye ask --case NAME "question"` answers one question from a case's documents, with a citation on every sentence.

- **Retrieval.** Hybrid search over the case: SQLite FTS5 (or a built-in BM25) for lexical matches, plus dense vectors from the `embedder` role when it is configured. The two rankings are fused with reciprocal rank fusion. The index lives next to the store (`cases.index.db`) and is brought up to date with the case on every `ask`, so only new or changed documents are re-indexed and re-embedded. Vectors from different embedding models are never mixed.
- **Answering.** The top `--k` documents (default 12) go to the `synthesizer` role as numbered evidence blocks, fenced with a per-call nonce and marked as untrusted data. The prompt requires a `[platform:id]` citation on every factual sentence, or an explicit `INSUFFICIENT EVIDENCE:` reply.
- **Checking.** Each (sentence, citation) pair is verified like report citations: lexically and, with `analysis.verify_llm`, by the `verifier` role. Unsupported citations and citations of documents that were not retrieved are removed. An answer left with no verified citation is reported as unanswered. Confidence is capped by corroboration: one source, or partial support only, means at most moderate.
- **Privacy.** Authors are never indexed, embedded, prompted or returned; the answer metadata counts them only.
- Flags: `--platform P`, `--no-verify`, `--json`. With the offline mock provider, the answer quotes the top documents verbatim.

When `--query` is set and `max_chunks` forces a cut, `analyze` uses the same ranking to keep the documents most relevant to the query (0.6) before engagement and recency (0.4). Turn this off with `analysis.relevance_selection: false`.

## Investigate

`vigileye investigate "question" --seed SPEC [--seed SPEC …] --case NAME` runs an iterative collection loop: collect the seeds, analyse, let the `planner` role propose new targets for the gaps, check them against the target policy, collect, and repeat.

- **Budgets.** `--rounds` (planner rounds after the seed round, default 3), `--max-new` (targets per round, default 4), `--limit` (documents per target, default 50) and `--max-docs` (case budget, default 2000). The loop also stops when the planner says it is done, no proposal survives, or a round adds no new documents.
- **Approval.** Without `--yes`, each round's policy-approved targets are listed and you choose which to collect. When stdin is not a terminal and `--yes` is not given, every planner target is declined.
- **Planner input.** The planner sees the BLUF, gaps, judgments and indicator counts between nonce markers, as untrusted data. It never sees person or account entity names or indicator values (URLs and indicator hosts are masked in the BLUF, gaps and judgments too), and it cannot target any host that appears as an IP, domain or URL indicator, defanged or not.
- **Provenance.** Interim analyses are lighter (no ACH, LLM critic or vision; JSON only, in `<out>/interim/round-N/`). The final report gets a "Collection plan & provenance" section listing every target, who proposed it and why, and what happened to it. The report is updated in place and the change is logged as an `amend` custody entry.

### Target policy

`vigileye.policy.check_target` decides whether a target may be collected. It has three modes:

| Mode | Used by | Allows |
|---|---|---|
| `auto` | planner proposals | topic-level targets only: `reddit:r/…`, `reddit:search:…`, `hn:search:…`, `hn:item:N`, `rss:https://…`, `web:https://…`, `mastodon:#tag@instance`, `mastodon:search:…@instance`, `bluesky:search:…`. Refused: accounts (`reddit:u/`, `hn:user:`, `@handle`), searches scoped to an account (`from:`, `author:`, `@handle`), profile-page URLs, `file:` |
| `manual` | CLI seeds | any collector form; `file:` is allowed |
| `api` | HTTP API | any collector form except `file:` |

In every mode, a URL-bearing target must be a public http(s) URL: no other scheme, no credentials, and no host that is or resolves to a loopback, private, link-local, multicast, reserved, CGNAT or cloud-metadata address (IPv4, IPv6 and IPv4-mapped forms). Name resolution fails closed. `investigate` refuses account-level seeds, and refused account specs appear in reports only as per-case pseudonyms (`acct_…`). Planner targets always respect robots.txt.

## CLI

```
vigileye collect TARGET… --case NAME [--limit N] [--opt k=v]
vigileye analyze --case NAME [--lens L] [--query Q] [--ach] [--vision] [--platform P] [--since ISO]
                 [--verify | --no-verify] [--critic] [--ground-cves | --no-ground-cves]
                 [--formats md,json,html,stix,navigator] [--tlp clear|green|amber|red] [--out DIR]
vigileye run TARGET… [same flags]        # collect + analyze
vigileye cases | reports --case NAME | show ID [--format md|json|html|stix|navigator] [--tlp T] | export --case NAME --out f.jsonl
vigileye signals --case NAME [--json]    # computed evidence signals, no LLM
vigileye ask --case NAME "question" [--k N] [--platform P] [--no-verify] [--json]
vigileye investigate "question" --seed SPEC… --case NAME [--rounds N] [--max-new N] [--limit N] [--max-docs N]
                     [--yes] [--opt k=v] [same analysis flags as analyze]
vigileye evidence verify|manifest|log --case NAME [--out case.json] [--anchor HEAD]
vigileye refdata update | refdata cve CVE-ID   # CISA KEV / FIRST EPSS cache
vigileye lenses | collectors | providers [--test]
vigileye serve [--host 127.0.0.1] [--port 8765]
```

| Flag | Effect | Default |
|---|---|---|
| `--verify` / `--no-verify` | claim-level citation verification | on (`analysis.verify`) |
| `--critic` | add the LLM red-team pass to the rule-based critic | off (`analysis.critic_llm`); rule-based critic always on (`analysis.critique`) |
| `--ground-cves` / `--no-ground-cves` | CISA KEV + FIRST EPSS grounding | off (`analysis.ground_cves`) |
| `--formats …,stix,navigator` | also write a STIX 2.1 bundle / ATT&CK Navigator layer | `output.formats` |
| `--tlp` | TLP marking on the STIX bundle | `output.tlp` (`clear`) |

Global flags: `--config`, `--store`, `-v`, `-q`. The v1 form `--reddit r/x --max_tokens N` still works and maps to `run`.

## HTTP API

Start it with `vigileye serve`. It binds to localhost by default. Set `VIGILEYE_API_TOKEN` to require `Authorization: Bearer …`.

| Method | Path | Body |
|---|---|---|
| GET | `/health`, `/v1/lenses`, `/v1/collectors`, `/v1/providers`, `/v1/cases`, `/v1/cases/{case}/reports`, `/v1/reports/{id}` | |
| GET | `/v1/reports/{id}/stix[?tlp=amber]`, `/v1/reports/{id}/navigator` | |
| GET | `/v1/cases/{case}/evidence` (manifest), `/v1/cases/{case}/evidence/verify[?anchor=HEAD]` | |
| POST | `/v1/collect` | `{targets, case, limit, opts}` |
| POST | `/v1/analyze` | `{case, lens, query, ach, vision, verify, critic, ground_cves, formats, tlp}` |
| POST | `/v1/run` | `{targets, case, limit, lens, query, …same options as /v1/analyze}` |
| POST | `/v1/ask` | `{case, question, k, platform, verify}` |
| POST | `/v1/investigate` | `{question, seeds, case, rounds (≤5), max_new (≤8), limit (≤200), max_docs (≤5000), lens, opts, …analysis options}`; `422` if it stops before any analysis |

Omitted options keep the configured defaults. An unknown format, TLP level or lens returns `400`, before anything is collected. A target or seed the policy refuses returns `400` with `{error, rejected: [{spec, reason}]}`, and nothing is collected. Planner targets inside `/v1/investigate` are collected without asking.

## Security model

- **Localhost bind.** `vigileye serve` binds to `127.0.0.1` by default. Binding anywhere else (the Docker image uses `0.0.0.0`) without a token prints a warning.
- **Bearer token.** Set `VIGILEYE_API_TOKEN` to require `Authorization: Bearer …` on everything except `/health`. The token is compared in constant time. Anyone with the token can collect, analyse and read every case in the store.
- **Browser requests.** POST bodies must be `Content-Type: application/json`. Without a token, a request whose `Host` is not `localhost`, an IP literal or a name in `VIGILEYE_API_ALLOWED_HOSTS`, or that carries a foreign `Origin`, is refused with 403. This blocks CSRF from web pages you open and DNS-rebinding reads of your cases.
- **The API cannot read local files.** `file:` targets and seeds are refused (target policy `mode="api"`), lenses are limited to the built-in names (a lens path would be a file read), and output paths are not accepted.
- **SSRF guard.** Every collect, run and investigate target passes `check_target`, which refuses URLs on non-public hosts. At fetch time, every request and redirect hop of every collector, and every image the vision role fetches, is checked again. Image references from collected content are never read from the local disk; only documents you load with `file:` can point at local images. Set `collectors.web.allow_private: true` (web pages and images) or `collectors.http.allow_private: true` (every collector) for lab use; the API's target policy stays strict either way, and targets submitted through the API or proposed by the investigation planner keep the fetch-time checks on every redirect hop. Response bodies are streamed and capped (`max_response_bytes`, 32 MiB; web pages `max_page_bytes`, 10 MiB; robots.txt 512 KiB) with an overall `fetch_deadline` (120 s).
- **Mastodon credentials.** A token in `collectors.mastodon.tokens.<instance>` is sent only to that instance. The global `access_token` / `MASTODON_TOKEN` is sent only to its home instance (`token_instance`, else the configured `instance`, else `MASTODON_INSTANCE`, else mastodon.social). Planner-proposed targets are fetched without credentials. The checks resolve names with the system resolver and fail closed. They do not stop DNS rebinding between the check and the connection.
- **Untrusted content.** Collected text reaches models only inside nonce-fenced blocks marked as data, for analysis, `ask`, citation checks and the planner.

There's also a Docker image: `docker build -t vigileye . && docker run -p 8765:8765 -e OPENAI_API_KEY vigileye`.

## Responsible use

VigilEye is built for analyzing **public** information about topics, communities, events and threats.

- Respect each platform's terms of service and rate limits. The web collector honors robots.txt.
- Do not use it to doxx, stalk, harass or de-anonymize private individuals. It deliberately does not harvest personal emails or phone numbers.
- Signals, coordination networks, STIX exports, `ask` answers, investigation reports and evidence manifests carry account counts or keyed pseudonyms, never author names. A coordination signal is a statistical anomaly, not proof of inauthenticity.
- `vigileye investigate` cannot be pointed at a person: account-level seeds, planner targets and profile pages are refused.
- Every judgment is machine-generated. A human analyst must review it before anyone acts on it.

## Development

```bash
pip install -e ".[dev]" && pytest -q
```

The test suite runs offline against the built-in `mock://` provider (chat and embeddings), a local fake OpenAI-compatible server and injected DNS resolvers.
