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
 targets ──► collectors ──► case store (SQLite) ──► enrichment ──► map (per chunk) ──► reduce ──► ACH ──► report
 reddit:r/x   reddit hn rss        dedupe, history      IOCs (regex)     analyst role       synthesizer  md / json / html
 hn:search:x  web mastodon         llm cache            entities         parallel, cited    role         + stored
 rss:https:…  bluesky files                             vision (images)
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

Reports land in `export/` as Markdown, JSON and a self-contained HTML page. They're also saved in the case store (`vigileye reports --case panos`, `vigileye show 1`).

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

- **Roles.** `analyst` reads chunks in parallel, `extractor` does cheap structured extraction, `vision` describes images and `synthesizer` writes the final product. A role you leave out falls back to `analyst`.
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

## CLI

```
vigileye collect TARGET… --case NAME [--limit N] [--opt k=v]
vigileye analyze --case NAME [--lens L] [--query Q] [--ach] [--vision] [--platform P] [--since ISO] [--formats md,json,html] [--out DIR]
vigileye run TARGET… [same flags]        # collect + analyze
vigileye cases | reports --case NAME | show ID [--format md|json] | export --case NAME --out f.jsonl
vigileye lenses | collectors | providers [--test]
vigileye serve [--host 127.0.0.1] [--port 8765]
```

Global flags: `--config`, `--store`, `-v`, `-q`. The v1 form `--reddit r/x --max_tokens N` still works and maps to `run`.

## HTTP API

Start it with `vigileye serve`. It binds to localhost by default. Set `VIGILEYE_API_TOKEN` to require `Authorization: Bearer …`.

| Method | Path | Body |
|---|---|---|
| GET | `/health`, `/v1/lenses`, `/v1/collectors`, `/v1/providers`, `/v1/cases`, `/v1/cases/{case}/reports`, `/v1/reports/{id}` | |
| POST | `/v1/collect` | `{targets, case, limit, opts}` |
| POST | `/v1/analyze` | `{case, lens, query, ach, vision}` |
| POST | `/v1/run` | `{targets, case, limit, lens, query}` |

There's also a Docker image: `docker build -t vigileye . && docker run -p 8765:8765 -e OPENAI_API_KEY vigileye`.

## Responsible use

VigilEye is built for analyzing **public** information about topics, communities, events and threats.

- Respect each platform's terms of service and rate limits. The web collector honors robots.txt.
- Do not use it to doxx, stalk, harass or de-anonymize private individuals. It deliberately does not harvest personal emails or phone numbers.
- Every judgment is machine-generated. A human analyst must review it before anyone acts on it.

## Development

```bash
pip install -e ".[dev]" && pytest -q
```

The test suite runs offline against the built-in `mock://` provider and a local fake OpenAI-compatible server.
