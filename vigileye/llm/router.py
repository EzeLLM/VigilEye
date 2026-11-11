"""Role-based routing over OpenAI-compatible endpoints.

Every model VigilEye talks to — OpenAI, a vLLM / llama.cpp / SGLang / LM Studio
box under your desk, Ollama's /v1 endpoint, OpenRouter, Groq, a private
"stealth" deployment behind custom auth headers — is reached through the same
Chat Completions protocol. Roles (analyst, extractor, vision, synthesizer, and
the optional verifier and critic) map to ordered fallback chains of ``provider/model`` entries.
The optional ``embedder`` role is served through ``/v1/embeddings`` instead and never falls back
to a chat role (see :meth:`LLMRouter.embed`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .mock import mock_completion, mock_embedding

log = logging.getLogger("vigileye.llm")

ROLE_FALLBACKS = {
    "synthesizer": ["analyst"],
    "extractor": ["analyst"],
    "verifier": ["extractor", "analyst"],  # citation grounding checks (analysis/verify.py)
    "critic": ["analyst"],                 # red-team review (analysis/critic.py)
    "planner": ["analyst"],                # collection planner (investigate.py)
    "vision": [],
    "analyst": [],
    "embedder": [],                        # /v1/embeddings only: a chat model can never stand in
}
EMBED_BATCH = 64  # inputs per /v1/embeddings request
MAX_IMAGE_BYTES = 20 << 20  # largest image fetched or read for the vision role

IMAGE_PROMPT = (
    "You are an imagery analyst. Describe this image factually and concisely for an intelligence "
    "analyst: visible text (transcribe it), people/objects/symbols/logos/flags, setting, apparent "
    "purpose (meme, screenshot, photo, chart, propaganda), and anything indicating time or place. "
    "Do not identify private individuals by face. Do not speculate beyond what is visible."
)


class LLMError(RuntimeError):
    pass


def _parses(text: str) -> bool:
    try:
        extract_json(text)
        return True
    except Exception:
        return False


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False

    def json(self) -> Any:
        return extract_json(self.text)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of model output (fences, prose, trailing commas).

    Every ``{`` / ``[`` is tried in turn, so prose that cites ``[reddit:t3_x]`` or mentions
    ``{draft}`` before the real JSON value does not stop the scan at the first bracket.
    """
    if text is None:
        raise LLMError("empty model output")
    s = text.strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", s, re.S)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(s)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
        for start, opener in ((i, ch) for i, ch in enumerate(cand) if ch in "{["):
            chunk = _balanced(cand, start, opener)
            if chunk is None:
                continue
            for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                try:
                    return json.loads(attempt)
                except Exception:
                    continue
    raise LLMError(f"could not parse JSON from model output: {s[:200]!r}")


def _balanced(cand: str, start: int, opener: str) -> str | None:
    """The balanced ``{...}`` / ``[...]`` chunk that starts at ``start``, or None when unclosed."""
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(cand)):
        ch = cand[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return cand[start : i + 1]
    return None


_TRUE_STRINGS = {"true", "yes", "on", "1"}
_FALSE_STRINGS = {"false", "no", "off", "0", ""}


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    """Coerce a config value (often an interpolated ``${ENV:-true}`` string) to a bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    return default


def _verify_value(value: Any) -> bool | str:
    """``verify_tls``: a bool (also from ``"true"``/``"false"`` strings) or a CA-bundle path."""
    b = _as_bool(value)
    if b is not None:
        return b
    s = str(value).strip()
    if Path(s).expanduser().exists():
        return str(Path(s).expanduser())
    raise LLMError(f"verify_tls must be true/false or the path of an existing CA bundle, got {s!r}")


@dataclass
class EmbeddingResult:
    """Vectors for every input, in input order, all from one route entry (one vector space)."""

    vectors: list[list[float]]
    model: str      # model name the endpoint reported
    provider: str
    label: str      # configured ``provider/model``: the stable key for caching these vectors
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str = ""
    headers: dict = field(default_factory=dict)
    timeout: float = 120.0
    max_retries: int = 3
    verify: bool | str = True
    proxy: str | None = None
    organization: str | None = None
    query: dict = field(default_factory=dict)
    supports_json_mode: bool = True
    inline_images: bool | None = None
    pricing: dict = field(default_factory=dict)  # model -> {input, output} USD per 1M tokens
    _client: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_mock(self) -> bool:
        return self.base_url.startswith("mock:")

    def client(self):
        if self.is_mock:
            return None
        with self._lock:
            if self._client is None:
                try:
                    import httpx
                    from openai import OpenAI
                except ImportError as e:  # pragma: no cover
                    raise LLMError("the 'openai' package is required for non-mock providers: pip install openai") from e
                http_kwargs: dict[str, Any] = {"verify": self.verify, "timeout": self.timeout}
                if self.proxy:
                    http_kwargs["proxy"] = self.proxy
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "sk-no-key",
                    organization=self.organization,
                    default_headers=self.headers or None,
                    default_query=self.query or None,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    http_client=httpx.Client(**http_kwargs),
                )
            return self._client


@dataclass
class RouteEntry:
    provider: str
    model: str
    context_window: int = 32000
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool | None = None
    extra_body: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)  # top_p, seed, reasoning_effort, ...

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


class LLMRouter:
    def __init__(self, providers: dict[str, Provider], roles: dict[str, list[RouteEntry]], *,
                 temperature: float = 0.2, cache: bool = True,
                 cache_backend: tuple[Callable[[str], dict | None], Callable[[str, dict], None]] | None = None):
        self.providers = providers
        self.roles = roles
        self.default_temperature = temperature
        self.cache_enabled = cache
        self._mem_cache: dict[str, dict] = {}
        self._cache_backend = cache_backend
        self._usage_lock = threading.Lock()
        self._usage: dict[str, dict] = {}
        self.allow_private_urls = False  # lab opt-out of the image SSRF guard (collectors.web.allow_private)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_config(cls, config, cache_backend=None) -> "LLMRouter":
        raw = config.raw if hasattr(config, "raw") else dict(config)
        llm_cfg = raw.get("llm", {}) or {}
        providers: dict[str, Provider] = {}
        for name, p in (raw.get("providers") or {}).items():
            p = p or {}
            base = str(p.get("base_url") or "").rstrip("/")
            if not base:
                log.debug("provider %s has no base_url; skipping", name)
                continue
            providers[name] = Provider(
                name=name,
                base_url=base,
                api_key=str(p.get("api_key") or ""),
                headers={k: str(v) for k, v in (p.get("headers") or {}).items()},
                timeout=float(p.get("timeout", llm_cfg.get("timeout", 120))),
                max_retries=int(p.get("max_retries", llm_cfg.get("max_retries", 3))),
                verify=_verify_value(p.get("verify_tls", p.get("verify", True))),
                proxy=p.get("proxy"),
                organization=p.get("organization"),
                query=p.get("query") or {},
                supports_json_mode=bool(_as_bool(p.get("supports_json_mode"), True)),
                inline_images=_as_bool(p.get("inline_images")),
                pricing=p.get("pricing") or {},
            )
        roles: dict[str, list[RouteEntry]] = {}
        for role, chain in (raw.get("roles") or {}).items():
            if isinstance(chain, (dict, str)):
                chain = [chain]
            entries = []
            for e in chain or []:
                if isinstance(e, str):  # "provider/model"
                    prov, _, model = e.partition("/")
                    e = {"provider": prov, "model": model}
                known = {"provider", "model", "context_window", "temperature", "max_tokens", "json_mode", "extra_body"}
                entries.append(RouteEntry(
                    provider=e["provider"],
                    model=str(e.get("model") or "default"),
                    context_window=int(e.get("context_window", 32000)),
                    temperature=e.get("temperature"),
                    max_tokens=e.get("max_tokens"),
                    json_mode=_as_bool(e.get("json_mode")),
                    extra_body=e.get("extra_body") or {},
                    params={k: v for k, v in e.items() if k not in known},
                ))
            roles[role] = entries
        if not roles and "mock" in providers:
            roles["analyst"] = [RouteEntry(provider="mock", model="mock-analyst")]
        router = cls(providers, roles, temperature=float(llm_cfg.get("temperature", 0.2)),
                     cache=bool(_as_bool(llm_cfg.get("cache"), True)), cache_backend=cache_backend)
        col = raw.get("collectors") or {}
        router.allow_private_urls = any(bool(_as_bool((col.get(k) or {}).get("allow_private"), False))
                                        for k in ("web", "http"))
        return router

    def set_cache_backend(self, get: Callable[[str], dict | None], put: Callable[[str, dict], None]) -> None:
        self._cache_backend = (get, put)

    def chain(self, role: str) -> list[RouteEntry]:
        seen, out = set(), []
        for r in [role, *ROLE_FALLBACKS.get(role, ["analyst"])]:
            for e in self.roles.get(r, []):
                if e.label not in seen and e.provider in self.providers:
                    seen.add(e.label)
                    out.append(e)
        return out

    def has_role(self, role: str) -> bool:
        """True when at least one configured provider can serve ``role`` (its own entries or its
        fallbacks; ``embedder`` has none, so this is False until ``roles.embedder`` is set)."""
        return bool(self.chain(role))

    def context_window(self, role: str = "analyst") -> int:
        chain = self.chain(role)
        return min((e.context_window for e in chain), default=32000)

    # ------------------------------------------------------------------ calls
    def chat(self, messages: list[dict], role: str = "analyst", *, temperature: float | None = None,
             max_tokens: int | None = None, json_mode: bool = False) -> ChatResult:
        chain = self.chain(role)
        if not chain:
            raise LLMError(f"no providers configured for role '{role}' (check 'roles' in your config)")
        errors = []
        for entry in chain:
            try:
                return self._call(entry, messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
            except Exception as e:  # try next in the chain
                log.warning("role=%s %s failed: %s", role, entry.label, e)
                errors.append(f"{entry.label}: {e}")
        raise LLMError(f"all providers failed for role '{role}': " + " | ".join(errors))

    def chat_json(self, messages: list[dict], role: str = "extractor") -> Any:
        res = self.chat(messages, role=role, json_mode=True)
        try:
            return res.json()
        except LLMError:
            repair = list(messages) + [
                {"role": "assistant", "content": res.text},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the valid JSON value, no prose, no code fences."},
            ]
            return self.chat(repair, role=role, json_mode=True, temperature=0.0).json()

    def embed(self, texts: list[str], role: str = "embedder") -> list[list[float]]:
        """Embed ``texts`` via ``/v1/embeddings`` (batches of :data:`EMBED_BATCH`), one vector per text.

        Raises :class:`LLMError` when ``role`` has no configured entry (``embedder`` never falls
        back to a chat role) or every entry fails, so callers can fall back to lexical search.
        """
        return self.embeddings(texts, role=role).vectors

    def embeddings(self, texts: list[str], role: str = "embedder") -> EmbeddingResult:
        """Like :meth:`embed`, plus the route entry that served the call. All vectors of one call
        come from a single entry: if any batch fails, the whole call moves to the next entry."""
        if isinstance(texts, str):
            raise TypeError("embed() takes a list of strings, not a string")
        texts = [str(t) for t in texts]
        chain = self.chain(role)
        if not chain:
            raise LLMError(f"no providers configured for role '{role}' (add 'roles.{role}' to your config)")
        if not texts:
            e = chain[0]
            return EmbeddingResult([], e.model, e.provider, e.label, {"prompt_tokens": 0, "completion_tokens": 0})
        errors = []
        for entry in chain:
            try:
                return self._embed_call(entry, texts)
            except Exception as e:  # try next in the chain
                log.warning("role=%s %s embeddings failed: %s", role, entry.label, e)
                errors.append(f"{entry.label}: {e}")
        raise LLMError(f"all providers failed for role '{role}': " + " | ".join(errors))

    def _embed_call(self, entry: RouteEntry, texts: list[str]) -> EmbeddingResult:
        provider = self.providers[entry.provider]
        vectors: list[list[float]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        model_name, total = entry.model, 0.0
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            t0 = time.monotonic()
            if provider.is_mock:
                vecs, u = mock_embedding(batch, model=entry.model)
            else:
                kwargs: dict[str, Any] = {"model": entry.model, "input": batch, "encoding_format": "float"}
                if entry.extra_body:
                    kwargs["extra_body"] = entry.extra_body
                kwargs.update(entry.params)  # dimensions, user, ...
                resp = provider.client().embeddings.create(**kwargs)
                data = sorted(getattr(resp, "data", None) or [], key=lambda d: getattr(d, "index", 0) or 0)
                vecs = [[float(x) for x in (getattr(d, "embedding", None) or [])] for d in data]
                u = {"prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0,
                     "completion_tokens": 0}
                model_name = getattr(resp, "model", None) or entry.model
            if len(vecs) != len(batch) or not all(vecs):
                raise LLMError(f"endpoint returned {len(vecs)} embeddings for {len(batch)} inputs")
            if len({len(v) for v in vecs + vectors[:1]}) != 1:
                raise LLMError("endpoint returned embeddings of differing dimensions")
            latency = time.monotonic() - t0
            total += latency
            self._record_usage(entry, u, latency)
            vectors.extend(vecs)
            usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        return EmbeddingResult(vectors, model_name, entry.provider, entry.label, usage, round(total, 3))

    def describe_image(self, url_or_path: str, prompt: str | None = None, role: str = "vision", *,
                       allow_local: bool = False) -> str:
        """Describe one image with the ``vision`` role.

        ``url_or_path`` is an http(s) URL, a ``data:`` URL or, only with ``allow_local`` (documents
        the analyst loaded with ``file:``), a local image file. Non-public URLs are refused unless
        ``collectors.web.allow_private`` is set; the mock provider never fetches anything.
        """
        chain = self.chain(role)
        if not chain:
            raise LLMError("no 'vision' role configured")
        entry = chain[0]
        provider = self.providers[entry.provider]
        inline = provider.inline_images
        if inline is None:
            inline = "api.openai.com" not in provider.base_url
        image_url = self._image_ref(url_or_path, inline=inline and not provider.is_mock, allow_local=allow_local,
                                    allow_private=self.allow_private_urls or provider.is_mock)
        messages = [
            {"role": "system", "content": prompt or IMAGE_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        return self.chat(messages, role=role, max_tokens=400).text.strip()

    @staticmethod
    def _image_ref(url_or_path: str, inline: bool, *, allow_local: bool = False, allow_private: bool = False,
                   resolver: Callable | None = None) -> str:
        """The ``image_url`` to send: a ``data:`` URL (inlined bytes) or the http(s) URL itself.

        * http(s): refused unless the host is public (:func:`vigileye.policy.is_public_url`) or
          ``allow_private``; when inlined, every redirect hop is checked again and the body is capped
          at :data:`MAX_IMAGE_BYTES`.
        * local paths: read only with ``allow_local`` and only for image files. A path taken from a
          collected media URL is never read otherwise, so collected content cannot make VigilEye send a
          local file to the vision endpoint.
        * anything else (``file://``, other schemes, a missing path) raises :class:`LLMError`.
        """
        ref = str(url_or_path or "").strip()
        if ref.startswith("data:"):
            return ref
        if ref.lower().startswith(("http://", "https://")):
            from ..policy import check_url, guard_request_hook

            if not allow_private:
                reason = check_url(ref, resolver=resolver)
                if reason:
                    raise LLMError(f"refusing to fetch a non-public image URL ({reason}): {ref}")
            if not inline:
                return ref
            import httpx

            hooks = {} if allow_private else {"event_hooks": {"request": [guard_request_hook(resolver)]}}
            with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "VigilEye/2.0"},
                              **hooks) as client, client.stream("GET", ref) as r:
                r.raise_for_status()
                mime = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if mime and not mime.startswith("image/") and mime != "application/octet-stream":
                    raise LLMError(f"not an image ({mime}): {ref}")
                buf, size = [], 0
                for chunk in r.iter_bytes():
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise LLMError(f"image larger than {MAX_IMAGE_BYTES // (1 << 20)} MiB: {ref}")
                    buf.append(chunk)
            data = b"".join(buf)
            if not mime.startswith("image/"):
                mime = mimetypes.guess_type(ref.split("?", 1)[0])[0] or "image/jpeg"
        elif allow_local and ref and Path(ref).is_file():
            p = Path(ref)
            mime = mimetypes.guess_type(p.name)[0] or ""
            if not mime.startswith("image/"):
                raise LLMError(f"refusing to send a local file that is not an image: {p.name}")
            if p.stat().st_size > MAX_IMAGE_BYTES:
                raise LLMError(f"image larger than {MAX_IMAGE_BYTES // (1 << 20)} MiB: {p.name}")
            data = p.read_bytes()
        else:
            raise LLMError("unsupported image reference (expected an http(s) or data: URL"
                           + (", or a local image file" if allow_local else "") + f"): {ref[:200]}")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _cache_key(self, entry: RouteEntry, messages, temperature, max_tokens, json_mode) -> str:
        blob = json.dumps([entry.label, messages, temperature, max_tokens, json_mode, entry.extra_body, entry.params],
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _call(self, entry: RouteEntry, messages, *, temperature, max_tokens, json_mode) -> ChatResult:
        provider = self.providers[entry.provider]
        temp = temperature if temperature is not None else (
            entry.temperature if entry.temperature is not None else self.default_temperature)
        mtok = max_tokens or entry.max_tokens
        use_json = json_mode and (entry.json_mode if entry.json_mode is not None else provider.supports_json_mode)

        key = self._cache_key(entry, messages, temp, mtok, use_json) if self.cache_enabled else None
        if key:
            hit = self._mem_cache.get(key)
            if hit is None and self._cache_backend:
                try:
                    hit = self._cache_backend[0](key)
                except Exception:
                    hit = None
            if hit is not None and json_mode and not _parses(str(hit.get("text") or "")):
                hit = None  # an unparseable entry from an older version: call the model again
            if hit is not None:
                return ChatResult(**{**hit, "cached": True})

        t0 = time.monotonic()
        if provider.is_mock:
            text, usage = mock_completion(messages, model=entry.model, json_mode=json_mode)
            model_name = entry.model
        else:
            kwargs: dict[str, Any] = {"model": entry.model, "messages": messages, "temperature": temp}
            if mtok:
                kwargs["max_tokens"] = mtok
            if entry.extra_body:
                kwargs["extra_body"] = entry.extra_body
            kwargs.update(entry.params)
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            client = provider.client()
            resp = self._create_with_retries(client, kwargs)
            if not getattr(resp, "choices", None):
                raise LLMError("endpoint returned no choices")
            text = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            }
            model_name = getattr(resp, "model", None) or entry.model
        latency = time.monotonic() - t0
        self._record_usage(entry, usage, latency)
        result = ChatResult(text=text, model=model_name, provider=entry.provider, usage=usage, latency_s=round(latency, 3))
        if key and text.strip() and json_mode and not _parses(text):
            key = None  # never cache unparseable JSON output: a rerun must be able to recover
        if key and text.strip():
            payload = {"text": text, "model": model_name, "provider": entry.provider, "usage": usage, "latency_s": result.latency_s}
            self._mem_cache[key] = payload
            if self._cache_backend:
                try:
                    self._cache_backend[1](key, payload)
                except Exception as e:  # cache must never break a run
                    log.debug("cache put failed: %s", e)
        return result

    @staticmethod
    def _create_with_retries(client, kwargs: dict):
        """``chat.completions.create`` that drops a parameter the server rejects and retries.

        Many self-hosted servers (and o-series / GPT-5 models) reject ``temperature`` or
        ``response_format``. Each rejected parameter is handled on its own, at most once, so a
        JSON-mode call on a model that rejects ``temperature`` still succeeds. A bare HTTP 400 that
        names neither parameter drops JSON mode only as a last resort, once.
        """
        dropped: set[str] = set()
        while True:
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg and "temperature" in kwargs and "temperature" not in dropped:
                    drop = "temperature"
                elif "response_format" in kwargs and "response_format" not in dropped and (
                        "response_format" in msg or "json" in msg):
                    drop = "response_format"
                elif "response_format" in kwargs and "response_format" not in dropped and "400" in msg:
                    drop = "response_format"
                else:
                    raise
                log.debug("endpoint rejected %s; retrying without it", drop)
                kwargs.pop(drop, None)
                dropped.add(drop)

    def _record_usage(self, entry: RouteEntry, usage: dict, latency: float) -> None:
        provider = self.providers[entry.provider]
        price = provider.pricing.get(entry.model) or provider.pricing.get("*") or {}
        cost = (usage.get("prompt_tokens", 0) * float(price.get("input", 0))
                + usage.get("completion_tokens", 0) * float(price.get("output", 0))) / 1_000_000
        with self._usage_lock:
            u = self._usage.setdefault(entry.label, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                     "latency_s": 0.0, "cost_usd": 0.0})
            u["calls"] += 1
            u["prompt_tokens"] += usage.get("prompt_tokens", 0)
            u["completion_tokens"] += usage.get("completion_tokens", 0)
            u["latency_s"] = round(u["latency_s"] + latency, 3)
            u["cost_usd"] = round(u["cost_usd"] + cost, 6)

    def usage_summary(self) -> dict:
        with self._usage_lock:
            per_model = {k: dict(v) for k, v in self._usage.items()}
        totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        for v in per_model.values():
            for k in totals:
                totals[k] += v[k]
        totals["cost_usd"] = round(totals["cost_usd"], 6)
        return {"models": per_model, "totals": totals}

    def describe(self) -> dict:
        return {
            "providers": {n: {"base_url": p.base_url, "auth": bool(p.api_key), "headers": sorted(p.headers)}
                          for n, p in self.providers.items()},
            "roles": {r: [e.label for e in self.chain(r)] for r in sorted(set(self.roles) | {"analyst", "extractor", "vision", "synthesizer", "verifier", "critic"})},
        }

    def ping(self, role: str) -> dict:
        try:
            t0 = time.monotonic()
            res = self.chat([{"role": "user", "content": "Reply with the single word: pong"}], role=role, max_tokens=8)
            return {"role": role, "ok": True, "model": res.model, "provider": res.provider,
                    "latency_s": round(time.monotonic() - t0, 3), "reply": res.text.strip()[:40]}
        except Exception as e:
            return {"role": role, "ok": False, "error": str(e)}
