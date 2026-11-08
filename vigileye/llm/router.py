"""Role-based routing over OpenAI-compatible endpoints.

Every model VigilEye talks to — OpenAI, a vLLM / llama.cpp / SGLang / LM Studio
box under your desk, Ollama's /v1 endpoint, OpenRouter, Groq, a private
"stealth" deployment behind custom auth headers — is reached through the same
Chat Completions protocol. Roles (analyst, extractor, vision, synthesizer, and
the optional verifier and critic) map to ordered fallback chains of ``provider/model`` entries.
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

from .mock import mock_completion

log = logging.getLogger("vigileye.llm")

ROLE_FALLBACKS = {
    "synthesizer": ["analyst"],
    "extractor": ["analyst"],
    "verifier": ["extractor", "analyst"],  # citation grounding checks (analysis/verify.py)
    "critic": ["analyst"],                 # red-team review (analysis/critic.py)
    "vision": [],
    "analyst": [],
}

IMAGE_PROMPT = (
    "You are an imagery analyst. Describe this image factually and concisely for an intelligence "
    "analyst: visible text (transcribe it), people/objects/symbols/logos/flags, setting, apparent "
    "purpose (meme, screenshot, photo, chart, propaganda), and anything indicating time or place. "
    "Do not identify private individuals by face. Do not speculate beyond what is visible."
)


class LLMError(RuntimeError):
    pass


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
    """Pull the first JSON object/array out of model output (fences, prose, trailing commas)."""
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
        start = min([i for i in (cand.find("{"), cand.find("[")) if i >= 0], default=-1)
        if start < 0:
            continue
        opener = cand[start]
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
                    chunk = cand[start : i + 1]
                    for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                        try:
                            return json.loads(attempt)
                        except Exception:
                            continue
                    break
    raise LLMError(f"could not parse JSON from model output: {s[:200]!r}")


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
                verify=p.get("verify_tls", p.get("verify", True)),
                proxy=p.get("proxy"),
                organization=p.get("organization"),
                query=p.get("query") or {},
                supports_json_mode=bool(p.get("supports_json_mode", True)),
                inline_images=p.get("inline_images"),
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
                    json_mode=e.get("json_mode"),
                    extra_body=e.get("extra_body") or {},
                    params={k: v for k, v in e.items() if k not in known},
                ))
            roles[role] = entries
        if not roles and "mock" in providers:
            roles["analyst"] = [RouteEntry(provider="mock", model="mock-analyst")]
        return cls(providers, roles, temperature=float(llm_cfg.get("temperature", 0.2)),
                   cache=bool(llm_cfg.get("cache", True)), cache_backend=cache_backend)

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

    def describe_image(self, url_or_path: str, prompt: str | None = None, role: str = "vision") -> str:
        chain = self.chain(role)
        if not chain:
            raise LLMError("no 'vision' role configured")
        entry = chain[0]
        provider = self.providers[entry.provider]
        inline = provider.inline_images
        if inline is None:
            inline = "api.openai.com" not in provider.base_url
        image_url = self._image_ref(url_or_path, inline=inline and not provider.is_mock)
        messages = [
            {"role": "system", "content": prompt or IMAGE_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        return self.chat(messages, role=role, max_tokens=400).text.strip()

    @staticmethod
    def _image_ref(url_or_path: str, inline: bool) -> str:
        p = Path(url_or_path)
        if url_or_path.startswith("data:"):
            return url_or_path
        if p.is_file():
            data, mime = p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/png"
        elif inline and url_or_path.startswith(("http://", "https://")):
            import httpx

            r = httpx.get(url_or_path, timeout=30, follow_redirects=True,
                          headers={"User-Agent": "VigilEye/2.0"})
            r.raise_for_status()
            data = r.content
            mime = r.headers.get("content-type", "").split(";")[0] or mimetypes.guess_type(url_or_path)[0] or "image/jpeg"
        else:
            return url_or_path
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
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                # Many self-hosted servers reject response_format / temperature for some models; retry plain.
                msg = str(e).lower()
                if use_json and ("response_format" in msg or "json" in msg or "400" in msg):
                    kwargs.pop("response_format", None)
                    resp = client.chat.completions.create(**kwargs)
                elif "temperature" in msg:
                    kwargs.pop("temperature", None)
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
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
        if key:
            payload = {"text": text, "model": model_name, "provider": entry.provider, "usage": usage, "latency_s": result.latency_s}
            self._mem_cache[key] = payload
            if self._cache_backend:
                try:
                    self._cache_backend[1](key, payload)
                except Exception as e:  # cache must never break a run
                    log.debug("cache put failed: %s", e)
        return result

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
