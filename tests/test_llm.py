"""LLM router tests: mock provider, real OpenAI SDK path against a local fake endpoint, fallback, JSON parsing."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vigileye.config import Config
from vigileye.llm import LLMError, LLMRouter, extract_json

SEEN = []


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if self.headers.get("X-Stealth-Token") != "s3cret":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "unauthorized"}}')
            return
        if self.path.endswith("/embeddings"):  # returned out of order: the router must sort by index
            inputs = body["input"]
            out = {"object": "list", "model": body["model"],
                   "data": [{"object": "embedding", "index": i, "embedding": [float(len(t)), float(i), 1.0]}
                            for i, t in reversed(list(enumerate(inputs)))],
                   "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        content = '{"ok": true}' if body.get("response_format") else "pong"
        out = {
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! here: {"a": [1, 2,], "b": "x}"} trailing') == {"a": [1, 2], "b": "x}"}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_mock_router_roles_and_usage():
    r = LLMRouter.from_config(Config({"providers": {"mock": {"base_url": "mock://"}},
                                      "roles": {"analyst": ["mock/m1"]}}))
    assert [e.label for e in r.chain("synthesizer")] == ["mock/m1"]  # falls back to analyst
    assert r.chat_json([{"role": "user", "content": "Return JSON"}])["bluf"]
    assert r.ping("analyst")["ok"]
    assert r.usage_summary()["totals"]["calls"] >= 1


def test_openai_compatible_endpoint_with_headers_and_fallback(fake_server):
    cfg = Config({
        "providers": {
            "broken": {"base_url": "http://127.0.0.1:9/v1", "api_key": "x", "max_retries": 0, "timeout": 2},
            "stealth": {"base_url": fake_server, "api_key": "k", "headers": {"X-Stealth-Token": "s3cret"},
                        "pricing": {"*": {"input": 1.0, "output": 2.0}}},
        },
        "roles": {"analyst": [{"provider": "broken", "model": "a"}, {"provider": "stealth", "model": "ghost-70b"}]},
        "llm": {"cache": True},
    })
    r = LLMRouter.from_config(cfg)
    res = r.chat([{"role": "user", "content": "ping"}])
    assert res.text == "pong" and res.provider == "stealth" and res.model == "ghost-70b"
    assert r.chat_json([{"role": "user", "content": "json please"}], role="analyst") == {"ok": True}
    assert SEEN[-1]["body"]["response_format"] == {"type": "json_object"}
    # cached second call does not hit the server
    n = len(SEEN)
    assert r.chat([{"role": "user", "content": "ping"}]).cached
    assert len(SEEN) == n
    u = r.usage_summary()
    assert u["models"]["stealth/ghost-70b"]["prompt_tokens"] == 14
    assert u["totals"]["cost_usd"] > 0


def test_all_fail_raises(fake_server):
    cfg = Config({"providers": {"p": {"base_url": fake_server, "api_key": "k", "max_retries": 0}},
                  "roles": {"analyst": ["p/m"]}})
    with pytest.raises(LLMError):
        LLMRouter.from_config(cfg).chat([{"role": "user", "content": "x"}])  # 401: missing stealth header


def test_embed_mock_has_role_and_no_chat_fallback():
    chat_only = LLMRouter.from_config(Config({"providers": {"mock": {"base_url": "mock://"}},
                                              "roles": {"analyst": ["mock/m1"]}}))
    assert chat_only.has_role("analyst") and chat_only.has_role("verifier")  # verifier falls back to analyst
    assert not chat_only.has_role("embedder")
    with pytest.raises(LLMError):
        chat_only.embed(["no chat model may stand in for an embedder"])

    r = LLMRouter.from_config(Config({"providers": {"mock": {"base_url": "mock://"}},
                                      "roles": {"analyst": ["mock/m1"], "embedder": ["mock/mock-embed"]}}))
    assert r.has_role("embedder")
    texts = ["Ransomware attacks on hospitals", "hospital ransomware attack", "a recipe for bread"]
    v = r.embed(texts)
    assert v == r.embed(texts)  # deterministic
    assert [len(x) for x in v] == [256, 256, 256]
    assert all(abs(sum(a * a for a in x) - 1.0) < 1e-9 for x in v)  # unit vectors
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(v[0], v[1]) > 0.9 > dot(v[0], v[2])
    assert r.embed([]) == []
    with pytest.raises(TypeError):
        r.embed("a bare string")
    res = r.embeddings(["x"] * 130)  # 3 batches of <= 64
    assert res.label == "mock/mock-embed" and len(res.vectors) == 130
    assert r.usage_summary()["models"]["mock/mock-embed"]["calls"] == 2 + 3


def test_embed_openai_compatible_endpoint_batches_and_fallback(fake_server):
    cfg = Config({
        "providers": {
            "broken": {"base_url": "http://127.0.0.1:9/v1", "api_key": "x", "max_retries": 0, "timeout": 2},
            "stealth": {"base_url": fake_server, "api_key": "k", "headers": {"X-Stealth-Token": "s3cret"},
                        "pricing": {"*": {"input": 0.02, "output": 0.0}}},
        },
        "roles": {"analyst": ["stealth/ghost-70b"],
                  "embedder": [{"provider": "broken", "model": "e"},
                               {"provider": "stealth", "model": "text-embedding-3-small", "dimensions": 3}]},
    })
    r = LLMRouter.from_config(cfg)
    texts = ["x" * (i + 1) for i in range(130)]
    n = len(SEEN)
    res = r.embeddings(texts)
    calls = [c for c in SEEN[n:] if c["path"].endswith("/embeddings")]
    assert [len(c["body"]["input"]) for c in calls] == [64, 64, 2]
    assert all(c["body"]["encoding_format"] == "float" and c["body"]["dimensions"] == 3 for c in calls)
    assert all(c["body"]["model"] == "text-embedding-3-small" for c in calls)
    assert res.provider == "stealth" and res.label == "stealth/text-embedding-3-small"
    assert [v[0] for v in res.vectors] == [float(len(t)) for t in texts]  # input order restored
    u = r.usage_summary()["models"]["stealth/text-embedding-3-small"]
    assert u["calls"] == 3 and u["prompt_tokens"] == 130 and u["completion_tokens"] == 0 and u["cost_usd"] > 0
    assert r.embed(["abc"]) == [[3.0, 0.0, 1.0]]
