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
