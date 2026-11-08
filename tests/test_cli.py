"""End-to-end CLI tests on the offline mock provider (no network)."""

import json
from pathlib import Path

from vigileye.cli import main

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = str(ROOT / "config" / "offline.yaml")


def _corpus(tmp_path: Path) -> Path:
    p = tmp_path / "posts.jsonl"
    rows = [
        {"id": "p1", "platform": "file", "source": "demo", "author": "alice", "created_at": "2026-11-01T10:00:00Z",
         "title": "Exploit chatter", "text": "PoC for CVE-2024-3400 posted, C2 at hxxp://bad-domain[.]com and 203.0.113.7",
         "metrics": {"score": 40}},
        {"id": "p2", "platform": "file", "source": "demo", "author": "bob", "created_at": "2026-11-02T10:00:00Z",
         "text": "Patch your Palo Alto firewalls now. T1190 initial access observed.", "metrics": {"score": 12}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_full_offline_flow(tmp_path, capsys):
    store = str(tmp_path / "cases.db")
    base = ["--config", OFFLINE, "--store", store, "-q"]
    corpus = _corpus(tmp_path)

    assert main(base + ["collect", f"file:{corpus}", "--case", "demo"]) == 0
    assert main(base + ["analyze", "--case", "demo", "--lens", "cyber", "--ach", "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "BLUF" in out
    files = sorted(p.suffix for p in (tmp_path / "out").iterdir())
    assert files == [".html", ".json", ".md"]

    md = next((tmp_path / "out").glob("*.md")).read_text()
    assert "CVE-2024-3400" in md

    assert main(base + ["cases"]) == 0
    assert "demo" in capsys.readouterr().out
    assert main(base + ["reports", "--case", "demo"]) == 0
    capsys.readouterr()
    assert main(base + ["show", "1", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["case"] == "demo"
    assert main(base + ["export", "--case", "demo", "--out", str(tmp_path / "e.jsonl")]) == 0
    assert len((tmp_path / "e.jsonl").read_text().strip().splitlines()) == 2


def test_run_and_info_commands(tmp_path, capsys):
    store = str(tmp_path / "cases.db")
    base = ["--config", OFFLINE, "--store", store, "-q"]
    assert main(base + ["run", f"file:{_corpus(tmp_path)}", "--case", "r", "--out", str(tmp_path / "o"), "--formats", "md"]) == 0
    assert main(base + ["lenses"]) == 0
    assert "influence" in capsys.readouterr().out
    assert main(base + ["collectors"]) == 0
    assert "reddit" in capsys.readouterr().out
    assert main(base + ["providers", "--test"]) == 0
    assert "OK" in capsys.readouterr().out


def test_errors_are_clean(tmp_path):
    base = ["--config", OFFLINE, "--store", str(tmp_path / "c.db"), "-q"]
    assert main(base + ["analyze", "--case", "nope"]) == 1
    assert main([]) == 2
