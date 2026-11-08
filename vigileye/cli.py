"""`vigileye` command line: collect → store → analyze → report, on any OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .config import Config, load_config


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "case"


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


TLP_LEVELS = ("clear", "green", "amber", "red")


def _formats(value: str | None) -> list[str] | None:
    """Parse and validate a ``--formats`` comma list before any work is done."""
    if not value:
        return None
    from .report import FORMATS

    out = []
    for f in value.split(","):
        f = f.strip().lower().lstrip(".")
        f = "md" if f == "markdown" else f
        if not f:
            continue
        if f not in FORMATS:
            raise ValueError(f"unknown report format {f!r}; choose from {','.join(FORMATS)}")
        out.append(f)
    return out or None


class App:
    """Shared wiring used by the CLI and the HTTP server."""

    def __init__(self, config: Config, store_path: str | None = None, quiet: bool = False):
        from .llm import LLMRouter
        from .store import CaseStore

        self.config = config
        self.quiet = quiet
        self.store = CaseStore(store_path or config.get("store.path", ".vigileye/cases.db"))
        self.router = LLMRouter.from_config(config)
        if config.get("llm.cache", True):
            self.router.set_cache_backend(self.store.cache_get, self.store.cache_put)

    def progress(self, msg: str) -> None:
        if not self.quiet:
            _err(f"  · {msg}")

    def collect(self, targets: list[str], case: str, limit: int = 100, opts: dict | None = None) -> dict:
        from .collectors import collect

        self.store.ensure_case(case)
        counts = {}
        for spec in targets:
            docs, failed = [], None
            try:
                for d in collect(spec, limit=limit, config=self.config.section("collectors"), **(opts or {})):
                    docs.append(d)
            except Exception as e:  # keep partial results, report the failure
                failed = str(e)
            new = self.store.add_documents(case, docs) if docs else 0
            counts[spec] = {"collected": len(docs), "new": new, **({"error": failed} if failed else {})}
            self.progress(f"{spec}: {len(docs)} collected, {new} new" + (f" (error: {failed})" if failed else ""))
        return counts

    def analyze(self, case: str, *, lens: str | None = None, query: str = "", platform: str | None = None,
                since: datetime | None = None, limit: int | None = None, vision: bool | None = None,
                ach: bool | None = None, formats: list[str] | None = None, out: str | None = None,
                verify: bool | None = None, critic: bool | None = None, ground_cves: bool | None = None,
                tlp: str | None = None) -> tuple:
        from .analysis import AnalysisEngine
        from .report import write_report

        docs = self.store.get_documents(case, platform=platform, since=since, limit=limit)
        if not docs:
            raise RuntimeError(f"case '{case}' has no documents — run `vigileye collect` first")
        self.progress(f"analyzing {len(docs)} documents from case '{case}'")
        engine = AnalysisEngine(self.router, self.config)
        report = engine.run(docs, lens=lens or self.config.get("analysis.lens", "general"), query=query, case=case,
                            use_vision=vision, ach=ach, verify=verify, critic=critic, ground_cves=ground_cves,
                            progress=self.progress)
        rid = self.store.save_report(report)
        paths = write_report(report, out or self.config.get("output.dir", "export"),
                             formats=tuple(formats or self.config.get("output.formats", ["md", "json", "html"])),
                             tlp=tlp or self.config.get("output.tlp", "clear") or "clear")
        return rid, report, paths


def _kv(pairs: list[str] | None) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        low = v.lower()
        out[k] = True if low == "true" else False if low == "false" else int(v) if v.isdigit() else v
    return out


def _add_analysis_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lens", help="analytic lens (see `vigileye lenses`) or path to a custom lens YAML")
    p.add_argument("--query", default="", help="intelligence requirement / question to answer")
    p.add_argument("--platform", help="only analyze documents from this platform")
    p.add_argument("--since", help="only documents created after this ISO date")
    p.add_argument("--max-docs", type=int, dest="max_docs", help="cap documents loaded from the case")
    v = p.add_mutually_exclusive_group()
    v.add_argument("--vision", dest="vision", action="store_true", default=None, help="describe images with the vision role")
    v.add_argument("--no-vision", dest="vision", action="store_false")
    p.add_argument("--ach", action="store_true", default=None, help="run Analysis of Competing Hypotheses")
    vf = p.add_mutually_exclusive_group()
    vf.add_argument("--verify", dest="verify", action="store_true", default=None,
                    help="check that each cited source supports its key judgment (default: analysis.verify)")
    vf.add_argument("--no-verify", dest="verify", action="store_false", help="skip citation verification")
    p.add_argument("--critic", action="store_true", default=None,
                   help="add an LLM red-team pass (critic role) to the rule-based tradecraft critic")
    gc = p.add_mutually_exclusive_group()
    gc.add_argument("--ground-cves", dest="ground_cves", action="store_true", default=None,
                    help="check CVEs against CISA KEV and FIRST EPSS (network unless intel.offline)")
    gc.add_argument("--no-ground-cves", dest="ground_cves", action="store_false")
    p.add_argument("--formats", help="comma list: md,json,html,stix,navigator")
    p.add_argument("--tlp", choices=TLP_LEVELS, default=None,
                   help="TLP marking for the STIX export (default: output.tlp, else clear)")
    p.add_argument("--out", help="output directory")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vigileye", description="VigilEye — open-source intelligence analysis on any OpenAI-compatible model.")
    ap.add_argument("--version", action="version", version=f"vigileye {__version__}")
    ap.add_argument("--config", help="YAML config (default: $VIGILEYE_CONFIG or config/vigileye.yaml)")
    ap.add_argument("--store", help="case database path (default from config / $VIGILEYE_STORE)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    # legacy v1 flags
    ap.add_argument("--reddit", help=argparse.SUPPRESS)
    ap.add_argument("--max_tokens", type=int, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("collect", help="collect documents into a case")
    c.add_argument("targets", nargs="+", help="target specs, e.g. reddit:r/osint hn:search:ransomware rss:https://…")
    c.add_argument("--case", required=True)
    c.add_argument("--limit", type=int, default=100)
    c.add_argument("--opt", action="append", help="collector option key=value (repeatable)")

    a = sub.add_parser("analyze", help="analyze a case and produce a report")
    a.add_argument("--case", required=True)
    _add_analysis_flags(a)

    r = sub.add_parser("run", help="collect + analyze in one go")
    r.add_argument("targets", nargs="+")
    r.add_argument("--case")
    r.add_argument("--limit", type=int, default=100)
    r.add_argument("--opt", action="append")
    _add_analysis_flags(r)

    sub.add_parser("cases", help="list cases")
    rp = sub.add_parser("reports", help="list reports for a case")
    rp.add_argument("--case", required=True)
    sh = sub.add_parser("show", help="print a stored report")
    sh.add_argument("report_id", type=int)
    sh.add_argument("--format", choices=["md", "json", "html", "stix", "navigator"], default="md")
    sh.add_argument("--tlp", choices=TLP_LEVELS, default=None,
                    help="TLP marking for --format stix (default: output.tlp, else clear)")
    sub.add_parser("lenses", help="list analytic lenses")
    sub.add_parser("collectors", help="list collectors and target syntax")
    pv = sub.add_parser("providers", help="show providers/roles; --test pings every role")
    pv.add_argument("--test", action="store_true")
    ex = sub.add_parser("export", help="export a case as JSONL")
    ex.add_argument("--case", required=True)
    ex.add_argument("--out", required=True)
    sg = sub.add_parser("signals", help="compute deterministic evidence signals for a case (no LLM)")
    sg.add_argument("--case", required=True)
    sg.add_argument("--json", action="store_true", help="print signals and clusters as JSON")
    rd = sub.add_parser("refdata", help="reference data cache: CISA KEV and FIRST EPSS")
    rd_sub = rd.add_subparsers(dest="refdata_cmd", required=True)
    rd_sub.add_parser("update", help="download the CISA KEV catalog now")
    rc = rd_sub.add_parser("cve", help="show the KEV entry and EPSS score for one CVE")
    rc.add_argument("cve_id")
    sv = sub.add_parser("serve", help="run the JSON HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    return ap


def _print_report_summary(rid, report, paths) -> None:
    print(f"\nBLUF: {report.bluf}\n")
    for i, kj in enumerate(report.key_judgments, 1):
        print(f"  KJ{i}. {kj.title} — {kj.likelihood or 'n/a'}, {kj.confidence} confidence")
    print(f"\nreport #{rid} saved:")
    for p in paths:
        print(f"  {p}")


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING if not args.quiet else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.reddit and not args.cmd:
        _err("note: --reddit/--max_tokens is the v1 interface; use `vigileye run reddit:r/<sub>` instead.")
        args = ap.parse_args(["run", f"reddit:{args.reddit}"] + (["--config", args.config] if args.config else []))
    if not args.cmd:
        ap.print_help()
        return 2

    try:
        config = load_config(args.config)
        if args.cmd == "lenses":
            from .analysis import list_lenses
            for l in list_lenses():
                print(f"{l.name:<11} {l.title} — {l.description}")
            return 0
        if args.cmd == "collectors":
            from .collectors import list_collectors
            for c in list_collectors():
                print(f"{c['name']:<11} {c['description']}")
                for e in c.get("examples", []):
                    print(f"{'':<13}{e}")
            return 0
        if args.cmd == "providers":
            from .llm import LLMRouter
            router = LLMRouter.from_config(config)
            print(json.dumps(router.describe(), indent=2))
            if args.test:
                ok = True
                for role in ("analyst", "extractor", "vision", "synthesizer", "verifier", "critic"):
                    res = router.ping(role)
                    ok &= res["ok"]
                    print(f"{role:<12} " + (f"OK  {res['provider']}/{res['model']} {res['latency_s']}s" if res["ok"] else f"FAIL {res['error']}"))
                return 0 if ok else 1
            return 0
        if args.cmd == "refdata":
            from .intel import RefCache, lookup_cve, refresh
            cache = RefCache.from_config(config)
            if args.refdata_cmd == "update":
                res = refresh(cache)
                for e in res["errors"]:
                    _err(f"warning: {e}")
                print(f"CISA KEV catalog {res['catalogVersion'] or 'unavailable'}: {res['count']} entries"
                      + ("" if res["refreshed"] else " (not refreshed)"))
                return 0 if res["refreshed"] else 1
            res = lookup_cve(args.cve_id, cache)
            print(json.dumps(res, indent=2))
            return 0 if res["in_kev"] is not None or res["epss"] else 1
        if args.cmd == "serve":
            from .server import serve
            serve(config, args.host, args.port, store_path=args.store)
            return 0

        app = App(config, store_path=args.store, quiet=args.quiet)
        if args.cmd == "collect":
            counts = app.collect(args.targets, args.case, args.limit, _kv(args.opt))
            print(json.dumps(counts, indent=2))
            return 0 if any(v["collected"] for v in counts.values()) else 1
        if args.cmd in ("analyze", "run"):
            formats = _formats(args.formats)  # fail fast on a typo, before collecting or calling a model
            case = getattr(args, "case", None) or _slug(args.targets[0])
            if args.cmd == "run":
                app.collect(args.targets, case, args.limit, _kv(args.opt))
            since = datetime.fromisoformat(args.since) if args.since else None
            rid, report, paths = app.analyze(case, lens=args.lens, query=args.query, platform=args.platform, since=since,
                                             limit=args.max_docs, vision=args.vision, ach=args.ach,
                                             formats=formats, out=args.out, verify=args.verify,
                                             critic=args.critic, ground_cves=args.ground_cves, tlp=args.tlp)
            _print_report_summary(rid, report, paths)
            return 0
        if args.cmd == "cases":
            for c in app.store.list_cases():
                print(f"{c['name']:<30} docs={c.get('documents', 0):<6} reports={c.get('reports', 0):<4} last={c.get('last_activity')}")
            return 0
        if args.cmd == "reports":
            for r in app.store.list_reports(args.case):
                print(f"#{r['id']:<5} {r['lens']:<10} {r['created_at']}  {str(r.get('bluf', ''))[:90]}")
            return 0
        if args.cmd == "show":
            from .report import render_html, render_json, render_markdown
            rep = app.store.get_report(args.report_id)
            if rep is None:
                _err(f"no report #{args.report_id}")
                return 1
            if args.format in ("stix", "navigator"):
                from .export import report_to_navigator_layer, report_to_stix
                tlp = args.tlp or app.config.get("output.tlp", "clear") or "clear"
                out = report_to_stix(rep, tlp=tlp) if args.format == "stix" else report_to_navigator_layer(rep)
                print(json.dumps(out, indent=2, ensure_ascii=False))
            elif args.format == "html":
                print(render_html(rep), end="")
            else:
                print(render_markdown(rep) if args.format == "md" else render_json(rep))
            return 0
        if args.cmd == "signals":
            from .analysis import AnalysisEngine
            from .analysis.signals import compute_signals, signals_markdown
            docs = app.store.get_documents(args.case)
            if not docs:
                _err(f"case '{args.case}' has no documents")
                return 1
            res = compute_signals(docs, settings=AnalysisEngine(app.router, config).settings, case=args.case)
            for e in res["errors"]:
                _err(f"warning: {e}")
            print(json.dumps(res, indent=2, default=str) if args.json else signals_markdown(res["signals"]))
            return 0
        if args.cmd == "export":
            n = app.store.export_jsonl(args.case, args.out)
            print(f"exported {n} documents to {args.out}")
            return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        if getattr(args, "verbose", False):
            raise
        _err(f"error: {e}")
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
