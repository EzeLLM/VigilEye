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
        from .evidence import EvidenceLedger

        self.store = CaseStore(store_path or config.get("store.path", ".vigileye/cases.db"))
        self.ledger = EvidenceLedger(self.store)  # chain of custody, same SQLite file as the store
        self.custody = bool(config.get("evidence.custody", True))
        self.actor = str(config.get("evidence.actor") or "vigileye")
        self.router = LLMRouter.from_config(config)
        if config.get("llm.cache", True):
            self.router.set_cache_backend(self.store.cache_get, self.store.cache_put)

    def progress(self, msg: str) -> None:
        if not self.quiet:
            _err(f"  · {msg}")

    def record(self, case: str, action: str, subject: str | None = None, details: dict | None = None) -> None:
        """Append a custody entry (no-op with ``evidence.custody: false``)."""
        if self.custody:
            self.ledger.record(case, action, subject=subject, details=details, actor=self.actor)

    def collect(self, targets: list[str], case: str, limit: int = 100, opts: dict | None = None, *,
                trust: str = "manual") -> dict:
        """Collect ``targets`` into ``case``. ``trust`` names where the targets came from
        (``manual``, ``api`` or ``auto``); see :func:`vigileye.collectors.base.collect`."""
        from .collectors import collect
        from .evidence import collect_details, stamp

        self.store.ensure_case(case)
        counts = {}
        for spec in targets:
            docs, failed = [], None
            try:
                for d in collect(spec, limit=limit, config=self.config.section("collectors"), trust=trust,
                                 **(opts or {})):
                    docs.append(d)
            except Exception as e:  # keep partial results, report the failure
                failed = str(e)
            if docs and self.custody:  # hash every item at collection; unchanged re-collections keep their stamp
                stamp(docs, spec=spec, existing=self.ledger.stamps(case, [d.fingerprint() for d in docs]))
            new = self.store.add_documents(case, docs) if docs else 0
            self.record(case, "collect", subject=spec,
                        details=collect_details(docs, new=new, **({"error": failed} if failed else {})))
            counts[spec] = {"collected": len(docs), "new": new, **({"error": failed} if failed else {})}
            self.progress(f"{spec}: {len(docs)} collected, {new} new" + (f" (error: {failed})" if failed else ""))
        return counts

    def analyze(self, case: str, *, lens: str | None = None, query: str = "", platform: str | None = None,
                since: datetime | None = None, limit: int | None = None, vision: bool | None = None,
                ach: bool | None = None, formats: list[str] | None = None, out: str | None = None,
                verify: bool | None = None, critic: bool | None = None, ground_cves: bool | None = None,
                tlp: str | None = None) -> tuple:
        from .analysis import AnalysisEngine
        from .evidence import file_hashes, report_sha256
        from .report import write_report

        docs = self.store.get_documents(case, platform=platform, since=since, limit=limit)
        if not docs:
            raise RuntimeError(f"case '{case}' has no documents — run `vigileye collect` first")
        self.progress(f"analyzing {len(docs)} documents from case '{case}'")
        engine = AnalysisEngine(self.router, self.config)
        report = engine.run(docs, lens=lens or self.config.get("analysis.lens", "general"), query=query, case=case,
                            use_vision=vision, ach=ach, verify=verify, critic=critic, ground_cves=ground_cves,
                            progress=self.progress)
        if self.custody:
            report.stats["custody"] = self.ledger.custody_summary(case, docs)  # rendered in Methodology
        rid = self.store.save_report(report)
        self.record(case, "analyze", subject=f"report:{rid}",
                    details={"report_id": rid, "lens": report.lens, "documents": len(docs),
                             "report_sha256": report_sha256(report)})
        paths = write_report(report, out or self.config.get("output.dir", "export"),
                             formats=tuple(formats or self.config.get("output.formats", ["md", "json", "html"])),
                             tlp=tlp or self.config.get("output.tlp", "clear") or "clear")
        self.record(case, "export", subject=f"report:{rid}", details={"report_id": rid, "files": file_hashes(paths)})
        return rid, report, paths

    def amend_report(self, rid: int | None, report, paths=()) -> int:
        """Store a changed report in place (a new row if ``rid`` is gone) and log it, so evidence
        verification checks the stored report against its newest hash. Returns the report id."""
        from .evidence import file_hashes, report_sha256

        if rid is not None and self.store.update_report(rid, report):
            action = "amend"
        else:
            rid, action = self.store.save_report(report), "analyze"
        self.record(report.case, action, subject=f"report:{rid}",
                    details={"report_id": rid, "lens": report.lens, "report_sha256": report_sha256(report)})
        if paths:
            self.record(report.case, "export", subject=f"report:{rid}",
                        details={"report_id": rid, "files": file_hashes(paths)})
        return rid

    def ask(self, case: str, question: str, *, k: int = 12, verify: bool = True, platform: str | None = None):
        from .ask import ask as _ask

        if not self.store.count_documents(case):
            raise RuntimeError(f"case '{case}' has no documents — run `vigileye collect` first")
        self.progress(f"retrieving up to {k} documents from case '{case}'")
        return _ask(self.store, case, question, self.router, k=k, verify=verify, platform=platform,
                    verify_llm=bool(self.config.get("analysis.verify_llm", True)))


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

    iv = sub.add_parser("investigate", help="iterative, policy-gated collect -> analyze -> plan loop")
    iv.add_argument("question", help="the investigation question (also the analysis query)")
    iv.add_argument("--seed", dest="seeds", action="append", required=True, metavar="SPEC",
                    help="seed target spec (repeatable)")
    iv.add_argument("--case", required=True)
    iv.add_argument("--rounds", type=int, default=3, help="planner rounds after the seed round (default 3)")
    iv.add_argument("--max-new", dest="max_new", type=int, default=4, help="new targets per round (default 4)")
    iv.add_argument("--limit", type=int, default=50, help="documents per target (default 50)")
    iv.add_argument("--opt", action="append", help="collector option key=value (repeatable)")
    iv.add_argument("--yes", action="store_true", help="collect policy-approved planner targets without asking")
    _add_analysis_flags(iv)  # --max-docs is the case document budget here (default 2000); --query is unused

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
    ak = sub.add_parser("ask", help="answer a question from a case's documents with verified citations")
    ak.add_argument("question", nargs="+", help="the question (quote it)")
    ak.add_argument("--case", required=True)
    ak.add_argument("--k", type=int, default=12, help="documents to retrieve (default 12)")
    ak.add_argument("--platform", help="only retrieve documents from this platform")
    akv = ak.add_mutually_exclusive_group()
    akv.add_argument("--verify", dest="verify", action="store_true", default=True,
                     help="check every citation against its source (default)")
    akv.add_argument("--no-verify", dest="verify", action="store_false")
    ak.add_argument("--json", action="store_true", help="print the answer as JSON")
    evp = sub.add_parser("evidence", help="chain of custody: verify, manifest, log")
    evp.add_argument("evidence_cmd", choices=["verify", "manifest", "log"])
    evp.add_argument("--case", required=True)
    evp.add_argument("--out", help="manifest: write JSON + .sha256 sidecar here")
    evp.add_argument("--anchor", help="verify: chain head recorded earlier (e.g. from a manifest)")
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
                if router.has_role("embedder"):  # embedding models reject chat, so ping() cannot test them
                    try:
                        emb = router.embeddings(["ping"])
                        print(f"{'embedder':<12} OK  {emb.label} dim={len(emb.vectors[0])} {emb.latency_s}s")
                    except Exception as e:  # noqa: BLE001
                        ok = False
                        print(f"{'embedder':<12} FAIL {e}")
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
        if args.cmd == "investigate":
            from .investigate import Investigation, prompt_approver

            formats = _formats(args.formats)
            if args.yes:
                approve = None
            elif sys.stdin.isatty():
                approve = prompt_approver()
            else:
                _err("note: stdin is not a TTY and --yes was not given: planner targets will be declined")
                approve = lambda cands: []  # noqa: E731
            since = datetime.fromisoformat(args.since) if args.since else None
            res = Investigation(
                app, case=args.case, question=args.question,
                lens=args.lens or config.get("analysis.lens", "general"), seeds=args.seeds,
                rounds=args.rounds, max_new_targets=args.max_new, limit_per_target=args.limit,
                max_docs=args.max_docs or 2000, approve=approve, progress=app.progress,
                seed_mode="manual", allow_files=True, opts=_kv(args.opt),
                analyze_kwargs={"platform": args.platform, "since": since, "vision": args.vision, "ach": args.ach,
                                "verify": args.verify, "critic": args.critic, "ground_cves": args.ground_cves,
                                "formats": formats, "out": args.out, "tlp": args.tlp}).run()
            for e in res.errors:
                _err(f"warning: {e}")
            if res.report is None:
                _err(f"error: investigation stopped ({res.stopped_reason}) before any analysis")
                return 1
            _print_report_summary(res.report_id, res.report, res.paths)
            print(f"stopped: {res.stopped_reason} after {len(res.rounds) - 1} planner round(s)")
            return 0
        if args.cmd == "ask":
            from .ask import render_answer

            ans = app.ask(args.case, " ".join(args.question), k=args.k, verify=args.verify, platform=args.platform)
            print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False, default=str) if args.json
                  else render_answer(ans), end="\n" if args.json else "")
            return 0
        if args.cmd == "evidence":
            from .evidence import format_entry

            if args.evidence_cmd == "verify":
                res = app.ledger.verify_case(args.case, anchor=args.anchor)
                print(json.dumps(res, indent=2))
                return 0 if res["ok"] else 1
            if args.evidence_cmd == "manifest":
                if args.out:
                    res = app.ledger.write_manifest(args.case, args.out, record=app.custody)
                    print(f"manifest {res['path']} sha256={res['sha256']} chain_head={res['chain_head']}")
                else:
                    print(json.dumps(app.ledger.manifest(args.case), indent=2))
                return 0
            entries = app.ledger.entries(args.case)
            if not entries:
                _err(f"case '{args.case}' has no custody entries")
                return 1
            for e in entries:
                print(format_entry(e))
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
            from .evidence import file_hashes

            n = app.store.export_jsonl(args.case, args.out)
            app.record(args.case, "export", subject=Path(args.out).name,
                       details={"documents": n, "files": file_hashes([args.out])})
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
