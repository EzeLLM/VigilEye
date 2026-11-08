"""Report renderers (Markdown, JSON, HTML) and file helpers. Stdlib only."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import Report

BANNER = "UNCLASSIFIED // OPEN SOURCE"
DISCLAIMER = (
    "Judgments in this report are machine-generated from open-source material and have not been "
    "validated. They require human review before any decision or dissemination."
)
_DEFANG_TYPES = {"url", "domain", "ipv4", "ipv6"}


def defang(value: Any, kind: str | None = None) -> str:
    s = str(value if value is not None else "")
    if kind is not None and str(kind).lower() not in _DEFANG_TYPES:
        return s
    s = re.sub(r"(?i)\bhttp", lambda m: "hxxp" if m.group(0).islower() else "HXXP", s)
    return s.replace(".", "[.]")


def _txt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return "; ".join(_txt(x) for x in v)
    return str(v)


def _cell(v: Any) -> str:
    return _txt(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _hyp(h: Any) -> dict:
    return h if isinstance(h, dict) else {"hypothesis": _txt(h)}


def _ach_meta(r: Report) -> dict:
    ach = (r.meta or {}).get("ach") if isinstance(r.meta, dict) else None
    return ach if isinstance(ach, dict) else {}


def _top_entities(report: Report, n: int = 50):
    return sorted(report.entities, key=lambda e: -(e.mentions or 0))[:n]


def _usage_totals(stats: dict) -> dict:
    for key in ("llm_usage", "usage"):
        u = stats.get(key)
        if isinstance(u, dict):
            t = u.get("totals", u)
            if isinstance(t, dict):
                return t
    return {}


def _custody_line(r: Report) -> str | None:
    """One plain-text line for ``stats["custody"]`` (evidence.py): the custody chain head at analysis time."""
    c = (r.stats or {}).get("custody")
    if not isinstance(c, dict) or not c.get("chain_head"):
        return None
    parts = [f"chain head {c['chain_head']}", f"{c.get('chain_entries', '?')} custody entries",
             "chain verified" if c.get("chain_ok") else "CHAIN BROKEN"]
    if "documents" in c:
        parts.append(f"{c.get('documents_stamped', 0)}/{c['documents']} analysed documents hash-stamped at collection")
    return "; ".join(parts) + f" (check: `vigileye evidence verify --case {r.case} --anchor <chain head>`)"


_STATS_HIDDEN = ("llm_usage", "usage", "custody")


def _extra_sections(report: Report) -> list[dict]:
    """``meta['extra_sections']`` entries with markdown (shared hook H0 for computed/added sections)."""
    out = []
    for sec in (report.meta or {}).get("extra_sections") or []:
        if isinstance(sec, dict) and str(sec.get("markdown") or "").strip():
            out.append(sec)
    return out


def _md_table(headers: list[str], rows: Iterable[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


# ----------------------------------------------------------------------------- markdown
def render_markdown(report: Report) -> str:
    r = report
    date = r.created_at.strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = [f"**{BANNER}**", "", f"# {r.case}: {r.lens} assessment", "",
                    f"- **Case:** {r.case}", f"- **Lens:** {r.lens}", f"- **Query:** {r.query}",
                    f"- **Date:** {date}", "", "## Bottom Line Up Front", "", r.bluf or "_None._", ""]
    L += ["## Key Judgments", ""]
    if r.key_judgments:
        for i, k in enumerate(r.key_judgments, 1):
            meta = f"confidence: {k.confidence}"
            if k.likelihood:
                meta += f"; likelihood: {k.likelihood}"
            L.append(f"{i}. **{k.title}** ({meta}) {k.summary}".rstrip())
            if k.evidence:
                L.append(f"   - Evidence: {', '.join(f'[{e}]' for e in k.evidence)}")
    else:
        L.append("_No key judgments._")
    L.append("")
    if (r.body_markdown or "").strip():
        L += [r.body_markdown.strip(), ""]
    if r.hypotheses:
        L += ["## Analysis of Competing Hypotheses", ""]
        L += _md_table(["#", "Hypothesis", "Likelihood", "Confidence", "Inconsistent items", "Consistent", "Inconsistent",
                        "Notes"],
                       [[i, h.get("hypothesis", ""), h.get("likelihood") or "not estimated", h.get("confidence", ""),
                         h.get("inconsistency_score", ""), h.get("consistent", ""), h.get("inconsistent", ""),
                         h.get("notes", "")]
                        for i, h in enumerate(map(_hyp, r.hypotheses), 1)])
        L.append("")
        ach = _ach_meta(r)
        if ach.get("conclusion"):
            L += [f"**Conclusion:** {_txt(ach['conclusion'])}", ""]
        if ach.get("evidence_items"):
            L += ["_Evidence items (ranked by least inconsistent evidence, after Heuer):_", ""]
            L += [f"- **{k}**: {_txt(v)}" for k, v in ach["evidence_items"].items()] + [""]
    for title, items in (("Key Assumptions", r.assumptions), ("Intelligence Gaps", r.gaps)):
        L += [f"## {title}", ""] + ([f"- {_txt(x)}" for x in items] or ["_None identified._"]) + [""]
    if r.indicators:
        L += ["## Indicators", "", "_Network indicators are defanged._", ""]
        L += _md_table(["Type", "Value", "Count"],
                       [[i.get("type", ""), defang(i.get("value", ""), i.get("type", "")), i.get("count", "")]
                        for i in r.indicators if isinstance(i, dict)])
        L.append("")
    if r.entities:
        L += ["## Entities", ""]
        L += _md_table(["Name", "Type", "Mentions", "Aliases"],
                       [[e.name, e.type, e.mentions, ", ".join(e.aliases)] for e in _top_entities(r)])
        L.append("")
    if r.sources:
        L += ["## Sources", ""]
        L += _md_table(["ID", "Platform", "Source", "Grade", "URL"],
                       [[s.get("id", ""), s.get("platform", ""), s.get("source", ""), s.get("grade", ""),
                         s.get("url", "")] for s in r.sources if isinstance(s, dict)])
        L.append("")
    for sec in _extra_sections(r):
        L += [f"## {sec.get('title') or sec.get('id')}", "", str(sec["markdown"]).strip(), ""]
    L += ["## Methodology", "", f"Lens `{r.lens}` applied to query: {r.query}", ""]
    custody = _custody_line(r)
    if custody:
        L.append(f"- **Chain of custody:** {custody}")
    stats = {k: v for k, v in (r.stats or {}).items() if k not in _STATS_HIDDEN or (k == "custody" and not custody)}
    for k, v in stats.items():
        L.append(f"- **{k}:** {_txt(json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)}")
    totals = _usage_totals(r.stats or {})
    if totals:
        L.append("- **LLM usage:** " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    L += ["", "---", "", f"_{DISCLAIMER}_", "", f"**{BANNER}**", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- json
def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2, default=str, ensure_ascii=False)


# ----------------------------------------------------------------------------- html
_E = html.escape


def _inline(s: str) -> str:
    """Inline markdown on an ALREADY-ESCAPED string."""
    s = s.replace("\x00", "\ufffd")  # NUL delimits code-span placeholders; collected text may carry \u0000
    codes: list[str] = []

    def stash(m):
        codes.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)

    def link(m):
        url = m.group(2)
        if not re.match(r"(?i)^https?://", url):
            return m.group(0)
        return f'<a href="{url}" rel="noopener noreferrer">{m.group(1)}</a>'

    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"__(.+?)__", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"<em>\1</em>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)


def md_to_html(text: str) -> str:
    out: list[str] = []
    para: list[str] = []
    table: list[list[str]] = []
    list_tag: str | None = None

    def flush_para():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def flush_table():
        if table:
            head, *body = table
            out.append('<div class="tw"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in body)
                       + "</tbody></table></div>")
            table.clear()

    for raw in (text or "").replace("\x00", "\ufffd").splitlines():
        line = _E(raw.rstrip())
        stripped = line.strip()
        if not stripped:
            flush_para(); close_list(); flush_table(); continue
        if len(stripped) > 1 and stripped.startswith("|") and stripped.endswith("|"):
            flush_para(); close_list()
            cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", stripped[1:-1])]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):  # skip the |---|---| separator row
                table.append(cells)
            continue
        flush_table()
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para(); close_list()
            lvl = min(6, len(m.group(1)) + 1)
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            continue
        ul = re.match(r"^[-*+]\s+(.*)$", stripped)
        ol = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if ul or ol:
            flush_para()
            tag = "ul" if ul else "ol"
            if list_tag != tag:
                close_list(); out.append(f"<{tag}>"); list_tag = tag
            out.append(f"<li>{_inline((ul or ol).group(1))}</li>")
            continue
        close_list()
        para.append(stripped)
    flush_para(); close_list(); flush_table()
    return "\n".join(out)


def _h_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    head = "".join(f"<th>{_E(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_E(_txt(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


_CSS = """
:root{--bg:#fff;--fg:#1b1f24;--mut:#5b636e;--bd:#d8dde3;--acc:#0b5cad;--ban:#007a33;--alt:#f5f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e4e7eb;--mut:#9aa3ad;--bd:#2c333b;
--acc:#6cb0f5;--ban:#2fa35a;--alt:#181c21}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
.banner{background:var(--ban);color:#fff;text-align:center;font-weight:700;letter-spacing:.08em;
padding:4px 8px;font-size:12px}
h1{font-size:1.6em;margin:.6em 0 .2em}h2{border-bottom:1px solid var(--bd);padding-bottom:4px;margin-top:1.6em}
.meta{color:var(--mut);font-size:.9em}.meta b{color:var(--fg)}
.bluf{border-left:4px solid var(--acc);background:var(--alt);padding:10px 14px}
a{color:var(--acc)}code{background:var(--alt);padding:1px 4px;border-radius:3px;font-size:.9em}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9em}
th,td{border:1px solid var(--bd);padding:5px 8px;text-align:left;vertical-align:top;word-break:break-word}
th{background:var(--alt)}.kj li{margin-bottom:8px}.ev{color:var(--mut);font-size:.85em}
.disc{color:var(--mut);font-size:.85em;border-top:1px solid var(--bd);margin-top:2em;padding-top:8px}
@media print{:root{--bg:#fff;--fg:#000;--mut:#333;--bd:#999;--alt:#eee;--acc:#000}
.banner{-webkit-print-color-adjust:exact;print-color-adjust:exact}a{text-decoration:none}
h2{break-after:avoid}tr{break-inside:avoid}main{max-width:none;padding:0}}
"""


def render_html(report: Report) -> str:
    r = report
    date = r.created_at.strftime("%Y-%m-%d %H:%M UTC")
    P: list[str] = [f"<h1>{_E(r.case)}: {_E(r.lens)} assessment</h1>",
                    f'<p class="meta"><b>Case:</b> {_E(r.case)} &middot; <b>Lens:</b> {_E(r.lens)} &middot; '
                    f"<b>Query:</b> {_E(r.query)} &middot; <b>Date:</b> {_E(date)}</p>",
                    "<h2>Bottom Line Up Front</h2>", f'<div class="bluf">{_E(r.bluf or "None.")}</div>',
                    "<h2>Key Judgments</h2>"]
    if r.key_judgments:
        P.append('<ol class="kj">')
        for k in r.key_judgments:
            meta = f"confidence: {k.confidence}" + (f"; likelihood: {k.likelihood}" if k.likelihood else "")
            ev = f'<div class="ev">Evidence: {_E(", ".join(k.evidence))}</div>' if k.evidence else ""
            P.append(f"<li><strong>{_E(k.title)}</strong> <em>({_E(meta)})</em> {_E(k.summary)}{ev}</li>")
        P.append("</ol>")
    else:
        P.append("<p><em>No key judgments.</em></p>")
    if (r.body_markdown or "").strip():
        P.append(f'<section class="body">{md_to_html(r.body_markdown)}</section>')
    if r.hypotheses:
        P += ["<h2>Analysis of Competing Hypotheses</h2>", _h_table(
            ["#", "Hypothesis", "Likelihood", "Confidence", "Inconsistent items", "Consistent", "Inconsistent", "Notes"],
            [[i, h.get("hypothesis", ""), h.get("likelihood") or "not estimated", h.get("confidence", ""),
              h.get("inconsistency_score", ""), h.get("consistent", ""), h.get("inconsistent", ""), h.get("notes", "")]
             for i, h in enumerate(map(_hyp, r.hypotheses), 1)])]
        ach = _ach_meta(r)
        if ach.get("conclusion"):
            P.append(f"<p><strong>Conclusion:</strong> {_E(_txt(ach['conclusion']))}</p>")
        if ach.get("evidence_items"):
            P.append("<p><em>Evidence items (ranked by least inconsistent evidence, after Heuer):</em></p><ul>"
                     + "".join(f"<li><strong>{_E(_txt(k))}</strong>: {_E(_txt(v))}</li>"
                               for k, v in ach["evidence_items"].items()) + "</ul>")
    for title, items in (("Key Assumptions", r.assumptions), ("Intelligence Gaps", r.gaps)):
        P.append(f"<h2>{title}</h2>")
        P.append("<ul>" + "".join(f"<li>{_E(_txt(x))}</li>" for x in items) + "</ul>" if items
                 else "<p><em>None identified.</em></p>")
    if r.indicators:
        P += ["<h2>Indicators</h2>", '<p class="meta">Network indicators are defanged.</p>', _h_table(
            ["Type", "Value", "Count"],
            [[i.get("type", ""), defang(i.get("value", ""), i.get("type", "")), i.get("count", "")]
             for i in r.indicators if isinstance(i, dict)])]
    if r.entities:
        P += ["<h2>Entities</h2>", _h_table(["Name", "Type", "Mentions", "Aliases"],
                                            [[e.name, e.type, e.mentions, ", ".join(e.aliases)]
                                             for e in _top_entities(r)])]
    if r.sources:
        rows = []
        for s in r.sources:
            if not isinstance(s, dict):
                continue
            url = _txt(s.get("url"))
            link = (f'<a href="{_E(url)}" rel="noopener noreferrer">{_E(url)}</a>'
                    if re.match(r"(?i)^https?://", url) else _E(url))
            rows.append("<tr>" + "".join(f"<td>{_E(_txt(s.get(k, '')))}</td>"
                                         for k in ("id", "platform", "source", "grade")) + f"<td>{link}</td></tr>")
        P += ["<h2>Sources</h2>", '<div class="tw"><table><thead><tr><th>ID</th><th>Platform</th><th>Source</th>'
              f'<th>Grade</th><th>URL</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>']
    for sec in _extra_sections(r):
        P += [f"<h2>{_E(str(sec.get('title') or sec.get('id')))}</h2>",
              f'<section class="extra">{md_to_html(str(sec["markdown"]))}</section>']
    P += ["<h2>Methodology</h2>", f"<p>Lens <code>{_E(r.lens)}</code> applied to query: {_E(r.query)}</p>"]
    custody = _custody_line(r)
    stats = ([("Chain of custody", custody)] if custody else []) + [
        (k, v) for k, v in (r.stats or {}).items() if k not in _STATS_HIDDEN or (k == "custody" and not custody)]
    totals = _usage_totals(r.stats or {})
    if totals:
        stats.append(("LLM usage", ", ".join(f"{k}={v}" for k, v in totals.items())))
    if stats:
        P.append(_h_table(["Metric", "Value"],
                          [[k, json.dumps(v, default=str) if isinstance(v, (dict, list)) else v] for k, v in stats]))
    P.append(f'<p class="disc">{_E(DISCLAIMER)}</p>')
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_E(r.case)} - {_E(r.lens)}</title><style>{_CSS}</style></head><body>"
            f'<div class="banner">{BANNER}</div><main>{"".join(P)}</main>'
            f'<div class="banner">{BANNER}</div></body></html>\n')


# ----------------------------------------------------------------------------- files
_RENDERERS = {"md": render_markdown, "json": render_json, "html": render_html}
_EXT = {"stix": "stix.json", "navigator": "layer.json"}
FORMATS = ("md", "json", "html", "stix", "navigator")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower() or "x"


def write_report(report: Report, out_dir, formats=("md", "json", "html"), basename: str | None = None,
                 *, tlp: str = "clear") -> list[Path]:
    """Write ``report`` in each of ``formats`` (md, json, html, stix, navigator); ``tlp`` marks the STIX bundle."""
    from .export import report_to_navigator_layer, report_to_stix  # lazy: keeps report.py import-light, no cycles

    renderers = {**_RENDERERS,
                 "stix": lambda r: json.dumps(report_to_stix(r, tlp=tlp), indent=2, ensure_ascii=False),
                 "navigator": lambda r: json.dumps(report_to_navigator_layer(r), indent=2, ensure_ascii=False)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = basename or f"{_slug(report.case)}-{_slug(report.lens)}-{report.created_at.strftime('%Y%m%d-%H%M%S')}"
    paths = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        fmt = "md" if fmt == "markdown" else fmt
        if fmt not in renderers:
            raise ValueError(f"unknown report format: {fmt}")
        p = out / f"{base}.{_EXT.get(fmt, fmt)}"
        p.write_text(renderers[fmt](report), encoding="utf-8")
        paths.append(p)
    return paths


def load_report(path) -> Report:
    return Report.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["render_markdown", "render_json", "render_html", "write_report", "load_report", "defang", "md_to_html",
           "FORMATS"]
