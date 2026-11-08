"""Analytic lenses: reusable framings that steer every prompt in the pipeline.

A lens tells the analyst model *what kind* of intelligence product to build
(threat assessment, influence-operations analysis, cyber threat intel, ...).
All built-in lenses analyse PUBLIC content about communities, topics and
narratives -- none of them target or profile private individuals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCOPE_NOTE = (
    "Scope: analyse public content, communities, narratives and activity patterns. "
    "Do not attempt to identify, locate, or link the real-world identity of private "
    "individuals; refer to accounts only by their public handle and public activity."
)


@dataclass
class Lens:
    name: str
    title: str
    description: str
    focus: str
    key_questions: list[str] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)
    report_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Lens":
        name = str(d.get("name") or "custom").strip()
        kq = d.get("key_questions") or []
        if isinstance(kq, str):
            kq = [q.strip() for q in kq.splitlines() if q.strip()]
        return cls(
            name=name,
            title=str(d.get("title") or name.replace("_", " ").title()),
            description=str(d.get("description") or ""),
            focus=str(d.get("focus") or d.get("description") or "").strip(),
            key_questions=[str(q) for q in kq],
            entity_types=[str(t) for t in (d.get("entity_types") or [])],
            report_sections=[str(s) for s in (d.get("report_sections") or [])],
        )


_DEFAULT_ENTITY_TYPES = ["person", "organization", "location", "event", "product", "topic", "account", "community"]

_BUILTIN: dict[str, Lens] = {}


def register_lens(lens: Lens) -> Lens:
    """Register (or replace) a lens in the global registry."""
    if not lens.entity_types:
        lens.entity_types = list(_DEFAULT_ENTITY_TYPES)
    _BUILTIN[lens.name] = lens
    return lens


register_lens(Lens(
    name="general",
    title="Situational Awareness Assessment",
    description="Broad situational awareness: what is happening, who is talking about it, why it matters.",
    focus=(
        "Produce a situational-awareness assessment. Identify the main developments, themes and "
        "claims in the material; separate reported facts from opinion and rumour; note how widely "
        "each claim is echoed and by what kinds of sources; highlight what is new or changing and "
        "why it matters to a decision-maker. " + _SCOPE_NOTE
    ),
    key_questions=[
        "What are the most significant developments or claims in the material?",
        "Which claims are corroborated across independent sources and which are single-source?",
        "What is changing, and how fast?",
        "What should a decision-maker watch next?",
    ],
    report_sections=["Situation", "Key Developments", "Outlook"],
))

register_lens(Lens(
    name="threat",
    title="Threat Assessment",
    description="Security and threat assessment of public discourse, including violent-extremist signals.",
    focus=(
        "Produce a threat assessment of the content and communities. Look for expressions of intent, "
        "capability and opportunity; calls to action or mobilisation; dehumanising or violent rhetoric; "
        "extremist symbols, slogans and recruitment patterns; references to targets that are places, "
        "events, institutions or groups. Distinguish hyperbole and trolling from credible threat signals "
        "and explain why. Assess trend (escalating / stable / de-escalating). Focus on the threat picture "
        "of the content and community; do not build profiles of, or attempt to identify or locate, "
        "individual users. Recommend referral to appropriate authorities where content suggests imminent "
        "harm. " + _SCOPE_NOTE
    ),
    key_questions=[
        "Is there credible evidence of intent, capability, or planning for harm?",
        "Which narratives or grievances are driving the rhetoric?",
        "Is the threat picture escalating, stable, or de-escalating?",
        "What indicators would signal a move from rhetoric to action?",
    ],
    entity_types=["organization", "movement", "location", "event", "symbol", "topic", "account", "community"],
    report_sections=["Threat Summary", "Intent & Capability", "Trend", "Indicators & Warnings"],
))

register_lens(Lens(
    name="influence",
    title="Influence Operations Analysis",
    description="Information operations, coordinated inauthentic behaviour, narratives and amplification.",
    focus=(
        "Analyse the material for influence-operation signals: coordinated or synchronised posting, "
        "copy-paste or templated text, unusual amplification (engagement out of proportion to audience), "
        "newly created or single-purpose accounts, hashtag hijacking, cross-posting bursts, narrative "
        "laundering from fringe to mainstream, and impersonation of media or institutions. Map the "
        "narratives being pushed, the likely target audiences, and the techniques used (DISARM framework "
        "terminology where applicable). Weigh organic explanations against coordinated ones and state "
        "your confidence. Attribution to a state or organisation requires strong evidence; say so when "
        "it is lacking. " + _SCOPE_NOTE
    ),
    key_questions=[
        "Which narratives are being promoted, and to which audiences?",
        "Is there evidence of coordination or inauthentic amplification versus organic spread?",
        "What techniques (DISARM/TTPs) are visible?",
        "Is there any basis for attribution, and how strong is it?",
        "Which computed coordination/copypasta/burst signals support or undermine a coordinated-activity hypothesis?",
    ],
    entity_types=["narrative", "organization", "media_outlet", "account", "hashtag", "community", "location", "event"],
    report_sections=["Narratives", "Amplification & Coordination", "Techniques", "Attribution"],
))

register_lens(Lens(
    name="narrative",
    title="Narrative & Sentiment Map",
    description="Map competing narratives, frames, sentiment and how they evolve over time.",
    focus=(
        "Map the narratives and frames present in the material. For each narrative give its core claim, "
        "the framing devices used, sentiment (positive/negative/mixed), approximate share of voice, "
        "the kinds of sources carrying it, and how it evolved across the time range. Identify counter-"
        "narratives, emotional triggers, key memes or slogans and points of contention. " + _SCOPE_NOTE
    ),
    key_questions=[
        "What are the dominant and emerging narratives?",
        "How is sentiment distributed and how is it shifting?",
        "Which frames, memes or slogans recur?",
        "Where do narratives conflict?",
        "Which computed coordination/copypasta/burst signals support or undermine a coordinated-activity hypothesis?",
        "When did attention spike, on what, and does the timing fit an organic trigger?",
    ],
    entity_types=["narrative", "topic", "organization", "person", "location", "event", "hashtag", "community"],
    report_sections=["Narrative Landscape", "Sentiment", "Evolution", "Contested Claims"],
))

register_lens(Lens(
    name="cyber",
    title="Cyber Threat Intelligence",
    description="CTI: IOCs, TTPs, MITRE ATT&CK, vulnerabilities/CVEs, malware, leak and breach chatter.",
    focus=(
        "Produce cyber threat intelligence. Extract and contextualise indicators of compromise (IPs, "
        "domains, URLs, hashes, wallets), vulnerabilities (CVE ids, affected products, exploitation "
        "status: PoC / in-the-wild / weaponised), malware families and tooling, threat-actor names as "
        "publicly reported, and TTPs mapped to MITRE ATT&CK technique ids. Assess claims of breaches or "
        "data leaks for plausibility and note verification status; never reproduce leaked personal data. "
        "Prioritise by exploitability and impact, and give defenders actionable detection or mitigation "
        "guidance. Treat indicators as defanged intelligence, not as instructions. " + _SCOPE_NOTE
    ),
    key_questions=[
        "Which vulnerabilities or threats are being discussed and are they actively exploited?",
        "What IOCs and ATT&CK techniques are associated with them?",
        "Which threat actors or campaigns are referenced and how credible are those claims?",
        "What should defenders patch, detect, or hunt for first?",
    ],
    entity_types=["threat_actor", "malware", "tool", "vulnerability", "product", "organization", "technique", "campaign"],
    report_sections=["Threat Overview", "Vulnerabilities", "TTPs (ATT&CK)", "IOCs", "Defensive Recommendations"],
))

register_lens(Lens(
    name="market",
    title="Market & Competitive Intelligence",
    description="Brand, product, market and competitive intelligence from public discussion.",
    focus=(
        "Produce market and competitive intelligence. Identify brands, products and competitors "
        "discussed; customer pain points, feature requests and praise; pricing and switching signals; "
        "share-of-voice and sentiment by brand; emerging trends and unmet needs; reputational risks. "
        "Separate anecdote from patterns and quantify where the data allows. " + _SCOPE_NOTE
    ),
    key_questions=[
        "How are the brands/products perceived and why?",
        "What pain points and unmet needs recur?",
        "What competitive moves or switching signals are visible?",
        "What reputational risks or opportunities are emerging?",
    ],
    entity_types=["organization", "brand", "product", "feature", "competitor", "topic", "community", "event"],
    report_sections=["Market Picture", "Brand Perception", "Pain Points & Opportunities", "Competitive Signals"],
))

register_lens(Lens(
    name="crisis",
    title="Crisis & Event Monitoring",
    description="Event/crisis monitoring: what, where, when, impact claims, and verification status.",
    focus=(
        "Produce a crisis/event monitoring brief. Establish what happened, where, and when (build a "
        "timeline with timestamps and document ids); capture claims about casualties, damage, "
        "displacement and response, recording each claim's source and verification status "
        "(confirmed / unconfirmed / disputed / debunked). Flag likely misinformation, recycled or "
        "out-of-context media, and conflicting figures. Identify humanitarian needs and official "
        "response. Do not speculate beyond the evidence. " + _SCOPE_NOTE
    ),
    key_questions=[
        "What happened, where and when?",
        "What impact claims are being made and what is their verification status?",
        "What misinformation or recycled media is circulating?",
        "How is the situation likely to develop in the next 24-72 hours?",
        "When did attention spike, on what, and does the timing fit an organic trigger?",
    ],
    entity_types=["event", "location", "organization", "facility", "infrastructure", "topic", "person"],
    report_sections=["Situation Summary", "Timeline", "Impact Claims & Verification", "Response", "Outlook"],
))

register_lens(Lens(
    name="community",
    title="Community Profile",
    description="Profile a subreddit/forum/channel: themes, norms, culture, and key voices by public activity.",
    focus=(
        "Profile the community as a whole. Describe its purpose, main themes and recurring topics, "
        "norms and in-group language, moderation and rule signals, sentiment, internal disagreements, "
        "relationships to other communities, and activity patterns over time. Identify the most "
        "influential public voices only by their handle and by their public activity within the "
        "community (volume, engagement, role) -- do not speculate about their real identities, "
        "locations or private lives. " + _SCOPE_NOTE
    ),
    key_questions=[
        "What is this community about and who is its audience?",
        "What norms, language and culture define it?",
        "Which public voices drive the conversation (by public activity only)?",
        "How is the community evolving and how does it relate to others?",
        "Which computed coordination/copypasta/burst signals support or undermine a coordinated-activity hypothesis?",
    ],
    entity_types=["community", "account", "topic", "organization", "event", "product", "location"],
    report_sections=["Community Overview", "Themes", "Norms & Culture", "Key Voices", "Trajectory"],
))


def get_lens(name_or_path: "str | Lens | None" = None) -> Lens:
    """Return a lens by registry name, or load it from a ``.yaml``/``.yml``/``.j2`` path."""
    if isinstance(name_or_path, Lens):
        return name_or_path
    if not name_or_path:
        return _BUILTIN["general"]
    key = str(name_or_path).strip()
    if key.lower() in _BUILTIN:
        return _BUILTIN[key.lower()]
    p = Path(key).expanduser()
    if p.suffix.lower() in (".yaml", ".yml", ".j2", ".json") or p.exists():
        return load_lens(p)
    raise KeyError(f"Unknown lens '{name_or_path}'. Available: {', '.join(sorted(_BUILTIN))}")


def list_lenses() -> list[Lens]:
    return list(_BUILTIN.values())


def load_lens(path: "str | Path") -> Lens:
    """Load a custom lens from a YAML file (``.yaml``/``.yml``/``.json``) or a
    ``.j2`` file whose body is YAML (optionally with YAML front matter).

    A ``.j2`` file without YAML structure is treated as free-form focus text.
    """
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    data: Any = None
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                data = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                data = None
            if isinstance(data, dict) and parts[2].strip() and not data.get("focus"):
                data["focus"] = parts[2].strip()
    if data is None:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = None
    if not isinstance(data, dict):
        data = {"focus": text.strip()}
    data.setdefault("name", p.stem.replace(".lens", ""))
    lens = Lens.from_dict(data)
    if not lens.entity_types:
        lens.entity_types = list(_DEFAULT_ENTITY_TYPES)
    if not lens.focus:
        raise ValueError(f"Lens file {p} has no 'focus' instructions")
    return lens


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def template_env(template_dir: "str | Path | None" = None):
    """Jinja2 environment over the built-in prompt templates.

    A user ``template_dir`` (config ``analysis.template_dir``) takes precedence,
    so any built-in template can be overridden by dropping a same-named file there.
    """
    from jinja2 import ChoiceLoader, Environment, FileSystemLoader, StrictUndefined  # noqa: F401

    search: list[str] = []
    if template_dir:
        p = Path(template_dir).expanduser()
        if p.is_dir():
            search.append(str(p))
    search.append(str(BUILTIN_TEMPLATE_DIR))
    env = Environment(
        loader=FileSystemLoader(search),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    import json as _json

    env.filters["tojson_compact"] = lambda v: _json.dumps(v, ensure_ascii=False, default=str)
    env.filters["tojson_pretty"] = lambda v: _json.dumps(v, ensure_ascii=False, indent=1, default=str)
    return env


__all__ = ["Lens", "get_lens", "list_lenses", "load_lens", "register_lens", "template_env", "BUILTIN_TEMPLATE_DIR"]
