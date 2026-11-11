"""Offline tests for the collectors subsystem (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from vigileye.collectors import (
    CollectorError,
    get_collector,
    list_collectors,
    parse_feed,
    resolve_target,
    strip_html,
    to_utc,
)
from vigileye.collectors import bluesky, files, hackernews, mastodon, reddit, web
from vigileye.models import Document


# --------------------------------------------------------------------------- #
# resolve_target
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec,expected",
    [
        ("reddit:r/osint", ("reddit", "r/osint")),
        ("reddit:u/name", ("reddit", "u/name")),
        ("reddit:search:some query", ("reddit", "search:some query")),
        ("r/osint", ("reddit", "r/osint")),
        ("u/name", ("reddit", "u/name")),
        ("/r/osint/", ("reddit", "r/osint")),
        ("hn:search:llm agents", ("hackernews", "search:llm agents")),
        ("hn:user:pg", ("hackernews", "user:pg")),
        ("hn:item:123", ("hackernews", "item:123")),
        ("rss:https://example.com/feed.xml", ("rss", "https://example.com/feed.xml")),
        ("web:https://example.com/page", ("web", "https://example.com/page")),
        ("https://example.com/page", ("web", "https://example.com/page")),
        ("https://example.com/feed.xml", ("rss", "https://example.com/feed.xml")),
        ("mastodon:#osint@mastodon.social", ("mastodon", "#osint@mastodon.social")),
        ("mastodon:@user@infosec.exchange", ("mastodon", "@user@infosec.exchange")),
        ("mastodon:search:query@mastodon.social", ("mastodon", "search:query@mastodon.social")),
        ("bluesky:search:query", ("bluesky", "search:query")),
        ("bsky:@handle.bsky.social", ("bluesky", "@handle.bsky.social")),
        ("file:data.jsonl", ("files", "data.jsonl")),
        ("data.csv", ("files", "data.csv")),
    ],
)
def test_resolve_target(spec, expected):
    assert resolve_target(spec) == expected


def test_resolve_target_bare_existing_path(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    assert resolve_target(str(d)) == ("files", str(d))


@pytest.mark.parametrize("spec", ["", "   ", "nonsense-without-prefix", "reddit:"])
def test_resolve_target_invalid(spec):
    with pytest.raises(CollectorError):
        resolve_target(spec)


def test_registry_and_list():
    names = {c["name"] for c in list_collectors()}
    assert {"reddit", "hackernews", "rss", "web", "mastodon", "bluesky", "files"} <= names
    for c in list_collectors():
        assert c["description"] and c["examples"]
    assert get_collector("hn").name == "hackernews"
    with pytest.raises(CollectorError):
        get_collector("nope")


def test_helpers():
    assert strip_html("<p>Hello&nbsp;<b>world</b></p><script>x()</script><p>two</p>") == "Hello world\n\ntwo"
    dt = to_utc(1700000000)
    assert dt.tzinfo is not None and dt.year == 2023
    assert to_utc("2024-01-02T03:04:05Z") == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert to_utc("Tue, 02 Jan 2024 03:04:05 GMT") == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert to_utc("2024-01-02T03:04:05.123456789Z").microsecond == 123456
    assert to_utc("garbage") is None


# --------------------------------------------------------------------------- #
# files.py
# --------------------------------------------------------------------------- #
def test_files_jsonl_and_bad_lines(tmp_path):
    p = tmp_path / "posts.jsonl"
    lines = [
        json.dumps({"id": "a1", "body": "first post", "user": "alice", "link": "https://x/1", "created_utc": 1700000000, "score": 5}),
        "{not json",
        json.dumps({"content": "second", "timestamp": "2024-05-01T00:00:00Z"}),
        json.dumps({"irrelevant": 1}),  # no text -> skipped, not fatal
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    docs = list(get_collector("files").collect(str(p)))
    assert [d.text for d in docs] == ["first post", "second"]
    a = docs[0]
    assert a.id == "a1" and a.author == "alice" and a.url == "https://x/1"
    assert a.created_at == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert a.metrics["score"] == 5
    assert docs[1].created_at.tzinfo is not None
    assert docs[1].id  # generated stable id


def test_files_json_documents_roundtrip(tmp_path):
    original = Document(id="t3_x", platform="reddit", source="r/osint", text="hello", title="T",
                        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    p = tmp_path / "case.json"
    p.write_text(json.dumps({"documents": [original.to_dict()]}), encoding="utf-8")
    docs = files.load_documents(str(p))
    assert len(docs) == 1
    d = docs[0]
    assert (d.id, d.platform, d.source, d.title, d.text) == ("t3_x", "reddit", "r/osint", "T", "hello")
    assert d.created_at == original.created_at


def test_files_json_list(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([{"title": "Only title"}, {"text": "body", "tags": "a, b"}]), encoding="utf-8")
    docs = files.load_documents(str(p))
    assert docs[0].text == "Only title"
    assert docs[1].tags == ["a", "b"]


def test_files_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,text,author,date,likes\n1,hello world,bob,2024-02-03,\"1,200\"\n2,second row,carol,,3\n", encoding="utf-8")
    docs = list(get_collector("files").collect(str(p)))
    assert len(docs) == 2
    assert docs[0].author == "bob"
    assert docs[0].metrics["likes"] == 1200
    assert docs[0].created_at == datetime(2024, 2, 3, tzinfo=timezone.utc)
    assert docs[1].created_at is None


def test_files_text_and_directory(tmp_path):
    (tmp_path / "notes.md").write_text("# Heading\n\npara one\n\npara two\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "t.txt").write_text("just text", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")
    whole = list(get_collector("files").collect(str(tmp_path)))
    assert len(whole) == 2
    md = next(d for d in whole if d.source == "notes.md")
    assert md.title == "Heading" and "para two" in md.text
    split = list(get_collector("files").collect(str(tmp_path / "notes.md"), split=True))
    assert [d.text for d in split] == ["# Heading", "para one", "para two"]
    assert len({d.id for d in split}) == 3
    limited = list(get_collector("files").collect(str(tmp_path), limit=1))
    assert len(limited) == 1


def test_files_missing_path():
    with pytest.raises(CollectorError):
        list(get_collector("files").collect("/definitely/not/here.jsonl"))


# --------------------------------------------------------------------------- #
# RSS / Atom
# --------------------------------------------------------------------------- #
RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Example News</title>
    <link>https://news.example.com/</link>
    <item>
      <title>Port strike &amp; supply chains</title>
      <link>https://news.example.com/a</link>
      <guid>abc-1</guid>
      <pubDate>Tue, 02 Jan 2024 03:04:05 GMT</pubDate>
      <dc:creator>Jane Doe</dc:creator>
      <category>Logistics</category>
      <description><![CDATA[<p>Workers <b>walked out</b>.</p><img src="/img/a.jpg" alt="pickets"/>]]></description>
      <media:content url="https://cdn.example.com/b.png" medium="image"/>
    </item>
    <item>
      <title>Second</title>
      <link>https://news.example.com/b</link>
      <content:encoded><![CDATA[<div>Full <i>content</i> body</div>]]></content:encoded>
      <description>short</description>
    </item>
  </channel>
</rss>"""

ATOM_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Blog</title>
  <entry>
    <title>Hello Atom</title>
    <link rel="alternate" href="https://blog.example.org/hello"/>
    <id>urn:uuid:1234</id>
    <updated>2024-03-04T05:06:07Z</updated>
    <author><name>Sam</name></author>
    <category term="osint"/>
    <summary type="html">&lt;p&gt;Summary &lt;em&gt;text&lt;/em&gt;&lt;/p&gt;</summary>
  </entry>
</feed>"""


def test_parse_rss():
    docs = parse_feed(RSS_SAMPLE, feed_url="https://news.example.com/rss")
    assert len(docs) == 2
    a, b = docs
    assert a.platform == "rss" and a.source == "Example News"
    assert a.title == "Port strike & supply chains"
    assert a.text == "Workers walked out."
    assert a.author == "Jane Doe"
    assert a.tags == ["Logistics"]
    assert a.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    urls = {m.url for m in a.media}
    assert "https://cdn.example.com/b.png" in urls and "https://news.example.com/img/a.jpg" in urls
    assert b.text == "Full content body"  # content:encoded preferred over description
    assert a.id != b.id and a.id == parse_feed(RSS_SAMPLE, feed_url="https://news.example.com/rss")[0].id


def test_parse_atom():
    docs = parse_feed(ATOM_SAMPLE.encode("utf-8"), feed_url="https://blog.example.org/atom.xml")
    assert len(docs) == 1
    d = docs[0]
    assert d.title == "Hello Atom" and d.url == "https://blog.example.org/hello"
    assert d.text == "Summary text" and d.author == "Sam" and d.tags == ["osint"]
    assert d.created_at == datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


def test_rss_collector_with_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=RSS_SAMPLE, headers={"content-type": "application/rss+xml"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    col = get_collector("rss", {"min_interval": 0}, http=client)
    docs = list(col.collect("https://news.example.com/rss", limit=1))
    assert len(docs) == 1 and docs[0].title.startswith("Port strike")


# --------------------------------------------------------------------------- #
# Reddit mapping
# --------------------------------------------------------------------------- #
REDDIT_POST = {
    "kind": "t3",
    "data": {
        "id": "abc123",
        "name": "t3_abc123",
        "subreddit": "osint",
        "title": "Satellite imagery of &amp; port",
        "selftext": "Look at this",
        "author": "analyst1",
        "permalink": "/r/osint/comments/abc123/satellite/",
        "url": "https://i.redd.it/xyz.jpg",
        "created_utc": 1700000000.0,
        "score": 42,
        "num_comments": 7,
        "upvote_ratio": 0.97,
        "is_self": False,
        "link_flair_text": "Imagery",
    },
}


def test_reddit_map_post():
    d = reddit._map_post(REDDIT_POST)
    assert d.id == "t3_abc123" and d.platform == "reddit" and d.source == "r/osint"
    assert d.title == "Satellite imagery of & port"
    assert d.text == "Look at this" and d.author == "analyst1"
    assert d.url == "https://www.reddit.com/r/osint/comments/abc123/satellite/"
    assert d.created_at == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert d.metrics == {"score": 42, "num_comments": 7, "upvote_ratio": 0.97}
    assert [m.url for m in d.media] == ["https://i.redd.it/xyz.jpg"]
    assert "Imagery" in d.tags
    json.dumps(d.to_dict())  # serialisable


def test_reddit_map_gallery_and_comment():
    gallery = {
        "name": "t3_g1", "subreddit": "pics", "title": "g", "selftext": "", "permalink": "/r/pics/comments/g1/g/",
        "url": "https://www.reddit.com/gallery/g1", "created_utc": 1, "is_self": False,
        "gallery_data": {"items": [{"media_id": "m1", "caption": "first"}, {"media_id": "m2"}]},
        "media_metadata": {"m1": {"m": "image/jpg", "s": {"u": "https://preview.redd.it/m1.jpg?a=1&amp;b=2"}},
                           "m2": {"m": "image/png"}},
    }
    d = reddit._map_post(gallery)
    assert [m.url for m in d.media] == ["https://preview.redd.it/m1.jpg?a=1&b=2", "https://i.redd.it/m2.png"]
    assert d.media[0].description == "first"
    c = reddit._map_comment({"name": "t1_c1", "body": "nice", "author": "x", "subreddit": "pics",
                             "permalink": "/r/pics/comments/g1/g/c1/", "created_utc": 2, "score": 3})
    assert c.id == "t1_c1" and c.text == "nice" and "comment" in c.tags
    with pytest.raises(ValueError):
        reddit._map_post({"title": "no id"})


def test_reddit_json_pagination_with_mock_transport():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        after = request.url.params.get("after")
        if after is None:
            body = {"data": {"after": "t3_p2", "children": [
                {"kind": "t3", "data": {**REDDIT_POST["data"], "name": "t3_p1", "id": "p1"}},
                {"kind": "t3", "data": {"bad": "item"}},  # malformed -> skipped
            ]}}
        else:
            body = {"data": {"after": None, "children": [
                {"kind": "t3", "data": {**REDDIT_POST["data"], "name": "t3_p2", "id": "p2"}}]}}
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    col = get_collector("reddit", {"min_interval": 0, "force_json": True}, http=client)
    docs = list(col.collect("r/osint", limit=10, sort="new"))
    assert [d.id for d in docs] == ["t3_p1", "t3_p2"]
    assert "/r/osint/new.json" in str(calls[0])


def test_retry_on_429():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"hits": [], "nbPages": 0})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    col = get_collector("hn", {"min_interval": 0}, http=client)
    assert list(col.collect("search:x")) == []
    assert state["n"] == 2


# --------------------------------------------------------------------------- #
# Other mappers
# --------------------------------------------------------------------------- #
def test_hn_mappers():
    d = hackernews._map_hit({"objectID": "1", "title": "Show HN: thing", "url": "https://t.example", "author": "pg",
                             "points": 10, "num_comments": 2, "created_at_i": 1700000000, "_tags": ["story", "show_hn"]})
    assert d.id == "hn_1" and d.url == "https://news.ycombinator.com/item?id=1" and d.metrics["points"] == 10
    assert "show_hn" in d.tags
    c = hackernews._map_item({"id": 2, "type": "comment", "text": "<p>hi &amp; bye</p>", "author": "x",
                              "created_at": "2024-01-01T00:00:00.000Z"}, "Root")
    assert c.text == "hi & bye" and c.title == "Root"


def test_mastodon_mapper_and_targets():
    s = {"id": "109", "url": "https://mastodon.social/@a/109", "created_at": "2024-01-01T00:00:00.000Z",
         "content": "<p>Hello <a href='#'>#osint</a></p>", "account": {"acct": "a", "url": "https://mastodon.social/@a"},
         "media_attachments": [{"type": "image", "url": "https://files/x.png", "description": "alt text"}],
         "tags": [{"name": "osint"}], "replies_count": 1, "reblogs_count": 2, "favourites_count": 3}
    d = mastodon._map_status(s)
    assert d.text == "Hello #osint" and d.author == "a@mastodon.social"
    assert d.media[0].description == "alt text" and d.tags == ["#osint"]
    p = mastodon.MastodonCollector.parse_target
    assert p("#osint@infosec.exchange") == ("tag", "osint", "infosec.exchange")
    assert p("@user@instance.social") == ("account", "user", "instance.social")
    assert p("search:foo bar@mastodon.social") == ("search", "foo bar", "mastodon.social")
    assert p("public@infosec.exchange") == ("public", "local", "infosec.exchange")


def test_bluesky_mapper():
    item = {"post": {"uri": "at://did:plc:abcdefgh12345678/app.bsky.feed.post/3kxyz", "cid": "c",
                     "author": {"handle": "alice.bsky.social", "did": "did:plc:abcdefgh12345678"},
                     "record": {"text": "hello", "createdAt": "2024-01-01T00:00:00.000Z"},
                     "embed": {"$type": "app.bsky.embed.images#view",
                               "images": [{"fullsize": "https://cdn.bsky.app/img/1.jpg", "alt": "a cat"}]},
                     "likeCount": 4}}
    d = bluesky._map_post(item)
    assert d.url == "https://bsky.app/profile/alice.bsky.social/post/3kxyz"
    assert d.media[0].url.endswith("1.jpg") and d.media[0].description == "a cat"
    assert d.metrics["likes"] == 4 and d.created_at.tzinfo is not None


def test_web_extract():
    page = """<html lang="en"><head><title>Report</title>
    <meta property="og:image" content="/og.png"><meta name="description" content="Desc here">
    <style>.x{}</style><script>var a=1;</script></head>
    <body><nav>Menu Home About</nav><main><h1>Findings</h1><p>Body text <img src="pic.jpg" alt="p"></p>
    <a href="/next">next</a></main><footer>copyright</footer></body></html>"""
    doc = web.page_to_document(page, "https://site.example/a/index.html")
    assert doc.title == "Report"
    assert "Findings" in doc.text and "Body text" in doc.text
    assert "Menu" not in doc.text and "copyright" not in doc.text and "var a" not in doc.text
    assert [m.url for m in doc.media] == ["https://site.example/og.png", "https://site.example/a/pic.jpg"]
    assert "https://site.example/next" in web.extract(page, "https://site.example/a/")["links"]


def test_web_collector_robots_and_crawl():
    pages = {
        "/robots.txt": "User-agent: *\nDisallow: /private\n",
        "/": "<html><title>Home</title><body><p>home</p><a href='/a'>a</a><a href='/private/x'>p</a>"
             "<a href='https://other.example/'>o</a></body></html>",
        "/a": "<html><title>A</title><body><p>page a</p></body></html>",
    }

    def handler(request):
        path = request.url.path
        if path in pages:
            ctype = "text/plain" if path.endswith(".txt") else "text/html"
            return httpx.Response(200, text=pages[path], headers={"content-type": ctype})
        return httpx.Response(404, text="nf")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    col = get_collector("web", {"min_interval": 0}, http=client)
    col.resolver = lambda host, port: ["93.184.216.34"]  # site.example "resolves" to a public address
    docs = list(col.collect("https://site.example/", limit=10, crawl_depth=1))
    assert [d.title for d in docs] == ["Home", "A"]
