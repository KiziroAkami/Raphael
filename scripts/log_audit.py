"""Ad-hoc log auditor. Invoked by /log-audit. Reads conversations.log,
applies retrieval-quality checks, prints a structured report, advances cursor."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.retriever import _normalize_query, _bare_entity_name  # noqa: E402

LOG_PATH = ROOT / "data" / "conversations.log"
CURSOR_PATH = ROOT / "data" / ".log-audit-cursor"

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"user=(?P<user>\S+) channel=(?P<channel>\S+) "
    r"q='(?P<q>(?:[^'\\]|\\.)*)' "
    r"(?:cleaned='(?P<cleaned>(?:[^'\\]|\\.)*)' )?"
    r"chunks=(?P<chunks>-?\d+) "
    r"(?:pages=(?P<pages>\[[^\]]*\]) )?"
    r"(?:scores=(?P<scores>\[[^\]]*\]) )?"
    r"model=(?P<model>\S+) "
    r"latency_ms=(?P<lat>\d+) "
    r"(?:response='(?P<resp>(?:[^'\\]|\\.)*)')?"
    r"(?: error=(?P<err>.+))?$"
)


def _parse_pylist_strs(s: str) -> list[str]:
    """Parse a python-repr list of strings like ['a', \"b's\", 'c']."""
    if not s:
        return []
    # Convert single-quoted repr into JSON by walking token by token.
    out: list[str] = []
    inner = s.strip()[1:-1].strip()
    if not inner:
        return out
    i = 0
    while i < len(inner):
        if inner[i] != "'":
            i += 1
            continue
        j = i + 1
        buf: list[str] = []
        while j < len(inner):
            if inner[j] == "\\" and j + 1 < len(inner):
                buf.append(inner[j + 1])
                j += 2
                continue
            if inner[j] == "'":
                break
            buf.append(inner[j])
            j += 1
        out.append("".join(buf))
        i = j + 1
    return out


def _parse_pylist_nums(s: str) -> list[float]:
    if not s:
        return []
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return []


def parse_line(line: str) -> dict | None:
    m = LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    d = m.groupdict()
    d["ts_dt"] = datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S")
    d["chunks"] = int(d["chunks"])
    d["latency_ms"] = int(d["lat"])
    d["pages_list"] = _parse_pylist_strs(d.get("pages") or "")
    d["scores_list"] = _parse_pylist_nums(d.get("scores") or "")
    d["q"] = d["q"].replace("\\'", "'")
    if d.get("cleaned"):
        d["cleaned"] = d["cleaned"].replace("\\'", "'")
    else:
        d["cleaned"] = d["q"]
    return d


def read_cursor() -> datetime:
    now = datetime.now()
    if CURSOR_PATH.exists():
        try:
            ts = datetime.fromisoformat(CURSOR_PATH.read_text().strip())
            if now - ts < timedelta(hours=48):
                return ts
        except ValueError:
            pass
    return now - timedelta(hours=48)


def write_cursor(ts: datetime) -> None:
    CURSOR_PATH.write_text(ts.isoformat(timespec="seconds"))


_NOISE = {"hello", "hi", "hey", "test", "yo", "sup", "ok", "okay"}
_NOISE_PAT = re.compile(
    r"^(?:hello|hi|hey|test|yo|sup|anyone|you there|anybody|who are you)[\s!?.]*$",
    re.I,
)


def is_noise(q: str) -> bool:
    stripped = q.strip().rstrip("?!. ").lower()
    if not stripped or stripped in _NOISE:
        return True
    return bool(_NOISE_PAT.match(q.strip()))


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "where", "when", "why", "how", "does", "do", "did",
    "can", "could", "will", "would", "should", "have", "has", "had",
    "of", "for", "in", "on", "at", "to", "from", "with", "without",
    "and", "or", "but", "vs", "versus",
    "tell", "me", "about", "explain", "describe", "list", "show",
    "this", "that", "these", "those", "it", "its",
    "as", "like", "than", "then", "there", "their", "they",
    "all", "any", "some", "many", "much", "more", "most", "less",
    "you", "your", "i", "my", "we", "our", "us",
    "mod", "tensura", "minecraft", "game", "best", "get", "got",
    "use", "using", "make", "made", "work", "works",
}


def key_entities(q: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']+", q.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) >= 3]


def main() -> None:
    cursor = read_cursor()
    entries: list[dict] = []
    malformed = 0
    with LOG_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = parse_line(line)
            if d is None:
                malformed += 1
                continue
            if d["ts_dt"] >= cursor:
                entries.append(d)

    if not entries:
        print(f"No entries after cursor {cursor.isoformat()}.")
        return

    first_ts = entries[0]["ts_dt"]
    last_ts = entries[-1]["ts_dt"]

    # --- Check 1: Normalization mismatch ---
    norm_changes: list[dict] = []
    for e in entries:
        normalized = _normalize_query(e["cleaned"])
        if normalized != e["cleaned"]:
            e["normalized"] = normalized
            ents = key_entities(normalized)
            if e["pages_list"]:
                page_blob = " ".join(e["pages_list"]).lower()
                matched = any(w in page_blob for w in ents)
            else:
                matched = False
            norm_changes.append({**e, "matched_intent": matched})

    # --- Check 2: Wrong page retrieval ---
    wrong_retrieval: list[dict] = []
    for e in entries:
        if e["chunks"] <= 0 or not e["pages_list"]:
            continue
        if is_noise(e["q"]):
            continue
        ents = key_entities(e["cleaned"])
        if not ents:
            continue
        page_blob = " ".join(e["pages_list"]).lower()
        hit = any(w in page_blob for w in ents)
        if not hit:
            wrong_retrieval.append(e)

    # --- Check 3: Low confidence (top score < 0.75) ---
    low_conf: list[dict] = []
    for e in entries:
        if e["chunks"] <= 0 or not e["scores_list"]:
            continue
        if e["scores_list"][0] < 0.75:
            low_conf.append(e)

    # --- Check 4: Repeated failures ---
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in wrong_retrieval:
        norm = e["cleaned"].lower().rstrip("?!.,;: ").strip()
        groups[norm].append(e)
    repeated = {k: v for k, v in groups.items()
                if len({x["user"] for x in v}) >= 2}

    # --- Check 5: Noise ---
    noise = [e for e in entries if is_noise(e["q"])]

    # --- Check 6: Errors ---
    errors: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e["model"] == "error" or e["chunks"] == -1:
            err_type = "unknown"
            if e.get("err"):
                err_type = e["err"].split(":", 1)[0].strip()
            errors[err_type].append(e)

    # --- Check 7: High latency (>10s) ---
    high_lat = [e for e in entries if e["latency_ms"] > 10000]

    # --- Check 8: Page monoculture ---
    # Skip bare-entity queries: `_query_by_page_title()` deliberately returns
    # every chunk of the target page, so monoculture is the correct outcome.
    monoculture: list[dict] = []
    for e in entries:
        if e["chunks"] > 3 and e["pages_list"]:
            if len(set(e["pages_list"])) == 1:
                if _bare_entity_name(e["cleaned"]) is not None:
                    continue
                monoculture.append(e)

    # --- Check 9: Score cliff ---
    score_cliff: list[dict] = []
    for e in entries:
        if e["chunks"] > 5 and len(e["scores_list"]) > 5:
            if (e["scores_list"][0] - e["scores_list"][-1]) > 0.15:
                score_cliff.append(e)

    # ---------- Print report ----------
    P = print
    P("=" * 78)
    P("LOG AUDIT REPORT")
    P(f"Scan range: {first_ts.isoformat()} → {last_ts.isoformat()}")
    P(f"Entries scanned: {len(entries)} (malformed skipped: {malformed})")
    P(f"Cursor (prev): {cursor.isoformat()}")
    P("=" * 78)

    P("\n── SUMMARY ──────────────────────────────────────────────────────────")
    P(f"  1. Normalization rewrites ..... {len(norm_changes)}")
    P(f"  2. Wrong-page retrieval ....... {len(wrong_retrieval)}")
    P(f"  3. Low confidence (<0.75) ..... {len(low_conf)}")
    P(f"  4. Repeated failures (groups).. {len(repeated)}")
    P(f"  5. Non-question noise ......... {len(noise)}")
    P(f"  6. Errors ..................... {sum(len(v) for v in errors.values())}")
    P(f"  7. High latency (>10s) ........ {len(high_lat)}")
    P(f"  8. Page monoculture ........... {len(monoculture)}")
    P(f"  9. Score cliff ................ {len(score_cliff)}")

    def show_entry(e: dict, extra: str = "") -> None:
        top = e["scores_list"][0] if e["scores_list"] else "n/a"
        pages = e["pages_list"][:3] if e["pages_list"] else []
        P(f"  [{e['ts']}] q={e['q']!r} top={top} pages={pages}{extra}")

    if norm_changes:
        P("\n── 1. NORMALIZATION REWRITES ───────────────────────────────────────")
        for e in norm_changes[:15]:
            tag = "  OK" if e["matched_intent"] else "  ⚠ intent MISS"
            P(f"  [{e['ts']}] {e['cleaned']!r} → {e['normalized']!r}{tag}")
            P(f"         pages={e['pages_list'][:4]}")

    if wrong_retrieval:
        P("\n── 2. WRONG-PAGE RETRIEVAL ──────────────────────────────────────────")
        for e in wrong_retrieval[:15]:
            show_entry(e)

    if low_conf:
        P("\n── 3. LOW CONFIDENCE (top score < 0.75) ─────────────────────────────")
        low_conf_sorted = sorted(low_conf, key=lambda x: x["scores_list"][0])
        for e in low_conf_sorted[:15]:
            show_entry(e)

    if repeated:
        P("\n── 4. REPEATED FAILURES (≥2 distinct users, wrong retrieval) ────────")
        for norm, items in sorted(repeated.items(), key=lambda kv: -len(kv[1])):
            users = len({x["user"] for x in items})
            P(f"  × {len(items)} hits / {users} users — {norm!r}")
            for e in items[:2]:
                show_entry(e, extra=f" user={e['user']}")

    if noise:
        P(f"\n── 5. NOISE (greetings/tests) — {len(noise)} total ──────────────")
        counts: dict[str, int] = defaultdict(int)
        for e in noise:
            counts[e["q"].strip().lower()] += 1
        for q, n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            P(f"  {n:3d}× {q!r}")

    if errors:
        P("\n── 6. ERRORS ───────────────────────────────────────────────────────")
        for err_type, items in errors.items():
            P(f"  {err_type} × {len(items)}")
            for e in items[:2]:
                P(f"    [{e['ts']}] q={e['q']!r} err={e.get('err')!r}")

    if high_lat:
        P("\n── 7. HIGH LATENCY (>10s) ──────────────────────────────────────────")
        for e in sorted(high_lat, key=lambda x: -x["latency_ms"])[:10]:
            P(f"  [{e['ts']}] {e['latency_ms']}ms model={e['model']} q={e['q']!r}")

    if monoculture:
        P("\n── 8. PAGE MONOCULTURE (all chunks one page, chunks>3) ──────────────")
        for e in monoculture[:10]:
            P(f"  [{e['ts']}] q={e['q']!r} page={e['pages_list'][0]!r} chunks={e['chunks']}")

    if score_cliff:
        P("\n── 9. SCORE CLIFF (top-bottom > 0.15, chunks>5) ─────────────────────")
        for e in score_cliff[:10]:
            delta = e["scores_list"][0] - e["scores_list"][-1]
            P(f"  [{e['ts']}] Δ={delta:.3f} top={e['scores_list'][0]} bot={e['scores_list'][-1]} q={e['q']!r}")

    P("\n" + "=" * 78)

    write_cursor(last_ts)
    P(f"Cursor advanced to {last_ts.isoformat()}.")


if __name__ == "__main__":
    main()
