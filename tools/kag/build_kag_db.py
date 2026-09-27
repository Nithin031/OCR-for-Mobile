#!/usr/bin/env python3
"""Build the offline KAG database (SQLite) for the Android app from the BuildForBillions26 web repo.

Mirrors the web ingestion so the phone searches the same knowledge:
  scheme/<slug>/{meta.json,content.txt}  -> documents  (backend/app/ingestion/scheme_folder.py)
  extract(".txt") + clean_text            -> one block  (backend/app/ingestion/extract.py)
  chunk_blocks (900 chars, 150 overlap)   -> chunks     (backend/app/ingestion/chunker.py)
  detect_scheme_mentions                  -> scheme codes per document / chunk (backend/app/ingestion/pipeline.py)
  data/seed/graph.json                    -> the scheme graph, stored verbatim (backend/app/graph/store.py)

Keyword lexemes approximate PostgreSQL to_tsvector('simple', section || ' ' || content):
lower-cased words, hyphenated compounds also kept whole and split into parts.

Usage:  python tools/kag/build_kag_db.py --repo <path to BuildForBillions26> [--out app/src/main/assets/kag/kag.db]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

TARGET_CHARS = 900
OVERLAP_CHARS = 150
MIN_CHARS = 200
SCHEMA_VERSION = 1


# ── backend/app/ingestion/extract.py ─────────────────────────────────────────
def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    return {}, text[end + 4:].lstrip("\n")


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_of(raw: bytes) -> str:
    _, body = parse_front_matter(raw.decode("utf-8", errors="replace"))
    return clean_text(body)


# ── backend/app/ingestion/chunker.py ─────────────────────────────────────────
def _split_long(text: str, size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur.strip())
            cur = ""
        while len(s) > size:
            out.append(s[:size])
            s = s[size:]
        cur += " " + s
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_blocks(blocks: list[dict], target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[dict]:
    chunks: list[dict] = []
    for block in blocks:
        paras = [p.strip() for p in re.split(r"\n\s*\n", block["text"]) if p.strip()]
        pieces: list[str] = []
        for p in paras:
            pieces += _split_long(p, target) if len(p) > target else [p]
        cur = ""
        for piece in pieces:
            if cur and len(cur) + len(piece) + 2 > target:
                chunks.append({"text": cur, "section": block.get("section")})
                tail = cur[-overlap:]
                cur = (tail[tail.find(" ") + 1:] if " " in tail else "") + "\n\n" + piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
        if cur.strip():
            chunks.append({"text": cur.strip(), "section": block.get("section")})
    return chunks


# ── backend/app/ingestion/pipeline.py detect_scheme_mentions ─────────────────
EXTRA_ALIASES = {
    "PM_KISAN": {"pm-kisan", "pm kisan"},
    "PMFBY": {"pmfby", "fasal bima", "crop insurance"},
    "KCC_CALAMITY_RELIEF": {"kisan credit card", "kcc", "crop loan"},
    "CROP_LOSS_RELIEF_KA": {"input subsidy", "sdrf", "ndrf", "crop loss relief"},
}


def scheme_aliases(graph: dict) -> dict[str, set[str]]:
    out = {}
    for s in graph["schemes"]:
        aliases = {s["name"], s.get("short_name") or s["name"], s["code"].replace("_", " ")}
        m = re.search(r"\(([^)]+)\)", s["name"])
        if m:
            aliases.add(m.group(1))
        aliases |= EXTRA_ALIASES.get(s["code"], set())
        out[s["code"]] = aliases
    return out


def detect_scheme_mentions(text: str, aliases: dict[str, set[str]]) -> list[str]:
    lower = text.lower()
    return [code for code, al in aliases.items()
            if any(a and len(a) > 2 and re.search(r"\b" + re.escape(a.lower()) + r"\b", lower) for a in al)]


# ── backend/app/ingestion/scheme_folder.py ───────────────────────────────────
def doc_id_for(slug: str) -> str:
    return "sf-" + hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]


def clean_title(title: str | None, slug: str) -> str:
    t = re.sub(r"\s*::\s*", " – ", title or "").strip(" -–|:")
    return (t or slug.replace("-", " ").title())[:500]


def publisher_of(meta: dict) -> tuple[str, str | None]:
    url = meta.get("final_url") or meta.get("source_url") or ""
    if url.startswith(("http://", "https://")):
        return urlparse(url).netloc.lower().removeprefix("www."), url
    return "Scheme library", None


def lexemes(text: str) -> list[str]:
    lower = text.lower()
    words = set(re.findall(r"\w+", lower))
    words |= {w.strip("-") for w in re.findall(r"\w+(?:-\w+)+", lower)}
    return sorted(w for w in words if w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "app/src/main/assets/kag/kag.db")
    args = ap.parse_args()
    repo: Path = args.repo.resolve()

    graph_text = (repo / "data/seed/graph.json").read_text(encoding="utf-8")
    graph = json.loads(graph_text)
    aliases = scheme_aliases(graph)

    docs, chunks, skipped = [], [], []
    for d in sorted(p for p in (repo / "scheme").iterdir() if p.is_dir()):
        meta_path, text_path = d / "meta.json", d / "content.txt"
        if not (meta_path.is_file() and text_path.is_file()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status", "ok") != "ok":
            skipped.append((d.name, "status"))
            continue
        full = text_of(text_path.read_bytes())
        if len(full.strip()) < MIN_CHARS:
            skipped.append((d.name, "short"))
            continue
        slug = meta.get("slug") or d.name
        doc_id = doc_id_for(slug)
        publisher, url = publisher_of(meta)
        codes = sorted(set(detect_scheme_mentions(full, aliases)))
        docs.append((doc_id, slug, clean_title(meta.get("title"), slug), publisher, url, json.dumps(codes)))
        for i, c in enumerate(chunk_blocks([{"text": full, "section": None}])):
            c_codes = sorted(set(codes if len(codes) == 1 else []) | set(detect_scheme_mentions(c["text"], aliases)))
            chunks.append((f"{doc_id}:{i}", doc_id, i, c["section"], c["text"], json.dumps(c_codes or codes),
                           " ".join(lexemes(((c["section"] or "") + " " + c["text"])))))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    con = sqlite3.connect(args.out)
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE graph (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL);
        CREATE TABLE documents (id TEXT PRIMARY KEY, slug TEXT NOT NULL, title TEXT NOT NULL, publisher TEXT,
                                url TEXT, scheme_codes TEXT NOT NULL);
        CREATE TABLE chunks (id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                             chunk_index INTEGER NOT NULL, section TEXT, content TEXT NOT NULL,
                             scheme_codes TEXT NOT NULL, lexemes TEXT NOT NULL);
        CREATE INDEX chunks_doc ON chunks(document_id, chunk_index);
    """)
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    con.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("schema_version", str(SCHEMA_VERSION)), ("source_repo", "BuildForBillions26"), ("source_commit", commit),
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("documents", str(len(docs))), ("chunks", str(len(chunks))),
    ])
    con.execute("INSERT INTO graph VALUES (1, ?)", (graph_text,))
    con.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)", docs)
    con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", chunks)
    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"kag.db: {len(docs)} documents, {len(chunks)} chunks, {len(graph['schemes'])} graph schemes, "
          f"skipped {len(skipped)}, source {commit}, {args.out.stat().st_size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
