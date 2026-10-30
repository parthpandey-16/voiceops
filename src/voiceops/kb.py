"""Load the ZentroMart knowledge base and catalog into retrievable chunks.

Every `## [SECTION-ID] Title` heading becomes one chunk. The chunk keeps the
section ID plus its document's doc_type, status and effective_from, which the
agent needs to apply the precedence rules in KB-00.1.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import settings

# Precedence from KB-00.1 (lower = stronger authority).
PRECEDENCE = {"policy": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5, "meta": 1}

_SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$")


@dataclass
class Chunk:
    section_id: str
    title: str
    text: str
    doc_id: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    file: str
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        tag = f"{self.doc_type}, {self.status}, effective {self.effective_from}"
        return f"[{self.section_id}] {self.title} ({self.doc_title}; {tag})"

    def render(self) -> str:
        return f"{self.label}\n{self.text.strip()}"

    @property
    def search_text(self) -> str:
        return f"{self.doc_title}. {self.title}. {self.text}"


def _parse_header(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body


def parse_document(path: Path) -> list[Chunk]:
    meta, body = _parse_header(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    current, lines = None, []
    preamble: list[str] = []

    def flush():
        if current:
            sid, title = current
            chunks.append(Chunk(
                section_id=sid, title=title, text="\n".join(lines).strip(),
                doc_id=meta.get("doc_id", path.stem), doc_title=meta.get("title", path.stem),
                doc_type=meta.get("doc_type", "help"), status=meta.get("status", "current"),
                effective_from=meta.get("effective_from", ""), file=path.name, meta=meta))

    for line in body.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            flush()
            current, lines = (match.group(1).strip(), match.group(2).strip()), []
        elif current:
            lines.append(line)
        elif not line.startswith("# "):
            preamble.append(line)
    flush()
    # A superseded doc's scope note ("still applies only to ...") sits above its
    # first section; keep it on every chunk so the scope is never lost.
    note = " ".join(l.lstrip("> ").strip() for l in preamble if l.strip())
    if note:
        for chunk in chunks:
            chunk.text = f"Note: {note}\n{chunk.text}"
    return chunks


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Chunk, ...]:
    kb_dir = settings.dataset_dir / "kb"
    out: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        out.extend(parse_document(path))
    return tuple(out)


def chunk_by_id() -> dict[str, Chunk]:
    return {c.section_id: c for c in load_chunks()}


def _read_csv(name: str) -> list[dict]:
    with open(settings.dataset_dir / "data" / name, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=1)
def load_products() -> tuple[dict, ...]:
    return tuple(_read_csv("products.csv"))


@lru_cache(maxsize=1)
def load_pincodes() -> dict[str, dict]:
    return {row["pincode"]: row for row in _read_csv("pincodes.csv")}


@lru_cache(maxsize=1)
def load_holidays() -> dict[str, str]:
    return {row["date"]: row["name"] for row in _read_csv("holidays.csv")}


def render_product(row: dict) -> str:
    seller = row["fulfilled_by"]
    marketplace = seller != "ZentroMart"
    return (f"[products.csv:{row['sku']}] {row['title']} | brand {row['brand']} | "
            f"return_category={row['return_category']} | price ₹{row['price_inr']} | "
            f"fulfilled_by={seller}{' (marketplace seller, see MKS-3)' if marketplace else ''} | "
            f"assured={row['assured']} | brand warranty_months={row['warranty_months']}")
