"""Hybrid retrieval over KB sections: BM25 + multilingual dense vectors, fused with RRF.

* BM25 catches exact terms (SKU codes, "reverse pickup", "COD").
* Dense vectors catch paraphrases and Hindi/Hinglish questions against English docs.
* Optional document expansion: an LLM writes a few customer-style questions for
  each section once (cached), so "nobody came to collect it" can meet RVP-2.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .kb import PRECEDENCE, Chunk, load_chunks

log = logging.getLogger(__name__)


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[\wऀ-ॿ]+", text.lower()) if len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int

    def as_dict(self) -> dict:
        return {"section_id": self.chunk.section_id, "doc": self.chunk.file,
                "doc_type": self.chunk.doc_type, "status": self.chunk.status,
                "score": round(self.score, 4), "rank": self.rank,
                "text": self.chunk.text.strip()}


class Embedder:
    """Dense encoder. Falls back to None (BM25 only) if the backend is unavailable."""

    def __init__(self, backend: str, llm=None):
        self.backend, self.llm, self.model = backend, llm, None
        if backend == "fastembed":
            try:
                from fastembed import TextEmbedding

                self.model = TextEmbedding(settings.fastembed_model,
                                           cache_dir=str(settings.cache_dir / "fastembed"))
            except Exception as exc:  # noqa: BLE001 - offline or not installed
                log.warning("fastembed unavailable (%s); using BM25 only", exc)
                self.backend = "none"
        elif backend == "api" and not (llm and llm.can_embed):
            log.warning("Embedding API not available for this provider; using BM25 only")
            self.backend = "none"

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "fastembed":
            vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        else:
            vecs = np.array(self.llm.embed(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


class Retriever:
    def __init__(self, llm=None, embeddings: str | None = None, expand: bool = True):
        self.chunks = list(load_chunks())
        self.llm = llm
        self.expansions = self._load_expansions(generate=expand and llm is not None) if expand else {}
        self.docs = [self._index_text(c) for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])
        self.embedder = Embedder(embeddings or settings.embeddings, llm)
        self.matrix = self._load_vectors() if self.embedder.enabled else None

    # -- indexing ---------------------------------------------------------
    def _index_text(self, chunk: Chunk) -> str:
        extra = " ".join(self.expansions.get(chunk.section_id, []))
        return f"{chunk.search_text}\n{extra}".strip()

    def _fingerprint(self) -> str:
        blob = "".join(c.section_id + c.text for c in self.chunks)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def _load_expansions(self, generate: bool) -> dict[str, list[str]]:
        path = settings.cache_dir / f"expansions_{self._fingerprint()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if not generate:
            return {}
        from .prompts import EXPANSION_PROMPT

        out: dict[str, list[str]] = {}
        for chunk in self.chunks:
            try:
                raw = self.llm.complete(EXPANSION_PROMPT.format(section=chunk.render()))
                questions = [q.strip("-•* \t") for q in raw.splitlines() if q.strip()]
                out[chunk.section_id] = questions[:10]
            except Exception as exc:  # noqa: BLE001
                log.warning("document expansion stopped at %s (%s); continuing without it",
                            chunk.section_id, exc)
                return {}
        if out:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out

    def _load_vectors(self) -> np.ndarray | None:
        name = f"vectors_{self.embedder.backend}_{self._fingerprint()}_{len(self.expansions)}.npy"
        path = settings.cache_dir / name
        if path.exists():
            return np.load(path)
        try:
            matrix = self.embedder.encode([self._dense_text(c) for c in self.chunks])
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding failed (%s); using BM25 only", exc)
            self.embedder.backend = "none"
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return matrix

    def _dense_text(self, chunk: Chunk) -> str:
        extra = " ".join(self.expansions.get(chunk.section_id, [])[:5])
        return f"{chunk.doc_title} {chunk.section_id} {chunk.title}: {chunk.text} {extra}"[:2000]

    # -- search -----------------------------------------------------------
    def search(self, queries: str | list[str], k: int | None = None) -> list[Hit]:
        """Fuse BM25 and dense rankings for one or more query strings (RRF, k=60)."""
        k = k or settings.top_k
        queries = [q for q in ([queries] if isinstance(queries, str) else queries) if q and q.strip()]
        fused = np.zeros(len(self.chunks))
        for query in queries:
            rankings = [np.argsort(-self.bm25.get_scores(tokenize(query)))]
            if self.matrix is not None:
                try:
                    qv = self.embedder.encode([query])[0]
                    rankings.append(np.argsort(-(self.matrix @ qv)))
                except Exception as exc:  # noqa: BLE001
                    log.warning("query embedding failed: %s", exc)
            for ranking in rankings:
                for rank, idx in enumerate(ranking[:30]):
                    fused[idx] += 1.0 / (60 + rank)
        # Light authority prior from KB-00.1 so a policy edges out an FAQ that
        # scored the same; superseded text stays retrievable for older orders.
        for i, chunk in enumerate(self.chunks):
            if fused[i]:
                prior = 1 - 0.02 * (PRECEDENCE.get(chunk.doc_type, 3) - 1)
                if chunk.status == "superseded":
                    prior -= 0.04
                fused[i] *= prior
        order = [i for i in np.argsort(-fused) if fused[i] > 0][:k]
        # Follow explicit cross-references ("REF-1 applies", "see DWM-1") from the
        # top hits, so a policy arrives with the section it defers to.
        index = {c.section_id: i for i, c in enumerate(self.chunks)}
        for i in list(order[:3]):
            for ref in re.findall(r"\b([A-Z]{2,4}-\d+(?:\.\d+)?)\b", self.chunks[i].text):
                j = index.get(ref)
                if j is not None and j not in order and len(order) < k + 2:
                    order.append(j)
        return [Hit(self.chunks[i], float(fused[i]), r + 1) for r, i in enumerate(order)]

