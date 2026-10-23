"""Hybrid retrieval: BM25 keyword index + optional vectors, fused and reranked."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "is", "are", "for", "with", "at", "by", "it", "this", "that", "as", "be", "or"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOP]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: dict[str, list[str]] = {}
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.docs:
            self.remove(doc_id)
        toks = tokenize(text)
        self.docs[doc_id] = toks
        for t in set(toks):
            self.df[t] += 1
        self._recompute()

    def remove(self, doc_id: str) -> None:
        toks = self.docs.pop(doc_id, None)
        if toks:
            for t in set(toks):
                self.df[t] -= 1
                if self.df[t] <= 0:
                    del self.df[t]
        self._recompute()

    def _recompute(self) -> None:
        self.avgdl = (sum(len(t) for t in self.docs.values()) / len(self.docs)) if self.docs else 0.0

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q or not self.docs:
            return []
        n = len(self.docs)
        scores: dict[str, float] = defaultdict(float)
        for term in set(q):
            df = self.df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, toks in self.docs.items():
                tf = toks.count(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * len(toks) / (self.avgdl or 1))
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked ID lists."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def rerank(query: str, candidates: list[tuple[str, str]], k: int = 5) -> list[tuple[str, float]]:
    """Cheap reranker: query-term coverage plus phrase bonus. Replace with a cross-encoder sidecar when available."""
    q = set(tokenize(query))
    out = []
    for doc_id, text in candidates:
        toks = set(tokenize(text))
        coverage = len(q & toks) / len(q) if q else 0.0
        phrase = 0.2 if query.lower().strip() and query.lower().strip() in (text or "").lower() else 0.0
        out.append((doc_id, coverage + phrase))
    return sorted(out, key=lambda x: -x[1])[:k]


def hybrid_search(query: str, bm25: BM25Index, texts: dict[str, str], k: int = 5, embed: Optional[Callable[[list[str]], list[list[float]]]] = None,
                  vectors: Optional[dict[str, list[float]]] = None) -> list[tuple[str, float]]:
    keyword = [d for d, _ in bm25.search(query, k * 4)]
    rankings = [keyword]
    if embed is not None and vectors:
        try:
            qv = embed([query])[0]
            vec_rank = sorted(vectors.items(), key=lambda kv: -cosine(qv, kv[1]))[: k * 4]
            rankings.append([d for d, _ in vec_rank])
        except Exception:
            pass
    fused = rrf(rankings)
    cands = [(d, texts.get(d, "")) for d, _ in fused[: k * 3]]
    return rerank(query, cands, k)
