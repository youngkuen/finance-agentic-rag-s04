# -*- coding: utf-8 -*-
"""Hybrid 검색: Dense + Kiwi BM25 를 RRF 로 합친다.

RRF(Reciprocal Rank Fusion)를 쓰는 이유는 점수 스케일이 다르기 때문이다.
코사인 유사도는 0~1, BM25 점수는 상한이 없다. 정규화해서 더하면 문서 수와
질의 길이에 따라 가중치가 멋대로 흔들린다. RRF 는 점수 대신 순위만 쓴다.

    score(d) = Σ 1 / (k + rank_i(d)),  k = 60
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from qdrant_client import models

from ..index import bm25_ko
from ..index.bm25_ko import BM25Index, tokenize
from ..settings import get_settings
from . import dense

RRF_K = 60


@lru_cache
def _chunks() -> list[dict]:
    path = get_settings().chunks_dir / "chunks.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _bm25_cache_path() -> Path:
    """캐시 파일 이름 = 청크 파일과 토크나이저 코드의 해시. 둘 중 하나라도 바뀌면 다른 파일이 된다."""
    s = get_settings()
    h = hashlib.sha256((s.chunks_dir / "chunks.jsonl").read_bytes()
                       + Path(bm25_ko.__file__).read_bytes()).hexdigest()[:16]
    return s.data / "bm25_cache" / f"{h}.json"


@lru_cache
def _bm25() -> BM25Index:
    """Kiwi 로 청크 3,700개를 토큰화하면 수십 초에서 몇 분이 걸린다. 프로세스를 새로 띄울 때마다
    그 시간을 다시 쓰지 않도록 토큰 목록을 파일에 캐시한다. 청크나 tokenize 가 바뀌면 캐시 이름이
    달라져서 자연히 다시 만든다.
    """
    chunks = _chunks()
    ids = [c["chunk_id"] for c in chunks]
    cache = _bm25_cache_path()
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("chunk_ids") == ids:
            return BM25Index(ids, data["tokens"])
    toks = [tokenize(c["text"]) or ["_"] for c in chunks]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"chunk_ids": ids, "tokens": toks}, ensure_ascii=False), encoding="utf-8")
    return BM25Index(ids, toks)


@lru_cache
def _by_id() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in _chunks()}


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """순위 목록 여러 개를 하나의 점수표로 합친다. {chunk_id: 점수}. 점수가 클수록 앞이다.

    rankings 의 각 원소는 chunk_id 를 순위 순서로 늘어놓은 목록이다(첫 번째가 1등).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for r, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + r)
    return fused


def _hydrate(chunk_id: str, score: float) -> dict:
    c = _by_id().get(chunk_id)
    if not c:
        return {"chunk_id": chunk_id, "score": score, "text": ""}
    m = c.get("meta", {})
    return {"chunk_id": chunk_id, "score": score, "text": c["text"], "doc_id": c["doc_id"],
            "article": c.get("article", ""), "article_title": c.get("article_title", ""),
            "page_start": c.get("page_start", 0), "page_end": c.get("page_end", 0),
            "doc_type": m.get("doc_type", ""), "issuer": m.get("issuer", ""),
            "product": m.get("product", ""), "generation": m.get("generation", ""),
            "effective_from": m.get("effective_from", ""), "review_expiry": m.get("review_expiry", ""),
            "expired": bool(m.get("expired")), "synthetic_degraded": bool(m.get("synthetic_degraded")),
            "license": m.get("license", "")}


def search(query: str, k: int = 10, *, candidates: int = 50,
           flt: models.Filter | None = None, collection: str | None = None,
           use_bm25: bool = True) -> list[dict]:
    """Dense 와 BM25 를 RRF 로 합쳐 상위 k 개를 돌려준다. 원소는 _hydrate() 가 만드는 dict 다.

    candidates: 두 검색에서 각각 몇 개를 후보로 받을지. flt: Qdrant 필터(발행사·문서 종류 등).
    use_bm25=False 면 Dense 순위만으로 같은 모양을 돌려준다(비교 실험용).
    """
    dense_hits = dense.search(query, k=candidates, flt=flt, collection=collection)
    dense_ids = [h["chunk_id"] for h in dense_hits]
    rankings = [dense_ids]
    if use_bm25:
        bm25_ids = [cid for cid, _ in _bm25().search(query, k=candidates, tokenizer=tokenize)]
        if flt is not None:
            # BM25 인덱스는 Qdrant 밖에 있어 필터를 모른다. Dense 가 필터를 통과시킨 문서(doc_id) 안에서만 받는다.
            allowed_docs = {h["doc_id"] for h in dense_hits}
            by_id = _by_id()
            bm25_ids = [cid for cid in bm25_ids if by_id.get(cid, {}).get("doc_id") in allowed_docs]
        rankings.append(bm25_ids)
    fused = rrf(rankings)
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [_hydrate(cid, score) for cid, score in top]
