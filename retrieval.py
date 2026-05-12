"""
Retrieval layer: loads pre-built FAISS index at startup.
Index is built once during deployment (build_index.py).
Falls back to keyword search if index file not found.
"""

from __future__ import annotations
import logging
import os
import pickle
import numpy as np
from catalog_data import CATALOG, search_catalog_by_keywords

logger = logging.getLogger(__name__)

_INDEX = None
_EMBEDDER = None
_INDEXED_CATALOG = []


def initialize_index() -> None:
    global _INDEX, _EMBEDDER, _INDEXED_CATALOG

    if not os.path.exists("catalog.index"):
        logger.warning("No pre-built index found — using keyword search fallback.")
        return

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info("Loading pre-built FAISS index...")
        _INDEX = faiss.read_index("catalog.index")

        with open("catalog_meta.pkl", "rb") as f:
            _INDEXED_CATALOG = pickle.load(f)

        logger.info("Loading embedding model...")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("FAISS index ready — %d vectors.", _INDEX.ntotal)

    except Exception as e:
        logger.warning("Failed to load FAISS index: %s — falling back to keyword search.", e)
        _INDEX = None
        _EMBEDDER = None


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    if _INDEX is None or _EMBEDDER is None:
        return search_catalog_by_keywords(query.lower().split(), top_k=top_k)

    import faiss
    q_vec = _EMBEDDER.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k * 2, len(_INDEXED_CATALOG))
    distances, indices = _INDEX.search(q_vec, k)

    sem_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            item = _INDEXED_CATALOG[idx].copy()
            item["_score"] = float(dist)
            sem_results.append(item)

    # Keyword boost
    kw_results = search_catalog_by_keywords(query.lower().split(), top_k=top_k * 2)
    kw_names = {r["name"] for r in kw_results}

    merged = {}
    for rank, item in enumerate(sem_results):
        name = item["name"]
        score = item.get("_score", 0.5) + (1.0 / (rank + 1)) * 0.3
        if name in kw_names:
            score += 0.25
        merged[name] = {**item, "_final_score": score}

    for rank, item in enumerate(kw_results):
        name = item["name"]
        if name not in merged:
            merged[name] = {**item, "_final_score": (1.0 / (rank + 1)) * 0.2}

    return sorted(merged.values(), key=lambda x: x["_final_score"], reverse=True)[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    return hybrid_search(query, top_k)


def get_item_by_name(name: str) -> dict | None:
    name_lower = name.strip().lower()
    for item in CATALOG:
        if item["name"].lower() == name_lower:
            return item
    for item in CATALOG:
        if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
            return item
    return None


def get_all_names() -> list[str]:
    return [item["name"] for item in CATALOG]