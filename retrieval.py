"""
Retrieval layer: embeds the SHL catalog and supports semantic search via FAISS.
Falls back to keyword scoring if embeddings are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from catalog_data import CATALOG, search_catalog_by_keywords

logger = logging.getLogger(__name__)

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss  # noqa: F401

    _EMBEDDING_AVAILABLE = True
except ImportError:
    _EMBEDDING_AVAILABLE = False
    logger.warning("sentence-transformers / faiss not available – using keyword fallback")

_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
_INDEX: Optional["faiss.IndexFlatIP"] = None
_EMBEDDER: Optional["SentenceTransformer"] = None
_INDEXED_CATALOG: list[dict] = []


def _build_document(item: dict) -> str:
    """Build a rich text document from a catalog item for embedding."""
    type_map = {"A": "Ability & Aptitude", "B": "Situational Judgement", "C": "Competencies",
                "D": "Development & 360", "E": "Assessment Exercises", "K": "Knowledge & Skills",
                "P": "Personality & Behavior", "S": "Simulations"}
    types = ", ".join(type_map.get(t, t) for t in item.get("test_type", []))
    levels = ", ".join(item.get("job_levels", []))
    keywords = ", ".join(item.get("keywords", []))
    langs = ", ".join(item.get("languages", []))
    duration = item.get("duration_minutes", "")
    adaptive = "adaptive/IRT" if item.get("adaptive") else "fixed-form"
    remote = "supports remote testing" if item.get("remote_testing") else ""

    return (
        f"Assessment: {item['name']}. "
        f"Type: {types}. "
        f"Description: {item['description']} "
        f"Suitable job levels: {levels}. "
        f"Key topics and roles: {keywords}. "
        f"Languages: {langs}. "
        f"Duration: {duration} minutes. "
        f"Format: {adaptive}. {remote}."
    )


def initialize_index() -> None:
    """Build FAISS index from catalog. Called once at startup."""
    global _INDEX, _EMBEDDER, _INDEXED_CATALOG

    if not _EMBEDDING_AVAILABLE:
        logger.info("Skipping FAISS build – embedding libraries not available")
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", _MODEL_NAME)
    try:
        _EMBEDDER = SentenceTransformer(_MODEL_NAME)
    except Exception as e:
        logger.warning(
            "Failed to load embedding model — falling back to keyword search. Error: %s", e
        )
        return

    documents = [_build_document(item) for item in CATALOG]
    _INDEXED_CATALOG = list(CATALOG)

    logger.info("Embedding %d catalog items …", len(documents))
    embeddings = _EMBEDDER.encode(documents, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised vecs = cosine sim
    index.add(embeddings)
    _INDEX = index
    logger.info("FAISS index ready (%d vectors, dim=%d)", index.ntotal, dim)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return up to top_k catalog items ranked by semantic similarity to query."""
    if _INDEX is None or _EMBEDDER is None:
        logger.debug("Falling back to keyword search")
        words = query.lower().split()
        return search_catalog_by_keywords(words, top_k=top_k)

    import faiss  # noqa

    q_vec = _EMBEDDER.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    k = min(top_k, len(_INDEXED_CATALOG))
    distances, indices = _INDEX.search(q_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        item = _INDEXED_CATALOG[idx].copy()
        item["_score"] = float(dist)
        results.append(item)
    return results


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Combine semantic search with keyword boosting for better precision.
    Items that appear in both semantic results and keyword matches are scored higher.
    """
    sem_results = semantic_search(query, top_k=top_k * 2)
    kw_results = search_catalog_by_keywords(query.lower().split(), top_k=top_k * 2)

    kw_names = {r["name"] for r in kw_results}
    merged: dict[str, dict] = {}

    for rank, item in enumerate(sem_results):
        name = item["name"]
        score = item.get("_score", 0.5) + (1.0 / (rank + 1)) * 0.3
        if name in kw_names:
            score += 0.25  # boost for keyword overlap
        merged[name] = {**item, "_final_score": score}

    for rank, item in enumerate(kw_results):
        name = item["name"]
        if name not in merged:
            score = (1.0 / (rank + 1)) * 0.2
            merged[name] = {**item, "_final_score": score}

    ranked = sorted(merged.values(), key=lambda x: x["_final_score"], reverse=True)
    return ranked[:top_k]


def get_item_by_name(name: str) -> dict | None:
    """Exact or fuzzy name lookup."""
    name_lower = name.strip().lower()
    for item in CATALOG:
        if item["name"].lower() == name_lower:
            return item
    # partial match
    for item in CATALOG:
        if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
            return item
    return None


def get_all_names() -> list[str]:
    return [item["name"] for item in CATALOG]
