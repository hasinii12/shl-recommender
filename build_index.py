"""Run this once at build time to pre-compute and save the FAISS index."""
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle
from catalog_data import CATALOG

def build_document(item):
    return (
        f"{item['name']}. {item['description']} "
        f"Levels: {' '.join(item.get('job_levels', []))}. "
        f"Keywords: {' '.join(item.get('keywords', []))}. "
        f"Types: {' '.join(item.get('test_type', []))}."
    )

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = [build_document(item) for item in CATALOG]

print("Embedding catalog...")
embeddings = model.encode(docs, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "catalog.index")
with open("catalog_meta.pkl", "wb") as f:
    pickle.dump(CATALOG, f)

print(f"Saved index with {index.ntotal} vectors.")