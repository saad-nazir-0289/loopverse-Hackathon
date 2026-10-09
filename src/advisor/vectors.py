"""Local vector index for the document chunks (the "vector database").

Backends (ADVISOR_EMBEDDINGS):
  "openai" - text-embedding-3-small via the OpenAI-compatible API (needs a key)
  "tfidf"  - local character n-gram TF-IDF vectors (no key, no download)
  "auto"   - openai if an LLM key is configured, else tfidf (default)
Vectors are stored in data/vector_index/<backend>.npz and rebuilt automatically when the
chunk texts change (content hash), so judges' documents are indexed on first use.
"""
import hashlib
import json
import os

import numpy as np

from . import config

INDEX_DIR = config.ROOT / "data" / "vector_index"
EMBED_MODEL = os.environ.get("ADVISOR_EMBED_MODEL", "text-embedding-3-small")


def backend():
    b = os.environ.get("ADVISOR_EMBEDDINGS", "auto")
    if b == "auto":
        return "openai" if os.environ.get("OPENAI_API_KEY") and config.LLM_PROVIDER == "openai" else "tfidf"
    return b


def _chunk_text(c):
    return f"{c.doc.title}. {c.heading}. {c.text}"


class VectorIndex:
    def __init__(self, chunks, kind=None):
        self.kind = kind or backend()
        self.chunks = chunks
        texts = [_chunk_text(c) for c in chunks]
        digest = hashlib.sha1(json.dumps([self.kind, EMBED_MODEL, texts]).encode()).hexdigest()
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            # char n-grams: robust to spelling variants (Roman Urdu) and word forms
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
            self.M = self._norm(self.vec.fit_transform(texts).toarray())
            return
        path = INDEX_DIR / f"{self.kind}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["digest"]) == digest:
                self.M = z["M"]
                return
        self.M = self._norm(self._embed(texts))
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, M=self.M, digest=digest, ids=np.array([c.chunk_id for c in chunks]))

    @staticmethod
    def _norm(M):
        M = np.asarray(M, dtype=np.float32)
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

    def _embed(self, texts):
        from openai import OpenAI
        client = OpenAI(timeout=config.LLM_TIMEOUT, max_retries=2)
        out = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in out.data]

    def similarities(self, query):
        if self.kind == "tfidf":
            q = self._norm(self.vec.transform([query]).toarray())
        else:
            q = self._norm(self._embed([query]))
        return (self.M @ q[0]).astype(float)
