"""Hybrid retrieval (BM25 + vector index) ranked for applicability, authority and currency, not similarity alone."""
import math
import os
import re
from collections import Counter

from .config import TOP_K_CHUNKS
from .documents import chunk_documents, load_documents

# auto = hybrid when semantic embeddings are available, else bm25 (local tfidf vectors add nothing: tests/retrieval_eval.py)
RETRIEVAL_MODE = os.environ.get("ADVISOR_RETRIEVAL", "auto")  # auto | hybrid | bm25 | vector
VECTOR_WEIGHT = 2.0  # hybrid: semantic rank counts double (eval: vector 96% vs bm25 88% hit@3)
STOP = set("""a an the of to in on for and or is are was were be been it this that with as at by from what which who
how do does can could should would will i we you my our your me us about any there their they them if then than
so not no yes please tell kya hai hain ka ki ke ko se mein main aur bhi""".split())
# English synonyms so policy words match how people ask
SYNONYMS = {"close": "closure closed shut", "closed": "closure close", "shut": "closure close", "kids": "children",
            "child": "children", "jog": "exercise outdoor", "jogging": "exercise outdoor", "run": "exercise",
            "running": "exercise", "walk": "outdoor exercise", "gym": "exercise", "car": "vehicle vehicles",
            "cars": "vehicle vehicles", "number": "plate", "bus": "transport metrobus", "metro": "metrobus transport",
            "old": "elderly", "senior": "elderly", "grandparents": "elderly", "pregnancy": "pregnant",
            "asthmatic": "asthma", "inhaler": "asthma inhalers", "purifiers": "purifier hepa", "filter": "hepa filtration",
            "mask": "masks n95", "brick": "kiln kilns", "burning": "burn", "factory": "industries industrial",
            "doctor": "hospital", "er": "emergency", "breathless": "breath shortness", "unsafe": "hazardous",
            "dangerous": "hazardous", "school": "schools", "schools": "school", "sports": "sport outdoor",
            "aqi": "aqi index", "sensor": "sensors network", "covered": "coverage areas", "cities": "city coverage"}


def tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9.]+", text.lower()):
        w = w.strip(".")
        if not w or w in STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude singular
        out.append(w)
    return out


class Retriever:
    def __init__(self):
        self.docs = load_documents()
        self.chunks = chunk_documents(self.docs)
        self.tf = [Counter(tokens(f"{c.doc.title} {c.heading} {c.text}")) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.by_id = {d.doc_id: d for d in self.docs}

    def _bm25(self, q_tokens, i, k1=1.4, b=0.75):
        tf, s = self.tf[i], 0.0
        for w in q_tokens:
            if w in tf:
                f = tf[w]
                s += self.idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.len[i] / self.avg))
        return s

    def _vectors(self):
        """Lazy vector index; falls back to local TF-IDF vectors if the embedding API fails."""
        if getattr(self, "_vi", None) is None:
            from .vectors import VectorIndex, backend
            try:
                self._vi = VectorIndex(self.chunks, backend())
            except Exception as e:
                print(f"note: {backend()} embeddings unavailable ({type(e).__name__}); using local tfidf vectors")
                self._vi = VectorIndex(self.chunks, "tfidf")
        return self._vi

    def search(self, query, include_superseded=False, k=TOP_K_CHUNKS, extra_terms="", mode=None):
        """mode: "hybrid" (default: BM25 + vector, reciprocal rank fusion), "bm25" or "vector"."""
        mode = mode or RETRIEVAL_MODE
        if mode == "auto":
            from .vectors import backend
            mode = "hybrid" if backend() != "tfidf" else "bm25"
        q = tokens(query + " " + extra_terms)
        q = q + [t for w in q for t in tokens(SYNONYMS.get(w, ""))]
        bm = [self._bm25(q, i) for i in range(len(self.chunks))]
        if mode == "bm25":
            fused, keep = bm, [x > 0 for x in bm]
        else:
            try:
                sims = self._vectors().similarities(query + " " + extra_terms)
            except Exception:
                sims = None
            if sims is None:
                fused, keep = bm, [x > 0 for x in bm]
            else:
                vr = {i: r for r, i in enumerate(sorted(range(len(sims)), key=lambda i: -sims[i]))}
                br = {i: r for r, i in enumerate(sorted(range(len(bm)), key=lambda i: -bm[i]))}
                top_sim = max(sims)
                if mode == "vector":
                    fused = list(sims)
                    keep = [s_ >= 0.8 * top_sim for s_ in sims]
                else:  # reciprocal rank fusion; a chunk needs keyword overlap or a strong vector match
                    fused = [(1 / (60 + br[i]) if bm[i] > 0 else 0) + VECTOR_WEIGHT / (60 + vr[i])
                             for i in range(len(bm))]
                    keep = [bm[i] > 0 or sims[i] >= 0.85 * top_sim for i in range(len(bm))]
        scored = []
        for i, c in enumerate(self.chunks):
            if not keep[i]:
                continue
            s = fused[i]
            # applicability and authority, not similarity alone
            if not c.doc.current:
                if not include_superseded:
                    continue
                s *= 0.8
            if not c.doc.official:
                s *= 0.5  # unofficial community bulletin
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return []
        top = scored[0][0]
        cut = 0.3 if mode == "bm25" else 0.45
        return [(s, c) for s, c in scored[:k] if s >= cut * top]
