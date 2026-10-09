"""Keyword retrieval (BM25) ranked for applicability, authority and currency, not similarity alone."""
import math
import re
from collections import Counter

from .config import TOP_K_CHUNKS
from .documents import chunk_documents, load_documents

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

    def search(self, query, include_superseded=False, k=TOP_K_CHUNKS, extra_terms=""):
        q = tokens(query + " " + extra_terms)
        q = q + [t for w in q for t in tokens(SYNONYMS.get(w, ""))]
        scored = []
        for i, c in enumerate(self.chunks):
            s = self._bm25(q, i)
            if s <= 0:
                continue
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
        return [(s, c) for s, c in scored[:k] if s >= 0.3 * top]
