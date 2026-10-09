"""Retrieval quality: does the right current document come back? BM25 vs vector vs hybrid.

Run:  python tests/retrieval_eval.py            (tfidf vectors, no key)
      ADVISOR_EMBEDDINGS=openai python tests/retrieval_eval.py
hit@3 = an expected document is among the top 3 retrieved documents; MRR = 1 / rank of the first hit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from advisor.retrieval import Retriever  # noqa: E402
from advisor.understand import understand  # noqa: E402

# (question, acceptable documents). Second half = paraphrases with little keyword overlap.
LABELLED = [
    ("What should people with asthma do when smog is hazardous?", {"DOC-11", "DOC-01"}),
    ("Do cloth masks protect against PM2.5?", {"DOC-12", "DOC-01"}),
    ("When does the odd-even vehicle restriction apply?", {"DOC-05"}),
    ("What must brick kilns do during a smog alert?", {"DOC-08"}),
    ("At what PM2.5 level do schools close fully?", {"DOC-03"}),
    ("What do residents recommend in the community bulletin?", {"DOC-07"}),
    ("What is the difference between AQI and PM2.5?", {"DOC-04"}),
    ("Do hospitals have extended OPD hours during smog?", {"DOC-09"}),
    ("Does the Metrobus run more often on smog days?", {"DOC-10"}),
    ("Which areas does the forecast cover?", {"DOC-13", "DOC-06"}),
    ("Is it okay to go for a run when the air is terrible?", {"DOC-04", "DOC-01"}),
    ("What precautions should my grandma take?", {"DOC-11", "DOC-01"}),
    ("Can factories keep running when pollution is extreme?", {"DOC-08"}),
    ("Do I need an air filter at home?", {"DOC-12", "DOC-04", "DOC-01"}),
    ("Can I burn leaves in my garden?", {"DOC-08"}),
    ("Who announces official smog warnings?", {"DOC-04"}),
    ("Where can I leave my car and take the bus?", {"DOC-10"}),
    ("My kid keeps coughing, when should I see a doctor?", {"DOC-11", "DOC-01", "DOC-09"}),
    ("I am expecting a baby, is going outside risky?", {"DOC-11", "DOC-01"}),
    ("When will they tell parents about school timing changes?", {"DOC-03"}),
    ("Buzurgon ko smog mein kya ehtiyat karni chahiye?", {"DOC-11", "DOC-01"}),
    ("Gari kab nahi chala sakte odd even mein?", {"DOC-05"}),
    ("Ghar ke andar hawa saaf kaise rakhein?", {"DOC-12", "DOC-04", "DOC-07"}),
    ("Saans lene mein mushkil ho to kya karein?", {"DOC-01", "DOC-09", "DOC-11"}),
]


def evaluate(r, mode):
    hits, rr, misses = 0, 0.0, []
    for q, want in LABELLED:
        u = understand(q)
        docs = list(dict.fromkeys(c.doc.doc_id for _, c in r.search(u.retrieval_query, mode=mode)))
        rank = next((i + 1 for i, d in enumerate(docs) if d in want), None)
        hits += bool(rank and rank <= 3)
        rr += 1 / rank if rank else 0
        if not rank or rank > 3:
            misses.append((q[:45], docs[:3]))
    return hits / len(LABELLED), rr / len(LABELLED), misses


if __name__ == "__main__":
    r = Retriever()
    print(f"vector backend: {r._vectors().kind}   ({len(LABELLED)} labelled questions)")
    for mode in ("bm25", "vector", "hybrid"):
        h, mrr, miss = evaluate(r, mode)
        print(f"{mode:7s} hit@3 {h:.0%}  MRR {mrr:.2f}  misses: {miss}")
