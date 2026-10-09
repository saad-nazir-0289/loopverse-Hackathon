"""Answer the released question set in one run.

Usage:
    python src/run_questions.py spec/25_unseen_questions.md      # .md / .txt / .csv / .json
    python src/run_questions.py questions.csv --out answers.json

Accepted input formats:
- .json: list of strings, or list of objects with "question" (+ optional "question_id"/"id")
- .csv : columns question (+ optional question_id/id)
- .md / .txt: one question per line; leading "1." / "Q1:" / "- " / "**Q1**" labels become the question_id
Writes answers.json (list of ask() results) and answers.csv next to it.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from advisor.ask import _ask

sys.stdout.reconfigure(encoding="utf-8")
# "1." "Q1:" "A3)" "**Q12**" "- " "Question 4 -" ...
LABEL = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*((?:question\s*|[a-z]{1,2})?\d+)\s*(?:\*\*)?\s*[.):\-]\s*(?:\*\*)?\s*", re.I)
BULLET = re.compile(r"^\s*[-*]\s+")


def _clean_id(raw):
    raw = raw.replace(" ", "")
    if raw.isdigit():
        return f"Q{raw}"
    return re.sub(r"(?i)^question", "Q", raw).upper()


def load_questions(path):
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return [q if isinstance(q, dict) else {"question": q} for q in data]
    if p.suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    out, skipped = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or re.fullmatch(r"[|:\- ]+", s):
            continue
        if s.startswith("|"):  # markdown table row: | id | question | ...
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 2 and cells[1] and cells[1].lower() not in ("question", "questions"):
                out.append({"question_id": _clean_id(cells[0]) if cells[0] else None, "question": cells[1]})
            continue
        m = LABEL.match(s)
        qid = _clean_id(m.group(1)) if m else None
        body = s[m.end():] if m else BULLET.sub("", s)
        body = body.strip().strip("*").strip()
        if len(body) > 3 and (m or BULLET.match(s) or body.endswith("?") or p.suffix == ".txt"):
            out.append({"question_id": qid, "question": body})
        else:
            skipped.append(s)
    for s in skipped:
        print(f"WARNING: line not read as a question: {s[:80]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--out", default="answers.json")
    ap.add_argument("--trace", action="store_true", help="also write answers_trace.json (mode, validation)")
    args = ap.parse_args()
    qs = load_questions(args.questions)
    print(f"{len(qs)} questions loaded from {args.questions}")
    results, traces = [], []
    for i, q in enumerate(qs, 1):
        qid = q.get("question_id") or q.get("id") or f"Q{i}"
        r, t = _ask(q.get("question") or q.get("text", ""), question_id=qid)
        results.append(r)
        traces.append({"question_id": qid, "mode": t.get("mode"), "validation": t.get("validation"),
                       "model": (t.get("llm") or {}).get("model")})
        print(f"[{qid}] mode={t.get('mode'):13s} forecast_called={r['forecast_called']!s:5s} sources={r['sources']}")
    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "answer", "sources", "forecast_called"])
        for r in results:
            w.writerow([r["question_id"], r["answer"], ";".join(r["sources"]), r["forecast_called"]])
    if args.trace:
        out.with_name(out.stem + "_trace.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
