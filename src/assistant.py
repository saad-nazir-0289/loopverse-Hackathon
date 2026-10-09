"""Entry point for Sprint 2.

    from assistant import ask, forecast        # run from src/, or add src/ to sys.path
    python src/assistant.py "Should schools in DHA close tomorrow?"
"""
import json
import sys

from advisor import ask, forecast  # noqa: F401  (re-exported)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or input("Question: ")
    print(json.dumps(ask(q), ensure_ascii=False, indent=2))
