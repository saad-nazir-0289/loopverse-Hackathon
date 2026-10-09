"""Load the 13 documents with their metadata, strip hidden instructions, and chunk them.

Documents are untrusted evidence. Anything that tries to instruct the assistant is
removed before retrieval and before the LLM sees it, and the removal is recorded.
"""
import re
from dataclasses import dataclass, field

from .config import DOCS_DIR

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*$", re.S | re.M)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
# Visible text that addresses the assistant instead of the reader
INSTRUCTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"system (note|prompt|message)",
    r"\b(note|message) to (the )?(assistant|ai|model)\b",
    r"do not (mention|cite|reveal)",
    r"from now on,? (always|you)",
    r"regardless of the actual",
    r"you are (now )?(an?|the) (assistant|ai)",
]
INSTRUCTION_RE = re.compile("|".join(INSTRUCTION_PATTERNS), re.I)


@dataclass
class Document:
    doc_id: str
    title: str
    authority: str
    published_date: str
    status: str
    supersedes: str = ""
    superseded_by: str = ""
    text: str = ""
    source_file: str = ""
    removed: list = field(default_factory=list)  # instruction-like content that was stripped

    @property
    def official(self):
        return "unofficial" not in self.authority.lower()

    @property
    def current(self):
        return self.status.lower() == "current"


@dataclass
class Chunk:
    chunk_id: str
    doc: Document
    heading: str
    text: str


def _parse_meta(block):
    meta = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def sanitize(text):
    """Remove hidden HTML comments and visible instruction-like sentences."""
    removed = [m.strip() for m in HTML_COMMENT.findall(text)]
    text = HTML_COMMENT.sub("", text)
    kept = []
    for line in text.splitlines():
        if INSTRUCTION_RE.search(line):
            removed.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept), removed


def load_documents(docs_dir=DOCS_DIR):
    docs = []
    for path in sorted(docs_dir.glob("*.md")):  # any markdown file with a metadata block
        raw = path.read_text(encoding="utf-8")
        m = FRONT_MATTER.search(raw)  # may not be at the top (DOC-03 has a note above it)
        if not m or "document_id" not in m.group(1):
            print(f"warning: {path.name} has no document_id metadata block; skipped")
            continue
        meta = _parse_meta(m.group(1))
        body = (raw[:m.start()] + "\n" + raw[m.end():]).strip()
        body, removed = sanitize(body)
        docs.append(Document(
            doc_id=meta["document_id"], title=meta.get("title", ""), authority=meta.get("authority", ""),
            published_date=meta.get("published_date", ""), status=meta.get("status", "current"),
            supersedes=meta.get("supersedes", ""), superseded_by=meta.get("superseded_by", ""),
            text=body, source_file=path.name, removed=removed))
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)), "duplicate document ids"
    return docs


def chunk_documents(docs):
    """One chunk per section (markdown heading or bold label); short docs stay whole."""
    chunks = []
    for d in docs:
        sections, heading, buf = [], d.title, []
        for line in d.text.splitlines():
            h = re.match(r"^#{1,6}\s+(.*)", line) or re.match(r"^\*\*(.+?)\*\*\s*$", line)
            if h:
                if any(x.strip() for x in buf):
                    sections.append((heading, "\n".join(buf).strip()))
                heading, buf = h.group(1).strip(), []
            else:
                buf.append(line)
        if any(x.strip() for x in buf):
            sections.append((heading, "\n".join(buf).strip()))
        # FAQ: each bold question is a heading-with-answer pair inside one paragraph
        for i, (hd, tx) in enumerate(sections):
            chunks.append(Chunk(f"{d.doc_id}#{i}", d, hd, tx))
    return chunks
