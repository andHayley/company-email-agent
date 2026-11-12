"""BM25 over short document passages, with line-addressable source reads."""

from collections.abc import Iterator
from dataclasses import dataclass

from company_email_agent.pipelines.bm25 import BM25Index, build_index
from company_email_agent.tokenizer import tokenize
from company_email_agent.tokens import CHARS_PER_TOKEN
from company_email_agent.types import DocWindow, SearchResult


_SNIPPET_CHARS = 150 * CHARS_PER_TOKEN
_WINDOW_TARGET = 750 * CHARS_PER_TOKEN
_WINDOW_MAX = 1500 * CHARS_PER_TOKEN


@dataclass(frozen=True)
class Passage:
    doc_id: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class Document:
    title: str
    lines: list[str]


class LocatedResult(SearchResult):
    start_line: int


@dataclass(frozen=True)
class DocIndex:
    documents: dict[str, Document]
    passages: list[Passage]
    bm25: BM25Index

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            return []
        ranked = self.bm25.search(query, len(self.passages))
        results: list[SearchResult] = []
        seen: set[str] = set()
        for passage_id, _ in ranked:
            passage = self.passages[passage_id]
            if passage.doc_id in seen:
                continue
            doc = self.documents[passage.doc_id]
            snippet = passage.text.strip()
            if len(tokenize(snippet)) < 4:
                # A short heading or table cell needs the neighboring source text.
                room = _SNIPPET_CHARS - len(snippet) - 2
                before = "\n".join(doc.lines[:passage.start_line - 1]).rstrip()
                after = "\n".join(doc.lines[passage.end_line:]).lstrip()
                prefix = before[-(room // 2):] if room >= 2 else ""
                snippet = "\n".join((prefix, snippet, after[:room - len(prefix)])).strip()
            results.append(LocatedResult(doc_id=passage.doc_id, title=doc.title,
                                         snippet=snippet, start_line=passage.start_line))
            seen.add(passage.doc_id)
            if len(results) == k:
                break
        return results

    def read(self, doc_id: str, start_line: int) -> DocWindow:
        doc = self.documents[doc_id]
        if not 1 <= start_line <= len(doc.lines):
            raise ValueError(f"Line {start_line} is outside {doc_id} (1–{len(doc.lines)}).")
        if len(doc.lines[start_line - 1]) > _WINDOW_MAX:
            raise ValueError(f"Line {start_line} exceeds the window limit by itself.")

        end_line = start_line
        size = len(doc.lines[start_line - 1])
        while end_line < len(doc.lines):
            extra = 1 + len(doc.lines[end_line])
            if size + extra > _WINDOW_TARGET:
                break
            size += extra
            end_line += 1
        return {
            "doc_id": doc_id, "start_line": start_line, "end_line": end_line,
            "total_lines": len(doc.lines), "text": "\n".join(doc.lines[start_line - 1:end_line]),
        }


def _passages(doc_id: str, lines: list[str]) -> Iterator[Passage]:
    chunk: list[str] = []
    start, size = 1, 0
    for number, line in enumerate(lines, 1):
        if chunk and size + 1 + len(line) > _SNIPPET_CHARS:
            yield Passage(doc_id, start, number - 1, "\n".join(chunk))
            chunk, size = [], 0
        if len(line) > _SNIPPET_CHARS:
            # Overlap slices so words at a boundary still occur in a passage.
            for offset in range(0, len(line), _SNIPPET_CHARS - 100):
                yield Passage(doc_id, number, number, line[offset:offset + _SNIPPET_CHARS])
            continue
        if not chunk:
            start = number
        else:
            size += 1
        chunk.append(line)
        size += len(line)
    if chunk:
        yield Passage(doc_id, start, len(lines), "\n".join(chunk))


def build_doc_index(docs: list[dict]) -> DocIndex:
    documents = {doc["doc_id"]: Document(doc["title"], doc["text"].splitlines()) for doc in docs}
    passages = [
        passage for doc_id, doc in documents.items()
        for passage in _passages(doc_id, doc.lines) if passage.text.strip()
    ]
    return DocIndex(documents, passages, build_index([passage.text for passage in passages]))
