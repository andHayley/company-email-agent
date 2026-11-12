"""A directory-scoped store with one atomic record per remembered fact."""

from collections.abc import Iterator
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time_ns
from uuid import uuid4

from company_email_agent.tokenizer import tokenize


# A single labeled identifier is a current-value field. Free-form memories,
# aliases and lists can contain legitimate conflicting or complementary facts.
_IDENTIFIER = re.compile(
    r"(?P<label>[a-z][a-z -]{0,60} (?:number|id|identifier|code)):\s*(?P<value>[a-z0-9-]+)[.]?",
    re.IGNORECASE,
)


_FILLER = frozenset("a about an and are for is me my of on our the to user was what which who".split())


class MemoryStore:
    def __init__(self, directory: Path | None):
        self._directory = None if directory is None else Path(directory)

    def remember(self, fact: str) -> str:
        if self._directory is None:
            raise ValueError("Persistent memory is not configured for this session.")
        fact = fact.strip()
        if not fact:
            raise ValueError("Provide a nonempty fact to store.")

        self._directory.mkdir(parents=True, exist_ok=True)
        record = self._directory / f"fact-{time_ns():020d}-{uuid4().hex}.json"
        # Separate files avoid losing updates when sessions save concurrently.
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._directory, prefix=".memory-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump({"fact": fact}, stream, ensure_ascii=False)
            temporary.replace(record)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return "Saved for future conversations."

    def facts(self) -> Iterator[str]:
        """Yield readable stored facts newest first, without ranking or writing."""
        if self._directory is None:
            return
        for path in sorted(self._directory.glob("fact-*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            fact = payload.get("fact") if isinstance(payload, dict) else None
            if isinstance(fact, str) and fact.strip():
                yield fact

    def recall(self, query: str) -> str:
        if self._directory is None:
            return "Persistent memory is not configured for this session."

        terms = set(tokenize(query)) - _FILLER
        ranked: list[tuple[int, str]] = []
        seen: set[str] = set()
        current_fields: set[str] = set()
        for fact in self.facts():
            if fact in seen:
                continue
            seen.add(fact)
            identifier = _IDENTIFIER.fullmatch(fact.strip())
            if identifier and any(char.isdigit() for char in identifier["value"]):
                field = " ".join(identifier["label"].casefold().split())
                if field in current_fields:
                    continue
                current_fields.add(field)
            overlap = len(terms.intersection(tokenize(fact)))
            if overlap or not terms:
                ranked.append((overlap, fact))

        if not ranked:
            return "No saved facts match this topic."
        ranked.sort(key=lambda match: match[0], reverse=True)
        return "Saved facts, best matches first (newest first on ties):\n" + "\n".join(
            f"- {fact}" for _, fact in ranked[:5]
        )
