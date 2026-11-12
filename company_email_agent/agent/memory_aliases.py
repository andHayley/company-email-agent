"""Conservative recognition of explicit user-defined document aliases."""

from collections.abc import Iterable
from functools import cache
import re
from unicodedata import normalize

from company_email_agent import data


_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_WORD = re.compile(r"[\u3400-\u9fff]+|[^\W\u3400-\u9fff]+(?:-[^\W\u3400-\u9fff]+)*")
_ALIAS_TEXT = re.compile(r"[\w-]+(?:\s+[\w-]+){0,3}")
_CLAUSE_BREAK = re.compile(r"[;；\n]+|(?<=[.!?。！？])\s+")
_SHORTHAND_PREFIX = re.compile(
    r"^(?:(?:the )?user(?:'s)? (?:shorthand|alias)(?:\s*:\s*|\s+)|(?:以后|今后)?(?:我说|用户说)\s*)"
)
_WHEN = re.compile(r"when (?:the user says|they say|i say)\s+(.+?)(?:\s*,\s*|\s+)(?:they|i) (?:always )?mean\s+(.+)")
_BINDING = re.compile(r"(.+?)(\s+(?:refers to|means)\s+|\s*[，,]?\s*(?:就是指|指的是|指|=)\s*)(.+)")
_NESTED_BINDING = re.compile(r"\b(?:refers to|means)\b|就是指|指的是|=")
_UNCERTAIN = re.compile(r"\b(?:or|either|maybe|sometimes|possibly|not)\b|或者|还是|可能|不指")
_REFERENCE = re.compile(r"[\\/]|(?<!\w)[\w-]+\.[a-z][a-z0-9]{0,7}\b|\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d{2,}\b")
_LOOKUP_WORDS = frozenset("""
    author authored authors wrote written version revision date dated published
    updated release released identifier id number
    作者 作者是谁 的作者是谁 谁写的 是谁写的 版本 的版本 版本是什么 的版本是什么
    当前版本是什么 的当前版本是什么 修订号 编号 文档编号 日期
""".split())
_SCOPE_WORDS = _LOOKUP_WORDS | frozenset("""
    a an and about are by can could current did do does find for get give how in
    information is it latest look me of on please tell that the this to up was
    what when which who whose would you s document details 请问 的 是 和 以及
""".split())


def _normalized(text: str) -> str:
    return normalize("NFKC", text).translate(_QUOTES).casefold()


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(_normalized(text)))


def _contains(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    return any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))


def _explicit_binding(clause: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    original = _normalized(clause).strip().rstrip(".。")
    text = _SHORTHAND_PREFIX.sub("", original)
    if _UNCERTAIN.search(text):
        return None
    conditional = _WHEN.fullmatch(text)
    if conditional:
        alias, meaning = conditional.groups()
    else:
        match = _BINDING.fullmatch(text)
        if match is None:
            return None
        alias, connector, meaning = match.groups()
        if text == original and alias[:1] not in ('"', "'"):
            if connector.strip() == "means" or (connector == "指" and "\u3400" <= alias[-1] <= "\u9fff"):
                return None
    if _NESTED_BINDING.search(meaning):
        return None
    alias = alias.strip()
    if alias[:1] in ('"', "'") and alias[-1:] == alias[:1]:
        alias = alias[1:-1].strip()
    if _ALIAS_TEXT.fullmatch(alias) is None:
        return None
    alias_words = _words(alias)
    meaning_words = _words(meaning)
    if (not set(alias_words) - _SCOPE_WORDS or not 1 <= len(meaning_words) <= 16
            or not any(char.isalpha() for char in meaning)):
        return None
    if meaning_words[0] in ("the", "a", "an"):
        meaning_words = meaning_words[1:]
    if not meaning_words or meaning_words == alias_words:
        return None
    return alias_words, meaning_words


@cache
def _document_names() -> tuple[tuple[str, ...], ...]:
    return tuple(_words(name) for doc in data.load_docs() for name in (doc["title"], doc["doc_id"]) if name)


def needs_alias_recall_hint(question: str, query: str, facts: Iterable[str]) -> bool:
    """Return a yes/no routing signal. No memory text is sent to the model here."""
    question_words = _words(question)
    query_words = set(_words(query))
    if (not set(question_words) & _LOOKUP_WORDS or not query_words
            or _REFERENCE.search(_normalized(question)) or _REFERENCE.search(_normalized(query))):
        return False

    matches = set()
    unclear = []
    for fact in facts:
        for clause in _CLAUSE_BREAK.split(fact):
            binding = _explicit_binding(clause)
            if binding is None:
                unclear.append(_words(clause))
            elif _contains(question_words, binding[0]):
                matches.add(binding)
    if len(matches) != 1:
        return False

    alias, _ = matches.pop()
    allowed_words = _SCOPE_WORDS | set(alias)
    if (set(question_words) - allowed_words or query_words - allowed_words
            or any(_contains(words, alias) for words in unclear)):
        return False

    # Even an explicitly saved alias must yield to a named document in this turn.
    return not any(_contains(question_words, name) for name in _document_names() if name)
