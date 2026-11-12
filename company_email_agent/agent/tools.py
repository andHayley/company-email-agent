"""Wire course tool schemas to session-scoped and archive-backed functions."""

from functools import cache
import json
import re
from threading import Lock

from company_email_agent import adapters, data
from company_email_agent.agent.memory import MemoryStore
from company_email_agent.agent.terminal import terminal_output
from company_email_agent.cardinal import cardinal_spec
from company_email_agent.pipelines.daily_digest import write_digest
from company_email_agent.sandbox_docker import DockerTerminal
from company_email_agent.types import AgentConfig, Email, LMCallable, ToolSpec


_EMAIL_INDEX_LOCK = Lock()


@cache
def _email_index():
    return adapters.run_bm25_build([data.email_text(email) for email in data.load_emails()])


_ASSISTANT_NOTE = re.compile(
    r"^[ \t]*>[ \t]*(?:\*\*)?(?:note|notice|instructions?) (?:to|for) "
    r"(?:automated (?:readers?|assistants?|agents?)|AI (?:assistants?|agents?))\b[^\n]*"
    r"(?:\n[ \t]*>[^\n]*)*",
    re.IGNORECASE | re.MULTILINE,
)
_VERSION = re.compile(r"^Policy number: .*?, version ([\d.]+)\s*$", re.MULTILINE)
_SECTION = re.compile(r"^## (\d+)\.", re.MULTILINE)
_AMENDMENT = re.compile(
    r"\b(?:replaces|amends|supersedes)\b[^.\n]*?\bsection (\d+) of the (.+?), version (\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)
_EFFECTIVE = re.compile(r"^Effective: (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


@cache
def _assistant_docs() -> list[dict]:
    # Remove blockquoted directives addressed to automation from the evidence
    # view. Preserve source line positions and the repository document.
    return [
        {**doc, "text": _ASSISTANT_NOTE.sub(lambda note: " \n" * note[0].count("\n") + " ", doc["text"])}
        for doc in data.load_docs()
    ]


@cache
def _doc_index():
    return adapters.run_build_doc_index(_assistant_docs())


@cache
def _linked_amendments() -> dict[str, list[tuple[int, int, str]]]:
    policies = {}
    for doc in _assistant_docs():
        version = _VERSION.search(doc["text"])
        if version:
            title = re.sub(r" \(\d{4}\)$", "", doc["title"]).casefold()
            policies[title, version[1]] = doc

    links: dict[str, list[tuple[int, int, str]]] = {}
    for doc in _assistant_docs():
        metadata = "\n".join(doc["text"].splitlines()[:12])
        effective = _EFFECTIVE.search(metadata)
        if effective is None or "\nPosted: " not in metadata:
            continue
        for reference in _AMENDMENT.finditer(doc["text"]):
            section, title, version = reference.groups()
            policy = policies.get((title.casefold(), version))
            if policy is None or policy["doc_id"] == doc["doc_id"]:
                continue
            sections = list(_SECTION.finditer(policy["text"]))
            for i, heading in enumerate(sections):
                if heading[1] != section:
                    continue
                first = policy["text"].count("\n", 0, heading.start()) + 1
                next_heading = sections[i + 1].start() if i + 1 < len(sections) else len(policy["text"])
                last = max(first, policy["text"].count("\n", 0, next_heading))
                amendment = doc["text"]
                start = amendment.rfind("\n## ", 0, reference.start()) + 1
                end = amendment.find("\n## ", reference.end())
                excerpt = amendment[start:end if end >= 0 else len(amendment)].strip()
                line = amendment.count("\n", 0, start) + 1
                note = (
                    f'Linked amendment for section {section}: {doc["title"]} '
                    f'(doc_id: {doc["doc_id"]}, start_line: {line}, effective {effective[1]}). '
                    f'Check the expense or event date and scope before applying it.\n{excerpt}'
                )
                links.setdefault(policy["doc_id"], []).append((first, last, note))
                break
    return links


def _search_docs(query: str, k: int = 5) -> str:
    matches = adapters.run_search_docs(_doc_index(), query, k)
    return "\n\n".join(
        f'{item["title"]} (doc_id: {item["doc_id"]}, start_line: {item["start_line"]})\n{item["snippet"]}'
        for item in matches
    ) or "No matching documents. Try more distinctive policy or topic keywords."


def _read_doc(doc_id: str, start_line: int = 1) -> str:
    window = adapters.run_read_doc(_doc_index(), doc_id, int(start_line))
    source = f'{doc_id}, lines {start_line}-{window["end_line"]} of {window["total_lines"]}:\n' + window["text"]
    amendments = (
        note for first, last, note in _linked_amendments().get(doc_id, ())
        if first <= window["end_line"] and last >= window["start_line"]
    )
    return "\n\n".join((source, *amendments))


@cache
def _emails_by_id() -> dict[str, Email]:
    return {email["id"]: email for email in data.load_emails()}


def _search_archive(query: str, k: int) -> list[Email]:
    # Several evaluation sessions may make their first archive query at once.
    with _EMAIL_INDEX_LOCK:
        index = _email_index()
    archive = data.load_emails()
    # JSON Schema also accepts an integral JSON number such as 5.0 as an integer.
    return [archive[doc_id] for doc_id, _ in adapters.run_bm25_search(index, query, int(k))]


def _search_emails(query: str, k: int = 5) -> str:
    matches = [
        {"id": email["id"], "date": email["date"], "from": email.get("from", ""),
         "subject": email["subject"], "snippet": email["body"][:500]}
        for email in _search_archive(query, k)
    ]
    return json.dumps(matches, ensure_ascii=False) if matches else "No matching emails."


def _read_email(email_id: str) -> str:
    email = _emails_by_id().get(email_id)
    if email is None:
        raise ValueError(f"Unknown email ID: {email_id!r}.")
    return json.dumps(email, ensure_ascii=False)


def session_tools(
    tools: list[ToolSpec] | None, config: AgentConfig, aux_lm: LMCallable, terminal: DockerTerminal
) -> list[ToolSpec]:
    def ask_user(question: str) -> str:
        if config.user is None:
            raise ValueError("No interactive user is available. Ask the question in your final reply.")
        return config.user.ask(question)

    memory = MemoryStore(config.memory_dir)
    extras = [
        (cardinal_spec(
            "remember",
            "Store a lasting fact or preference stated by the user for later conversations. "
            "Use for requests to save a detail, future reference, and corrections to saved facts. "
            "For corrections, store the current value without repeating the superseded value.",
            memory.remember,
        ), config.memory_dir is not None),
        (cardinal_spec(
            "recall",
            "Look up saved user details and preferences by topic. Use before guessing or asking "
            "about personalized references such as a team, a saved number, or a preferred document. "
            "Use the latest correction when stored details conflict.",
            memory.recall,
        ), config.memory_dir is not None),
        (cardinal_spec("ask_user", "Ask one focused question when a decision, identity, or scope is ambiguous. "
                       "Use available evidence first, and wait for the user's answer before acting.", ask_user),
         config.user is not None),
    ]
    terminal_tool = cardinal_spec(
        "run_terminal",
        "Run one Bash command for user-authorized local analysis in the isolated session workspace "
        "(emails.jsonl and docs/). Use python3 -c '...' or standard text utilities. "
        "No here-documents, command substitution, or network. Request concise output. "
        "A long result may give a relative file to search with another command.",
        lambda cmd: terminal_output(cmd, config, terminal),
    )
    if tools is not None:
        # The reserved tool always retains its validated schema and approval policy.
        by_name = {tool.name: terminal_tool if tool.name == "run_terminal" else tool for tool in tools}
        for tool, enabled in extras:
            if enabled:
                by_name.setdefault(tool.name, tool)
        return list(by_name.values())

    return [
        cardinal_spec("search_emails", "Find 2001 archive emails by keywords. Returns IDs and brief snippets.", _search_emails),
        cardinal_spec("read_email", "Read an archive email by its ID from search results.", _read_email),
        cardinal_spec("email_qa", "Answer a factual or multi-hop question about the 2001 email archive with evidence.",
                      lambda question: json.dumps(adapters.run_email_qa(question, _search_archive, aux_lm),
                                                  ensure_ascii=False)),
        cardinal_spec("daily_digest", "Summarize the archive email for one UTC date (YYYY-MM-DD).",
                      lambda date: write_digest(data.emails_on(date), aux_lm, context_budget=config.context_budget)),
        cardinal_spec(
            "search_docs",
            "Search Cardinal Energy policies, product docs, and announcements by distinctive keywords. "
            "Returns excerpts with doc IDs and line pointers. Check effective dates and later amendments.",
            _search_docs,
        ),
        cardinal_spec(
            "read_doc",
            "Read company document lines from a search pointer. Continue with end line + 1 "
            "only if you need more evidence.",
            _read_doc,
        ),
        terminal_tool,
        *(tool for tool, _ in extras),
    ]
