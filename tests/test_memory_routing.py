"""Cross-session routing regressions, with separate opt-in model evaluation."""

import os

import pytest

from company_email_agent import adapters, data
from company_email_agent.agent import tools
from company_email_agent.llm import LM
from company_email_agent.types import AgentConfig


_FIRMWARE = "firmware-spec-emberos"
_API = "gridlens-api-spec"
_ALIASES = [
    pytest.param(_FIRMWARE, "firmware", id="firmware-alias"),
    pytest.param(_API, "api", id="api-alias"),
]
_ALIAS_QUESTION = "Who authored the spec?"
_DIRECT_QUESTION = "Who authored the GridLens Ingestion API Specification and what version is it?"


@pytest.fixture(autouse=True)
def block_terminal(monkeypatch):
    def unexpected_terminal(*args, **kwargs):
        pytest.fail("Memory routing must not execute a terminal command.")

    monkeypatch.setattr(tools, "terminal_output", unexpected_terminal)


def _doc_metadata(doc_id):
    lines = (data.DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8").splitlines()
    author = next(line.removeprefix("Author: ").split(",", 1)[0] for line in lines if line.startswith("Author: "))
    version = next(line.split(": ", 1)[1] for line in lines if line.startswith(("Revision: ", "Version: ")))
    return lines[0].removeprefix("# "), author, version


def _send(lm, memory_dir, message):
    config = AgentConfig(memory_dir=memory_dir, max_turns=12)
    return adapters.run_agent_session(lm, None, config).send(message)


def _scripted_call(name, **args):
    return {"text": "", "tool_call": {"name": name, "args": args}}


def _calls(result):
    return [event for event in result.transcript if event["type"] == "tool_call"]


def _observation(result, name):
    return next(event for event in result.transcript
                if event["type"] == "tool_result" and event["name"] == name)


def _assert_alias_route(first, second, doc_id, qualifier):
    assert first.status == second.status == "done"
    assert _observation(first, "remember")["status"] == "ok"
    assert _calls(second)[0]["name"] == "recall"
    recalled = _observation(second, "recall")
    assert recalled["status"] == "ok" and qualifier in recalled["content"].lower()
    search = next(event for event in _calls(second) if event["name"] == "search_docs")
    assert qualifier in search["args"]["query"].lower()
    docs = _observation(second, "search_docs")
    assert docs["status"] == "ok" and f"doc_id: {doc_id}," in docs["content"]

    _, author, version = _doc_metadata(doc_id)
    _, other_author, _ = _doc_metadata(_API if doc_id == _FIRMWARE else _FIRMWARE)
    assert author in second.text and other_author not in second.text
    if doc_id == _FIRMWARE:
        assert version in second.text


@pytest.mark.parametrize(("doc_id", "qualifier"), _ALIASES)
def test_scripted_cross_session_alias_uses_stored_definition(scripted, tmp_path, doc_id, qualifier):
    title, author, version = _doc_metadata(doc_id)
    fact = f"The user's shorthand 'the spec' means the {title}."
    first = _send(scripted([_scripted_call("remember", fact=fact), "Saved."]),
                  tmp_path, f"以后我说 the spec，就是指 {title}。")
    lm = scripted([
        _scripted_call("recall", query="the spec"),
        _scripted_call("search_docs", query=f"{title} version author"),
        _scripted_call("read_doc", doc_id=doc_id),
        f"{title}, {version}, was authored by {author}.",
    ])
    second = _send(lm, tmp_path, _ALIAS_QUESTION)

    _assert_alias_route(first, second, doc_id, qualifier)
    assert [event["name"] for event in _calls(second)] == ["recall", "search_docs", "read_doc"]
    assert fact in _observation(second, "recall")["content"]
    assert all(fact not in message["content"] for message in lm.calls[0])
    assert any(fact in message["content"] for message in lm.calls[1])
    original_doc = _observation(second, "read_doc")
    assert original_doc["status"] == "ok" and f"Author: {author}" in original_doc["content"]


def test_scripted_explicit_document_question_needs_no_recall(scripted, tmp_path):
    title, author, version = _doc_metadata(_API)
    lm = scripted([_scripted_call("search_docs", query=f"{title} version author"), f"{version}, by {author}."])
    result = _send(lm, tmp_path, _DIRECT_QUESTION)

    assert result.status == "done"
    assert [event["name"] for event in _calls(result)] == ["search_docs"]
    assert _observation(result, "search_docs")["status"] == "ok"
    assert f"doc_id: {_API}," in _observation(result, "search_docs")["content"]
    assert author in result.text


def test_memory_routing_guidance_is_scoped_to_memory_sessions(scripted, tmp_path):
    enabled, disabled = scripted(["Done."]), scripted(["Done."])
    _send(enabled, tmp_path, _ALIAS_QUESTION)
    _send(disabled, None, _DIRECT_QUESTION)

    assert "call recall before searching documents" in enabled.calls[0][0]["content"]
    assert "call recall before searching documents" not in disabled.calls[0][0]["content"]


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("CS329Z_RUN_MEMORY_ROUTING_LIVE") != "1",
    reason="Autonomous model validation requires explicit opt-in and may incur API costs.",
)
@pytest.mark.parametrize(("doc_id", "qualifier"), [*_ALIASES, pytest.param(_API, None, id="explicit-document")])
def test_live_model_selects_memory_route(tmp_path, doc_id, qualifier):
    """Only an actual uncached model can supply tool-choice evidence for this test."""
    lm = LM(tag="memory-routing/autonomous", cache=False)
    title, author, _ = _doc_metadata(doc_id)
    if qualifier is None:
        result = _send(lm, tmp_path, _DIRECT_QUESTION)
        calls = _calls(result)
        print(f"{lm.model} explicit document: {calls}")
        assert result.status == "done" and calls and calls[0]["name"] == "search_docs"
        assert all(event["name"] != "recall" for event in calls)
        assert f"doc_id: {doc_id}," in _observation(result, "search_docs")["content"]
        assert author in result.text
        return

    first = _send(lm, tmp_path, f"以后我说 the spec，就是指 {title}。")
    second = _send(lm, tmp_path, _ALIAS_QUESTION)
    print(f"{lm.model} {doc_id}: first session={_calls(first)}, second session={_calls(second)}")
    _assert_alias_route(first, second, doc_id, qualifier)
