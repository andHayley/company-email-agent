"""Alias policy regressions with synthetic inputs and opt-in original replay."""

import pytest

from company_email_agent import adapters
from company_email_agent.agent.memory import MemoryStore
from company_email_agent.agent.memory_aliases import needs_alias_recall_hint
from company_email_agent.llm import ScriptedLM
from company_email_agent.tokens import count_message_tokens, count_tokens
from company_email_agent.types import AgentConfig
from tests.helpers import load_fixture


_SYNTHETIC = load_fixture("memory_alias_live_trace.synthetic.json")
_QUESTION = _SYNTHETIC["alias_question"]
_HINT = "Routing hint from the agent:"


def _store(directory, *facts):
    store = MemoryStore(directory)
    for fact in facts:
        store.remember(fact)
    return store


@pytest.mark.parametrize("fact_key", [
    pytest.param("firmware_fact", id="firmware-binding"),
    pytest.param("api_fact", id="api-binding"),
])
def test_recorded_broad_search_gets_one_hint_and_can_still_proceed(tmp_path, monkeypatch, fact_key, alias_trace):
    fact = alias_trace[fact_key]
    _store(tmp_path, fact)

    def unexpected_recall(*args):
        pytest.fail("The replay never requests recall; it must not run implicitly.")

    monkeypatch.setattr(MemoryStore, "recall", unexpected_recall)
    lm = ScriptedLM([alias_trace["broad_search"], alias_trace["broad_search"], alias_trace["broad_answer"]])
    result = adapters.run_agent_session(lm, None, AgentConfig(memory_dir=tmp_path)).send(alias_trace["alias_question"])

    assert result.status == "done" and result.text == alias_trace["broad_answer"]
    assert [event["type"] for event in result.transcript].count("routing_hint") == 1
    assert [e["name"] for e in result.transcript if e["type"] == "tool_call"] == ["search_docs"]
    assert [_HINT in messages[0]["content"] for messages in lm.calls] == [False, True, False]
    assert all(fact not in msg["content"] for messages in lm.calls for msg in messages)


@pytest.mark.parametrize(("facts", "question", "query", "expected"), [
    pytest.param(["the spec 指固件规格"], "Who authored “THE SPEC”?", "spec", True, id="chinese-binding"),
    pytest.param(["spec refers to the firmware specification"], "Who authored the spec?", "spec", True, id="plain-binding"),
    pytest.param(["The user's shorthand 'the spec' means the firmware specification."],
                 "请问 the spec 的作者是谁？", "spec author", True, id="user-shorthand-and-chinese-question"),
    pytest.param([_SYNTHETIC["firmware_fact"], 'THE SPEC refers to EmberOS 3 Firmware Specification.'],
                 _QUESTION, "spec", True, id="same-binding-duplicate"),
    pytest.param([], _QUESTION, "spec", False, id="no-memory"),
    pytest.param(["The user works on firmware and reads the spec."], _QUESTION, "spec", False, id="term-overlap-only"),
    pytest.param(["The spec means a lot to the firmware team."], _QUESTION, "spec", False, id="figurative-means"),
    pytest.param([_SYNTHETIC["firmware_fact"], _SYNTHETIC["api_fact"]], _QUESTION, "spec", False, id="conflicting-definitions"),
    pytest.param([_SYNTHETIC["firmware_fact"], "spec refers to the API specification"],
                 _QUESTION, "spec", False, id="overlapping-aliases"),
    pytest.param([_SYNTHETIC["firmware_fact"], "The spec may refer to a different document."],
                 _QUESTION, "spec", False, id="unclear-related-record"),
    pytest.param(["the spec refers to firmware or API"], _QUESTION, "spec", False, id="disjunctive-definition"),
    pytest.param(["the spec refers to firmware and the spec means API"], _QUESTION, "spec", False, id="compound-definition"),
    pytest.param(["spec refers to the firmware specification"], "Who authored specs?", "specs", False, id="no-substring"),
    pytest.param(["spec refers to the firmware specification"], "Who authored the spec_sheet?", "spec", False, id="no-identifier-fragment"),
    pytest.param([_SYNTHETIC["firmware_fact"]], "Who authored the spec in docs/firmware-spec-emberos.md?",
                 "spec", False, id="explicit-path"),
    pytest.param([_SYNTHETIC["firmware_fact"]], "Who authored the spec FW-SPEC-003?",
                 "spec", False, id="explicit-document-number"),
    pytest.param(["Oakland Office Guide refers to the firmware specification"],
                 "Who authored Oakland Office Guide?", "author", False, id="catalog-title-wins"),
    pytest.param([_SYNTHETIC["firmware_fact"]], _QUESTION, "EmberOS firmware spec author", False, id="already-narrowed-query"),
    pytest.param([_SYNTHETIC["firmware_fact"]], "Please remember what I mean by the spec.", "spec",
                 False, id="no-lookup-intent"),
    pytest.param([_SYNTHETIC["firmware_fact"]], "Who authored the employee handbook?", "employee handbook",
                 False, id="unrelated-search"),
])
def test_conservative_alias_matching(tmp_path, facts, question, query, expected):
    store = _store(tmp_path, *facts)
    assert needs_alias_recall_hint(question, query, store.facts()) is expected


def test_recorded_explicit_document_keeps_direct_search(tmp_path, alias_trace):
    _store(tmp_path, "spec refers to the firmware specification")
    lm = ScriptedLM([alias_trace["direct_search"], alias_trace["direct_answer"]])
    result = adapters.run_agent_session(lm, None, AgentConfig(memory_dir=tmp_path)).send(alias_trace["direct_question"])

    assert result.status == "done" and result.text == alias_trace["direct_answer"]
    assert [event["name"] for event in result.transcript if event["type"] == "tool_call"] == ["search_docs"]
    assert not any(event["type"] == "routing_hint" for event in result.transcript)
    assert len(lm.calls) == 2


def test_recorded_recall_then_dsml_search_needs_no_hint(tmp_path, alias_trace, dsml_trace):
    _store(tmp_path, alias_trace["api_fact"])
    lm = ScriptedLM([dsml_trace["recall"], dsml_trace["native_calls"], dsml_trace["json_block"]])
    config = AgentConfig(memory_dir=tmp_path, max_turns=3)
    result = adapters.run_agent_session(lm, None, config).send(alias_trace["alias_question"])

    assert [event["name"] for event in result.transcript if event["type"] == "tool_call"] == ["recall", "search_docs"]
    assert [event["status"] for event in result.transcript if event["type"] == "tool_result"] == ["ok", "ok"]
    assert not any(event["type"] == "routing_hint" for event in result.transcript)


def test_last_model_turn_is_not_spent_deferring_a_search(tmp_path, alias_trace):
    _store(tmp_path, alias_trace["firmware_fact"])
    lm = ScriptedLM([alias_trace["broad_search"]])
    config = AgentConfig(memory_dir=tmp_path, max_turns=1)
    result = adapters.run_agent_session(lm, None, config).send(alias_trace["alias_question"])

    assert result.status == "max_turns"
    assert [event["name"] for event in result.transcript if event["type"] == "tool_call"] == ["search_docs"]
    assert not any(event["type"] == "routing_hint" for event in result.transcript)


def test_thin_token_budget_leaves_the_original_search_in_place(tmp_path, alias_trace):
    _store(tmp_path, alias_trace["firmware_fact"])
    probe = ScriptedLM([alias_trace["broad_search"]])
    adapters.run_agent_session(probe, None, AgentConfig(memory_dir=tmp_path, max_turns=1)).send(alias_trace["alias_question"])
    budget = count_message_tokens(probe.calls[0]) + count_tokens(alias_trace["broad_search"]) + 1

    lm = ScriptedLM([alias_trace["broad_search"], alias_trace["broad_answer"]])
    result = adapters.run_agent_session(
        lm, None, AgentConfig(memory_dir=tmp_path, max_turns=2, token_budget=budget),
    ).send(alias_trace["alias_question"])

    assert [event["name"] for event in result.transcript if event["type"] == "tool_call"] == ["search_docs"]
    assert not any(event["type"] == "routing_hint" for event in result.transcript)
    assert result.text == alias_trace["broad_answer"]
