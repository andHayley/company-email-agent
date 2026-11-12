"""Regression tests for recovering from observed model replies."""

import json

import pytest

from company_email_agent import adapters
from company_email_agent.llm import ScriptedLM
from company_email_agent.types import AgentConfig


@pytest.mark.parametrize("recovery_query", ["northstar delivery", None, "vendor delivers cedar order deliver"])
def test_email_qa_reconsiders_first_pass_abstention(recovery_query):
    def email(email_id, body):
        return {
            "id": email_id, "date": "2001-07-01", "to": [],
            "subject": "Order update", "body": body,
        }

    queries = []

    def search(query, k):
        queries.append(query)
        if query == "northstar delivery":
            return [email("em-2", "Northstar will deliver on Friday.")]
        return [email("em-1", "Northstar is handling the Cedar order.")]

    lm = ScriptedLM([
        '{"answer": null, "support": []}',
        json.dumps({"query": recovery_query}),
        '{"answer": "Friday", "support": ["em-1", "em-2"]}',
    ])
    result = adapters.run_email_qa("When does the vendor that delivers the Cedar order deliver?", search, lm)

    assert queries[0] == "vendor delivers cedar order deliver"
    if recovery_query == "northstar delivery":
        assert queries[1:] == ["northstar delivery"]
        assert result == {"answer": "Friday", "support": ["em-1", "em-2"]}
    else:
        assert len(queries) == 1
        assert result == {"answer": None, "support": []}


def test_session_recovers_from_native_tool_call_markup(scripted):
    native_call = '<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="echo">'
    session = adapters.run_agent_session(
        scripted([native_call, "I can answer directly."]),
        [],
        AgentConfig(),
    )

    result = session.send("Can you answer this directly?")

    assert result.text == "I can answer directly."
    assert not any(event["type"] == "tool_call" for event in result.transcript)
    assert any(
        event["type"] == "error" and "fenced tool_call block" in event["content"]
        for event in result.transcript
    )

_NO_EVIDENCE = {"answer": None, "support": []}
_FIRST_QUERY = "will buyer cedar deal take delivery"
_QUESTION = "Where will the buyer in the Cedar deal take delivery?"
_BUYER_QUERY = "Northstar Utilities delivery location"


def _archive_email(email_id, thread, body):
    return {
        "id": email_id, "thread_id": thread, "reply_to": None, "date": "2001-07-01",
        "from": "staff@example.test", "to": [], "subject": "Order update", "body": body,
    }


def _run_email_recovery(replies, search_results, question=_QUESTION):
    searches = []
    prompts = []
    scripted = ScriptedLM([json.dumps(reply) for reply in replies])

    def search(query, k):
        searches.append(query)
        assert k == 10
        return search_results.get(query, [])

    def lm(messages):
        prompts.append(json.loads(messages[-1]["content"]))
        return scripted(messages)

    result = adapters.run_email_qa(question, search, lm)
    return result, searches, prompts


@pytest.mark.parametrize(("clue", "second_id"), [
    ("Yevgenia Balestrieri", "em-11801"),
    ("Amira Nwosu", "em-contact"),
    ("FAC-5518", "em-reference"),
    ("Kestrow-Dray Services", "em-organization"),
])
def test_email_qa_recovers_across_threads_after_second_search(clue, second_id):
    first = _archive_email("em-internal", "th-internal", f"Northstar Utilities lists {clue} for Cedar.")
    second = _archive_email("em-unrelated", "th-other", "Northstar Utilities signed a contract.")
    target = _archive_email(second_id, "th-external", f"{clue}: We need the shipment at West Gate.")
    answer = {"answer": "West Gate", "support": [first["id"], target["id"]]}

    result, searches, prompts = _run_email_recovery(
        [{"query": _BUYER_QUERY}, _NO_EVIDENCE, {"query": clue}, answer],
        {_FIRST_QUERY: [first], _BUYER_QUERY: [second], clue: [target]},
    )

    assert searches == [_FIRST_QUERY, _BUYER_QUERY, clue]
    assert [email["id"] for email in prompts[2]["emails"]] == [first["id"]]
    assert clue in prompts[2]["emails"][0]["body"]
    assert [email["id"] for email in prompts[3]["emails"]] == [first["id"], second["id"], target["id"]]
    assert result == answer


def test_email_qa_skips_alternate_search_when_second_search_is_sufficient():
    first = _archive_email("em-internal", "th-internal", "The contact is Amira Nwosu.")
    second = _archive_email("em-answer", "th-other", "Please deliver at West Gate.")
    answer = {"answer": "West Gate", "support": [second["id"]]}

    result, searches, prompts = _run_email_recovery(
        [{"query": _BUYER_QUERY}, answer], {_FIRST_QUERY: [first], _BUYER_QUERY: [second]},
    )

    assert result == answer
    assert searches == [_FIRST_QUERY, _BUYER_QUERY]
    assert len(prompts) == 2


def test_email_qa_skips_alternate_search_after_budget_is_exhausted():
    first = _archive_email("em-internal", "th-internal", "The contact is Amira Nwosu.")

    result, searches, prompts = _run_email_recovery(
        [{"query": _BUYER_QUERY}, {"query": "cedar shipment location"}, _NO_EVIDENCE],
        {_FIRST_QUERY: [first]},
    )

    assert result == _NO_EVIDENCE
    assert searches == [_FIRST_QUERY, _BUYER_QUERY, "cedar shipment location"]
    assert [p["remaining_searches"] for p in prompts] == [2, 1, 0]


@pytest.mark.parametrize("candidate", ["delivery", "Delivery Status", "Amira Nwosu", None])
def test_email_qa_rejects_generic_or_unseen_alternate_clues(candidate):
    first = _archive_email("em-internal", "th-internal", "Delivery Status: we are waiting for details.")
    second = _archive_email("em-other", "th-other", "Amira Nwosu will reply.")

    result, searches, _ = _run_email_recovery(
        [{"query": _BUYER_QUERY}, _NO_EVIDENCE, {"query": candidate}],
        {_FIRST_QUERY: [first], _BUYER_QUERY: [second]},
    )

    assert result == _NO_EVIDENCE
    assert searches == [_FIRST_QUERY, _BUYER_QUERY]


def test_email_qa_rejects_an_entity_already_used_in_a_larger_query():
    first = _archive_email("em-internal", "th-internal", "The contact is Amira Nwosu.")
    previous = "Amira Nwosu delivery location"

    result, searches, _ = _run_email_recovery(
        [{"query": previous}, _NO_EVIDENCE, {"query": "amira nwosu"}], {_FIRST_QUERY: [first]},
    )

    assert result == _NO_EVIDENCE
    assert searches == [_FIRST_QUERY, previous]


def test_email_qa_single_hop_returns_without_an_extra_call():
    first = _archive_email("em-answer", "th-first", "The shipment goes to West Gate.")
    answer = {"answer": "West Gate", "support": [first["id"]]}

    result, searches, prompts = _run_email_recovery([answer], {_FIRST_QUERY: [first]})

    assert result == answer
    assert searches == [_FIRST_QUERY]
    assert len(prompts) == 1


@pytest.mark.parametrize("final_reply", [_NO_EVIDENCE, {"answer": "West Gate", "support": ["em-absent"]}])
def test_email_qa_name_match_is_not_evidence_of_the_answer(final_reply):
    clue = "Amira Nwosu"
    first = _archive_email("em-internal", "th-internal", f"The contact is {clue}.")
    unrelated = _archive_email("em-unrelated", "th-other", f"{clue} rescheduled a staff meeting.")

    result, searches, prompts = _run_email_recovery(
        [{"query": _BUYER_QUERY}, _NO_EVIDENCE, {"query": clue}, final_reply],
        {_FIRST_QUERY: [first], clue: [unrelated]},
    )

    assert result == _NO_EVIDENCE
    assert searches == [_FIRST_QUERY, _BUYER_QUERY, clue]
    assert [m["id"] for m in prompts[-1]["emails"]] == [first["id"], unrelated["id"]]


@pytest.mark.parametrize("reply", [
    {"answer": "", "support": []},
    {"answer": None},
    {"answer": None, "support": ["em-unknown"]},
])
def test_email_qa_alternate_search_requires_explicit_abstention(reply):
    first = _archive_email("em-internal", "th-internal", "The contact is Amira Nwosu.")

    result, searches, prompts = _run_email_recovery(
        [{"query": _BUYER_QUERY}, reply], {_FIRST_QUERY: [first]},
    )

    assert result == _NO_EVIDENCE
    assert searches == [_FIRST_QUERY, _BUYER_QUERY]
    assert len(prompts) == 2


def test_email_qa_alternate_search_requires_first_search_evidence():
    second = _archive_email("em-other", "th-other", "The contact is Amira Nwosu.")

    result, searches, prompts = _run_email_recovery(
        [{"query": _BUYER_QUERY}, _NO_EVIDENCE], {_BUYER_QUERY: [second]},
    )

    assert result == _NO_EVIDENCE
    assert searches == [_FIRST_QUERY, _BUYER_QUERY]
    assert len(prompts) == 2
