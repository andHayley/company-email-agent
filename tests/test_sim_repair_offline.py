"""Offline regressions for the failed simulated evaluation paths."""

import json

import pytest

from company_email_agent import adapters, data
from company_email_agent.agent.memory import MemoryStore
from company_email_agent.agent.protocol import parse_response
from company_email_agent.agent.tools import _assistant_docs, _read_doc, _search_docs
from company_email_agent.llm import ScriptedLM
from company_email_agent.tokens import count_message_tokens
from company_email_agent.types import AgentConfig, ToolCall
from tests.helpers import load_fixture


def call(name, **args):
    return adapters.run_format_tool_call("", ToolCall(name, args))


def tools_called(result):
    return [event["name"] for event in result.transcript if event["type"] == "tool_call"]


def test_saved_conditional_alias_before_premature_clarification(tmp_path):
    fact = 'User is on the Platform team. When they say "the spec", they always mean the GridLens API spec.'
    config = AgentConfig(memory_dir=tmp_path)
    first = adapters.run_agent_session(ScriptedLM([call("remember", fact=fact), "Saved."]), None, config)
    assert tools_called(first.send("Remember my preferred meaning of the spec.")) == ["remember"]

    prompt = "What's the document number of the spec, and what date is on it?"
    premature = "I need a bit more context to find the right spec. Which specification are you referring to?"
    lm = ScriptedLM([
        premature, call("recall", query="the spec"),
        call("search_docs", query="GridLens API specification"), "API-SPEC-002, 2026-02-20.",
    ])
    result = adapters.run_agent_session(lm, None, config).send(prompt)

    assert tools_called(result) == ["recall", "search_docs"]
    assert sum(e["type"] == "routing_hint" for e in result.transcript) == 1
    assert "earlier conversation" in lm.calls[1][0]["content"]
    assert not any(fact in message["content"] for message in lm.calls[1])
    assert any(fact in message["content"] for message in lm.calls[2])
    assert any("API-SPEC-002" in message["content"] and "2026-02-20" in message["content"]
               for message in lm.calls[3])


@pytest.mark.parametrize("org_details", [True, False])
def test_recalled_team_continues_only_when_company_details_are_needed(tmp_path, org_details):
    MemoryStore(tmp_path).remember("User is on the Customer Success team, based in Rotterdam.")
    question = "Who manages my team, and how many people are on it?" if org_details else "What team am I on?"
    after_recall = "I don't have a saved record of who manages that team." if org_details else "Customer Success."
    replies = [call("recall", query="my team"), after_recall]
    if org_details:
        replies.extend([
            call("search_docs", query="Customer Success team manager Rotterdam org chart"),
            call("read_doc", doc_id="org-chart", start_line=55),
            "Hannah Kowalski manages the team of 15.",
        ])
    lm = ScriptedLM(replies)
    result = adapters.run_agent_session(lm, None, AgentConfig(memory_dir=tmp_path)).send(question)

    assert tools_called(result) == (["recall", "search_docs", "read_doc"] if org_details else ["recall"])
    assert sum(e["type"] == "routing_hint" for e in result.transcript) == int(org_details)
    if org_details:
        assert "organizational details" in lm.calls[2][0]["content"]
        assert any("Hannah Kowalski" in m["content"] and "15" in m["content"] for m in lm.calls[-1])


def test_direct_lookup_after_recall_does_not_receive_a_second_nudge(tmp_path):
    MemoryStore(tmp_path).remember("User is on the Customer Success team, based in Rotterdam.")
    lm = ScriptedLM([
        call("recall", query="my team"),
        call("search_docs", query="Customer Success team manager Rotterdam org chart"),
        "I found the manager in the org chart.",
    ])
    result = adapters.run_agent_session(lm, None, AgentConfig(memory_dir=tmp_path)).send("Who manages my team?")
    assert tools_called(result) == ["recall", "search_docs"]
    assert not any(e["type"] == "routing_hint" for e in result.transcript)


def test_only_the_current_labeled_number_is_recalled(tmp_path):
    store = MemoryStore(tmp_path)
    store.remember("Badge number: 98311")
    store.remember("Badge office: Rotterdam")
    store.remember("badge NUMBER: 98391")

    found = store.recall("badge")
    assert "98391" in found and "98311" not in found
    assert "Rotterdam" in found
    assert "98311" not in store.recall("98311")
    assert "Badge number: 98311" in list(store.facts())


@pytest.mark.parametrize("variant", ["spaced-json", "typed-arguments", "typed-envelope"])
def test_recorded_dsml_shapes_dispatch_one_call(variant):
    if variant == "spaced-json":
        reply = '\n<｜｜DSML｜｜ tool_call>\n{"name":"search_docs","args":{"query":"expense approvals"}}\n</｜｜DSML｜｜ tool_call>'
    else:
        args = ('<｜｜DSML｜｜ parameter name="query" string="true">expense approvals</｜｜DSML｜｜ parameter>'
                if variant == "typed-arguments" else
                '<｜｜DSML｜｜ parameter name="args" string="false">{"query":"expense approvals"}</｜｜DSML｜｜ parameter>')
        reply = (
            '\n<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="search_docs">\n'
            + args + '\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>'
        )
    assert parse_response(reply).tool_call == ToolCall("search_docs", {"query": "expense approvals"})
    assert parse_response(reply + reply).error
    assert parse_response(reply + "\n" + call("search_docs", query="extra")).error
    lm = ScriptedLM([reply, "I found the policy."])
    result = adapters.run_agent_session(lm, None, AgentConfig()).send("Look up expense approval policies.")
    assert tools_called(result) == ["search_docs"]


def test_approval_window_includes_the_linked_effective_amendment():
    old_window = _read_doc("expense-policy-2026", 72)
    assert "$1,000" in old_window
    assert "finance-memo-2026-08-approval-thresholds" in old_window
    assert "$750" in old_window and "effective 2026-09-01" in old_window
    assert "Linked amendment" not in _read_doc("expense-policy-2026", 1)
    assert "Linked amendment" not in _read_doc("expense-policy-2025", 1)


def test_employee_policy_evidence_keeps_source_lines_and_excludes_the_automation_note():
    raw = next(doc for doc in data.load_docs() if doc["doc_id"] == "expense-policy-2026")
    view = next(doc for doc in _assistant_docs() if doc["doc_id"] == raw["doc_id"])
    original_lines = raw["text"].splitlines()
    actual_lines = view["text"].splitlines()
    assert len(original_lines) == len(actual_lines)
    changed = [(before, after) for before, after in zip(original_lines, actual_lines) if before != after]
    assert len(changed) == 1 and "automated assistants" in changed[0][0] and not changed[0][1].strip()
    evidence = _read_doc(raw["doc_id"], 34)
    assert "half of the daily amount" in evidence and "up to two drinks" in evidence
    assert changed[0][0] not in evidence
    assert "automated assistants" not in _search_docs("meals while traveling daily limit")


@pytest.mark.archive
def test_june_archive_digest_tool_fits_context_with_all_urgent_items():
    day = next(day for day in load_fixture("digest_days.json") if day["date"] == "2001-06-22")
    archive = data.emails_on(day["date"])
    gold_urgent = {item["id"] for item in day["urgent"]}
    gold_ignore = {item["id"] for item in day["ignore"]}

    def key(email):
        return email.get("from", ""), email["date"], email["subject"]

    labels = {
        key(email): "urgent" if email["id"] in gold_urgent else "ignore" if email["id"] in gold_ignore else "normal"
        for email in archive
    }
    budget = 16_000
    sizes = []
    digest_requests = []

    def aux(messages):
        size = count_message_tokens(messages)
        assert size <= budget
        sizes.append(size)
        payload = json.loads(messages[-1]["content"])
        if "urgent_threads" in payload:
            digest_requests.append(payload)
            return "The digest request fit the budget."
        return json.dumps({"category": labels[key(payload)], "reason": "Offline classification replay."})

    lm = ScriptedLM([call("daily_digest", date=day["date"]), "Digest retrieved."])
    result = adapters.run_agent_session(lm, None, AgentConfig(aux_lm=aux, context_budget=budget)).send(
        "Give me the digest for June 22, 2001."
    )
    assert tools_called(result) == ["daily_digest"]
    assert any(e["type"] == "tool_result" and e["status"] == "ok" for e in result.transcript)
    assert len(sizes) == len(archive) + 1
    assert len(digest_requests) == 1
    actual = {key(item) for thread in digest_requests[0]["urgent_threads"] for item in thread}
    assert actual == {key(email) for email in archive if email["id"] in gold_urgent}
    ignored = {key(email) for email in archive if email["id"] in gold_ignore}
    assert not ignored.intersection(key(item) for item in digest_requests[0]["routine_updates"])


def test_small_digest_budget_keeps_urgent_metadata_and_drops_routine_first():
    from company_email_agent.pipelines.daily_digest import write_digest

    emails = [
        {"id": f"test-{i}", "thread_id": f"thread-{i}", "to": [], "from": "",
         "date": "2026-01-01", "subject": f"subject {i}",
         "body": "Action required. " + ('"\\\n' * 3000) + "Deadline Friday."}
        for i in range(5)
    ]
    budget = 1100
    requests = []

    def aux(messages):
        assert count_message_tokens(messages) <= budget
        payload = json.loads(messages[-1]["content"])
        if "urgent_threads" in payload:
            requests.append(payload)
            return "Digest."
        category = "normal" if payload["subject"] == "subject 4" else "urgent"
        return json.dumps({"category": category, "reason": "Respond by Friday."})

    assert write_digest(emails, aux, context_budget=budget) == "Digest."
    [request] = requests
    assert request["routine_updates"] == []
    assert [thread[0]["subject"] for thread in request["urgent_threads"]] == [f"subject {i}" for i in range(4)]
    assert all("Deadline Friday." in thread[0]["body"] for thread in request["urgent_threads"])
