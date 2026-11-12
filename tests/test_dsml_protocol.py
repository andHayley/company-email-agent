"""Protocol regressions with synthetic inputs and opt-in original replay."""

import pytest

from company_email_agent import adapters
from company_email_agent.llm import ScriptedLM
from company_email_agent.types import AgentConfig, ToolCall


_QUERY = "GridLens Ingestion API Specification"
_SEARCH = ToolCall("search_docs", {"query": _QUERY, "k": 5})


def test_dsml_json_call_parses_live_response(dsml_trace):
    parsed = adapters.run_parse_response(dsml_trace["json_block"])

    assert parsed.error is None
    assert parsed.tool_call == _SEARCH
    assert parsed.text == ""


def test_fenced_json_call_still_round_trips():
    reply = adapters.run_format_tool_call("Checking the specification.", _SEARCH)
    parsed = adapters.run_parse_response(reply)

    assert parsed.error is None
    assert parsed.tool_call == _SEARCH
    assert parsed.text == "Checking the specification."


def test_dsml_search_docs_dispatches_to_the_existing_tool(dsml_trace):
    lm = ScriptedLM([dsml_trace["json_block"], "The lookup is complete."])
    session = adapters.run_agent_session(lm, None, AgentConfig())
    result = session.send("Look up the GridLens Ingestion API Specification.")

    assert result.status == "done" and result.text == "The lookup is complete."
    calls = [event for event in result.transcript if event["type"] == "tool_call"]
    assert [(event["name"], event["args"]) for event in calls] == [(_SEARCH.name, _SEARCH.args)]
    observation = next(event for event in result.transcript if event["type"] == "tool_result")
    assert observation["status"] == "ok"
    assert any("doc_id: gridlens-api-spec" in m["content"] for m in lm.calls[1])


def test_recorded_recall_then_dsml_search_reaches_next_model_turn(tmp_path, dsml_trace):
    config = AgentConfig(memory_dir=tmp_path)
    first_lm = ScriptedLM([dsml_trace["remember"], dsml_trace["stored_acknowledgement"]])
    first = adapters.run_agent_session(first_lm, None, config)
    first.send(dsml_trace["first_message"])
    assert any(
        event["type"] == "tool_result" and event["name"] == "remember" and event["status"] == "ok"
        for event in first.transcript
    )

    # The final turn is scripted in both modes. Historical inputs are loaded unchanged.
    final = "The next model turn received the document search result."
    lm = ScriptedLM([dsml_trace["recall"], dsml_trace["native_calls"], dsml_trace["json_block"], final])
    second = adapters.run_agent_session(lm, None, config)
    result = second.send(dsml_trace["second_message"])

    assert result.status == "done" and result.text == final
    assert [event["name"] for event in result.transcript if event["type"] == "tool_call"] == [
        "recall", "search_docs",
    ]
    assert [event["status"] for event in result.transcript if event["type"] == "tool_result"] == ["ok", "ok"]
    assert [event["type"] for event in result.transcript].count("error") == 1
    assert len(lm.calls) == 4
    assert any("doc_id: gridlens-api-spec" in m["content"] for m in lm.calls[-1])


_TRACE_NEGATIVE_CASES = (
    "recorded-unsupported-native-wrapper", "missing-close", "invalid-json",
    "duplicate-json-key", "unknown-before-valid", "unknown-after-valid",
)


@pytest.mark.parametrize(("case", "trace_mode"), [
    *(pytest.param(case, "synthetic", id=case) for case in (
        *_TRACE_NEGATIVE_CASES, "unknown-marker-after-text", "orphan-close",
    )),
    *(pytest.param(case, "historical", id=f"{case}-historical", marks=pytest.mark.historical_fixture)
      for case in _TRACE_NEGATIVE_CASES),
], indirect=["trace_mode"])
def test_invalid_dsml_is_a_protocol_error(case, dsml_trace):
    block = dsml_trace["json_block"]
    native = dsml_trace["native_calls"]
    reply = {
        "recorded-unsupported-native-wrapper": native,
        "missing-close": block.rsplit("\n", 1)[0],
        "invalid-json": block.replace('"k":5', '"k":'),
        "duplicate-json-key": block.replace('"k":5', '"k":5,"k":10'),
        "unknown-marker-after-text": "Looking up the doc.\n<｜｜DSML｜｜unknown_call>",
        "orphan-close": "</｜｜DSML｜｜tool_call>",
        "unknown-before-valid": native + "\n" + block,
        "unknown-after-valid": block + "\n" + native,
    }[case]
    parsed = adapters.run_parse_response(reply)

    assert parsed.error is not None
    assert parsed.tool_call is None


@pytest.mark.parametrize("dsml_first", [True, False])
def test_mixed_call_formats_still_allow_only_one_call(dsml_first, dsml_trace):
    fenced = adapters.run_format_tool_call("", _SEARCH)
    dsml = dsml_trace["json_block"]
    reply = (dsml + "\n" + fenced) if dsml_first else (fenced + "\n" + dsml)
    parsed = adapters.run_parse_response(reply)

    assert parsed.error is not None and parsed.tool_call is None


def test_dsml_marker_in_json_argument_is_data():
    value = "<｜｜DSML｜｜tool_call>"
    original = ToolCall("echo", {"text": value})
    parsed = adapters.run_parse_response(adapters.run_format_tool_call("", original))

    assert parsed.error is None and parsed.tool_call == original


def test_dsml_parameters_still_require_the_existing_schema(dsml_trace):
    malformed_args = dsml_trace["json_block"].replace('"k":5', '"k":"five"')
    lm = ScriptedLM([malformed_args, "The search argument was invalid."])
    session = adapters.run_agent_session(lm, None, AgentConfig())
    result = session.send("Look up the specification.")

    observations = [event for event in result.transcript if event["type"] == "tool_result"]
    assert len(observations) == 1 and observations[0]["status"] == "invalid_args"
    assert observations[0]["name"] == "search_docs"
    assert "integer" in observations[0]["content"]
