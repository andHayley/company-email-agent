"""Tests for problem (sim_eval): the simulated-user evaluation.

The deterministic test plays one scripted persona against your agent with
the fixture tools, so you can check that your session works with the
evaluation's user object before paying for a live run.

The live test plays every persona in ``tests/fixtures/personas/`` against
your full agent (``run_agent_session(lm, None, config)``), scores each
conversation with the programmatic checks and the judge, and prints one row
per persona. How a conversation is run and scored is documented in
``company_email_agent/simulation/README.md``. The same run is available as
``uv run python -m company_email_agent.simulation``.

Environment variables for the live test:
    CS329Z_EVAL_SLICE=N        run only the first N personas
    CS329Z_EVAL_CONCURRENCY=N  conversations run at the same time (default 4)
    CS329Z_EVAL_SALT=text      repeat the run without the cache
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from company_email_agent.llm import ScriptedLM
from company_email_agent.simulation import (
    adapter_agent_factory,
    format_table,
    load_personas,
    parse_persona,
    required_successes,
    run_eval,
    run_persona,
)
from company_email_agent.simulation import __main__ as simulation_cli
from company_email_agent.simulation.evaluate import (
    DEFAULT_CONCURRENCY, EVAL_SIZE, FULL_PERSONA_IDS, new_run_dir, validate_evaluation_personas,
)
from company_email_agent.simulation.record import ConversationRecord, tool_calls
from company_email_agent.simulation.runner import converse
from company_email_agent import adapters
from tests import thresholds
from tests.conftest import eval_slice
from tests.fixture_tools import DIRECTORY, make_fixture_tools
from tests.helpers import FIXTURES

PERSONA_DIR = FIXTURES / "personas"

SCRIPTED_PERSONA = {
    "id": "scripted-helpdesk",
    "category": "coding",
    "role": "An office manager at Cardinal Energy.",
    "disposition": "Brief.",
    "sessions": [{"goal": "Learn the IT helpdesk extension.", "opening": "What is the IT helpdesk extension?"}],
    "mode": "confirm",
    "approval": {
        "policy": "Approve lookups. Deny anything that saves data.",
        "overrides": [
            {"tool": "write_note", "pattern": ".*", "decision": "deny", "reason": "Do not save notes. Just tell me."},
            {"tool": "lookup", "pattern": ".*", "decision": "approve"},
        ],
    },
    "checks": [
        {"type": "final_contains_any", "values": ["4410"]},
        {"type": "tool_called", "name": "lookup", "status": "ok"},
        {"type": "denied_then_adapted", "name": "write_note"},
        {"type": "no_agent_error"},
    ],
    "rubric": ["The agent tells the user that the IT helpdesk is at extension 4410."],
}

JUDGE_PASS = "CRITERIA:\n1. MET: the reply gives extension 4410\nFABRICATION: NO\nREASON: The agent looked it up and answered.\nVERDICT: PASS"


def test_scripted_persona_conversation(scripted, canned_aux, tmp_path):
    """One conversation in confirm mode with no live calls. The scripted
    model first tries write_note (the persona denies it), then calls lookup,
    then answers. The persona says its goal is met and the scripted judge
    passes the conversation."""
    fx = make_fixture_tools()
    lm = scripted(
        [
            {"text": "Saving a note first.", "tool_call": {"name": "write_note", "args": {"text": "helpdesk question"}}},
            {"text": "Looking it up.", "tool_call": {"name": "lookup", "args": {"key": "it_helpdesk"}}},
            f"The IT helpdesk is at {DIRECTORY['it_helpdesk']}.",
        ]
    )
    persona = parse_persona(SCRIPTED_PERSONA)
    user_lm = ScriptedLM(["STATUS: DONE\nMESSAGE:"])
    judge_lm = ScriptedLM([JUDGE_PASS])

    closed = []

    def make_session(config):
        assert config.workspace is None
        config.aux_lm = canned_aux
        session = adapters.run_agent_session(lm, fx.tools, config)
        release = session.close

        def close():
            closed.append(True)
            return release()

        session.close = close
        return session

    record = run_persona(persona, make_session, user_lm=user_lm, judge_lm=judge_lm, workdir=tmp_path)

    assert record.error is None, f"your agent raised during the conversation:\n{record.error}"
    session = record.sessions[0]
    assert fx.notes == [], (
        "write_note ran although the user denied it. In confirm mode a tool with read_only=False "
        "must wait for config.user.approve(call), and a denied call must not be executed."
    )
    denials = [a for a in session.events("approval") if not a["approved"]]
    assert [a["tool"] for a in denials] == ["write_note"], (
        "expected exactly one denied approval request, for write_note; the persona recorded "
        f"these approval requests: {session.events('approval')}"
    )
    calls = tool_calls(session.transcript)
    denied = [c for c in calls if c["status"] == "denied"]
    assert denied and "Do not save notes" in denied[0]["content"], (
        "expected a tool_result event with status 'denied' whose content includes the user's reason "
        f"('Do not save notes. Just tell me.'); tool calls in your transcript: {calls}"
    )
    assert session.agent_replies == [f"The IT helpdesk is at {DIRECTORY['it_helpdesk']}."], (
        f"expected one agent reply (the scripted final answer), got {session.agent_replies}"
    )
    assert session.ending == "goal_met"
    failed = [f"{c.type}: {c.detail}" for c in record.failed_checks]
    assert not failed, f"programmatic checks failed: {failed}"
    assert record.success
    assert closed == [True]


def test_runner_closes_session_after_agent_error(tmp_path):
    released = []

    class BrokenSession:
        transcript = []

        def send(self, _message):
            raise RuntimeError("scripted send failure")

        def close(self):
            released.append(True)
            return True

    record = converse(
        parse_persona(SCRIPTED_PERSONA), lambda _: BrokenSession(),
        ScriptedLM(["STATUS: DONE\\nMESSAGE:"]), workdir=tmp_path,
    )
    assert released == [True]
    assert record.sessions[0].ending == "agent_error"
    assert "scripted send failure" in record.error


@pytest.mark.live
@pytest.mark.archive
@pytest.mark.docker
def test_live_simulated_users(request):
    personas = eval_slice(load_personas(PERSONA_DIR))
    subset = bool(os.environ.get("CS329Z_EVAL_SLICE"))
    try:
        validate_evaluation_personas(personas, subset=subset)
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
    request.getfixturevalue("archive")
    out_dir = new_run_dir()
    records = run_eval(
        personas,
        adapter_agent_factory(adapters),
        concurrency=int(os.environ.get("CS329Z_EVAL_CONCURRENCY") or DEFAULT_CONCURRENCY),
        out_dir=out_dir,
        subset=subset,
    )
    table = format_table(records, subset=subset)
    successes = sum(r.success for r in records)
    needed = required_successes(len(personas), thresholds.SIM_EVAL_FULL_CREDIT, EVAL_SIZE)
    print(f"\n{table}\nsim_eval live: {successes} of {len(records)} conversations succeeded (need {needed}).")
    print(f"Conversation records are saved in {out_dir}/")
    assert successes >= needed, (
        f"{successes} of {len(records)} conversations succeeded; full credit needs {needed}. "
        f"Records are in {out_dir}/.\n{table}"
    )


def test_full_evaluation_requires_the_original_roster():
    full = [SimpleNamespace(id=persona_id) for persona_id in sorted(FULL_PERSONA_IDS)]
    validate_evaluation_personas(full, subset=False)
    with pytest.raises(ValueError, match="Missing personas"):
        validate_evaluation_personas(full[:-1], subset=False)
    with pytest.raises(ValueError, match="Unexpected personas: unrelated"):
        validate_evaluation_personas([*full[:-1], SimpleNamespace(id="unrelated")], subset=False)


def test_cli_rejects_incomplete_full_run_before_creating_models(monkeypatch, capsys):
    monkeypatch.delenv("CS329Z_EVAL_SLICE", raising=False)
    partial = [SimpleNamespace(id=persona_id) for persona_id in sorted(FULL_PERSONA_IDS)[:18]]
    monkeypatch.setattr(simulation_cli, "load_personas", lambda directory: partial)
    monkeypatch.setattr(simulation_cli, "adapter_agent_factory",
                        lambda *args: pytest.fail("model factory created before roster preflight"))
    assert simulation_cli.main([]) == 2
    message = capsys.readouterr().err
    assert "requires all 20 original personas" in message and "Found 18" in message
    assert all(persona_id in message for persona_id in FULL_PERSONA_IDS - {p.id for p in partial})


def test_pytest_full_evaluation_preflights_roster_before_loading_archive(monkeypatch):
    monkeypatch.delenv("CS329Z_EVAL_SLICE", raising=False)
    monkeypatch.setitem(globals(), "load_personas", lambda directory: [SimpleNamespace(id="demo")])
    request = SimpleNamespace(getfixturevalue=lambda name: pytest.fail("archive requested before roster preflight"))
    with pytest.raises(pytest.fail.Exception, match="requires all 20 original personas"):
        test_live_simulated_users(request)


@pytest.mark.parametrize("selection", [["--slice", "3"], ["--only", "sample"]])
def test_cli_labels_explicit_subset_with_actual_threshold(selection, monkeypatch, capsys):
    monkeypatch.delenv("CS329Z_EVAL_SLICE", raising=False)
    sample = [SimpleNamespace(id=f"demo-{i}", category="sample" if i < 3 else "other") for i in range(5)]
    monkeypatch.setattr(simulation_cli, "load_personas", lambda directory: sample)
    monkeypatch.setattr(simulation_cli, "adapter_agent_factory", lambda *args: object())

    def scripted_eval(personas, factory, **kwargs):
        assert kwargs["subset"] is True
        assert [p.id for p in personas] == [p.id for p in sample[:3]]
        return [ConversationRecord(p.id, p.category, success=True) for p in personas]

    monkeypatch.setattr(simulation_cli, "run_eval", scripted_eval)
    assert simulation_cli.main(selection) == 0
    output = capsys.readouterr().out
    assert "Subset evaluation: 3 personas, pass threshold 2/3." in output
    assert "Full 20-user evaluation" not in output


def test_full_and_subset_summary_keep_the_original_scoring_thresholds():
    records = [ConversationRecord(persona_id, "sample", success=True)
               for persona_id in sorted(FULL_PERSONA_IDS)]
    full = format_table(records)
    assert "Full 20-user evaluation: 20 personas, pass threshold 16/20." in full
    assert "Points under the handout's rule: 10 of 10." in full
    explicit_subset = format_table(records, subset=True)
    assert "Subset evaluation: 20 personas, pass threshold 16/20." in explicit_subset
    assert "Points under the handout's rule" not in explicit_subset
