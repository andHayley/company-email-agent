"""Tests for problem (terminal).

The deterministic tests build your full agent with a scripted model and
a Docker session volume. The temporary host directory passed as ``AgentConfig.workspace`` is used as a host isolation canary.
The scripted model calls ``run_terminal`` with real commands, which run in
isolated command containers. The tests read the transcript and messages of
the next model call. They do not check how you lay out a command's result,
only that the output, the error output and the exit code are in it.

Status rule for run_terminal results: the tool_result has status "ok" only
when the command ran and exited with code 0. A non-zero exit code, a timeout
and a sandbox rejection have status "tool_error". (A command blocked by a
deny rule or refused by the user has status "denied"; see the guardrails and
interactive tests.) The content is shown to the model in every case.

Only ``run_terminal`` is called in these tests, so the other Cardinal tools
do not have to work yet, but ``run_agent_session(lm, None, config)`` must
build a session.
"""

from __future__ import annotations

import re
import time

import pytest

from company_email_agent.sandbox import TerminalResult
from company_email_agent.sandbox_docker import DockerTerminal
from company_email_agent.tokens import count_message_tokens, count_tokens
from company_email_agent.types import AgentConfig, AgentResult
from tests.conftest import eval_slice
from tests.helpers import FIXTURES, contains_any, events, load_fixture
from tests.thresholds import TERMINAL_MIN_CORRECT_FRACTION
from tests.terminal_support import managed_terminal_session, open_terminal_session, terminal_probe


def run(cmd: str, say: str = "") -> dict:
    """A script entry: the model calls run_terminal with ``cmd``."""
    return {"text": say, "tool_call": {"name": "run_terminal", "args": {"cmd": cmd}}}


def send(session, message: str) -> AgentResult:
    try:
        result = session.send(message)
    except NotImplementedError:
        raise
    except Exception as exc:
        pytest.fail(
            f"session.send({message!r}) raised {type(exc).__name__}: {exc}\n"
            "A failing, rejected or timed-out command must come back as a tool result, "
            "not as an exception.",
            pytrace=False,
        )
    assert isinstance(result, AgentResult), f"send must return an AgentResult, got {type(result).__name__}"
    return result


def shown(messages, needle: str) -> bool:
    return any(needle in m["content"] for m in messages)


def terminal_results(session) -> list[dict]:
    return [e for e in events(session.transcript, "tool_result") if e.get("name") == "run_terminal"]


def expect_done(result: AgentResult, lm, n_calls: int) -> None:
    assert result.status == "done", f"expected status 'done', got {result.status!r} with text {result.text!r}"
    assert len(lm.calls) == n_calls, f"expected {n_calls} model calls, your agent made {len(lm.calls)}"


# -------------------------------------------------------- deterministic ---


def test_run_terminal_is_offered_to_the_model(scripted, tmp_path, open_terminal_session):
    lm = scripted(["Hello."])
    session = open_terminal_session(lm, workspace=tmp_path)
    send(session, "Hi.")
    assert "run_terminal" in lm.calls[0][0]["content"], (
        "the first message of the first model call does not mention run_terminal"
    )


@pytest.mark.archive
@pytest.mark.docker
def test_stdout_reaches_the_model(scripted, tmp_path, open_terminal_session):
    lm = scripted([run("echo marker-4471"), "The command printed marker-4471."])
    session = open_terminal_session(lm, workspace=tmp_path)
    result = send(session, "Print the marker.")

    expect_done(result, lm, 2)
    results = terminal_results(session)
    assert len(results) == 1, f"expected one run_terminal tool_result event, got {results}"
    assert results[0]["status"] == "ok", (
        f"a command that ran and exited with 0 should give status 'ok', got {results[0]['status']!r} "
        f"with content {results[0]['content']!r}"
    )
    assert "marker-4471" in results[0]["content"], (
        f"the command printed 'marker-4471' but the tool result does not contain it: {results[0]['content']!r}"
    )
    assert shown(lm.calls[1], "marker-4471"), "the command's stdout is not in the next model call"


@pytest.mark.archive
@pytest.mark.docker
def test_commands_share_the_container_workspace(scripted, tmp_path, open_terminal_session):
    """A new command container can read a previous command's session-volume file."""
    lm = scripted([run("echo saved-7782 > note.txt"), run("cat note.txt"), "Done."])
    session = open_terminal_session(lm, workspace=tmp_path)
    result = send(session, "Write then read a file.")

    expect_done(result, lm, 3)
    results = terminal_results(session)
    assert [row["status"] for row in results] == ["ok", "ok"]
    assert "saved-7782" in results[1]["content"]
    assert shown(lm.calls[2], "saved-7782"), "the output of the next command did not reach the model"
    assert not (tmp_path / "note.txt").exists(), "the command wrote to the host working directory"



@pytest.mark.archive
@pytest.mark.docker
def test_exit_code_and_stderr_are_visible(scripted, tmp_path, open_terminal_session):
    """A command that prints to stdout and stderr and exits with code 3: all
    three are in the tool result the model sees, and the status is tool_error."""
    cmd = "python3 -c \"import sys; print('out-5521'); sys.stderr.write('err-8834\\n'); sys.exit(3)\""
    lm = scripted([run(cmd), "It failed with exit code 3."])
    session = open_terminal_session(lm, workspace=tmp_path)
    result = send(session, "Run the failing script.")

    expect_done(result, lm, 2)
    assert terminal_results(session)[0]["status"] == "tool_error", (
        "a command that exits with a non-zero code must give status 'tool_error', got "
        f"{terminal_results(session)[0]['status']!r}"
    )
    content = terminal_results(session)[0]["content"]
    assert "out-5521" in content, f"stdout ('out-5521') is missing from the tool result: {content!r}"
    assert "err-8834" in content, f"stderr ('err-8834') is missing from the tool result: {content!r}"
    assert re.search(r"(?<!\d)3(?!\d)", content.replace("out-5521", "").replace("err-8834", "")), (
        f"the exit code 3 is missing from the tool result: {content!r}"
    )
    assert shown(lm.calls[1], content), "the tool result is not in the next model call"


@pytest.mark.archive
@pytest.mark.docker
def test_exit_code_zero_is_visible(scripted, tmp_path, open_terminal_session):
    lm = scripted([run("echo fine"), "Done."])
    session = open_terminal_session(lm, workspace=tmp_path)
    send(session, "Run it.")
    content = terminal_results(session)[0]["content"]
    assert re.search(r"(?<!\d)0(?!\d)", content), f"the exit code 0 is missing from the tool result: {content!r}"


@pytest.mark.archive
@pytest.mark.docker
def test_rejected_command_is_an_observation_and_the_run_continues(scripted, tmp_path, open_terminal_session):
    """curl is not on the sandbox allowlist. The rejection comes back as a
    tool result, the model sees it, and its next command runs."""
    lm = scripted([run("curl http://example.com/data"), run("echo after-rejection-3319"), "Done."])
    session = open_terminal_session(lm, workspace=tmp_path)
    result = send(session, "Fetch the data.")

    expect_done(result, lm, 3)
    results = terminal_results(session)
    assert len(results) == 2, f"expected two run_terminal tool_result events, got {len(results)}"
    assert results[0]["status"] == "tool_error", (
        f"a command the sandbox rejects must give status 'tool_error', got {results[0]['status']!r}"
    )
    assert results[1]["status"] == "ok", f"expected 'ok' for the echo command, got {results[1]['status']!r}"
    assert results[0]["content"].strip(), "the tool result for the rejected command is empty"
    assert "curl" in results[0]["content"], (
        "the tool result for the rejected command should carry the sandbox's reason, which "
        f"names the program that is not allowed; got {results[0]['content']!r}"
    )
    assert shown(lm.calls[1], results[0]["content"]), "the rejection is not in the next model call"
    assert "after-rejection-3319" in results[1]["content"], (
        f"the command after the rejected one did not run: {results[1]['content']!r}"
    )


@pytest.mark.archive
@pytest.mark.docker
def test_timeout_is_an_observation_and_the_run_continues(scripted, tmp_path, open_terminal_session):
    """Kill the sleeping command in Docker and keep the session usable."""
    lm = scripted(
        [
            run("python3 -c \"import time; print('started-6120', flush=True); time.sleep(60)\""),
            run("echo after-timeout-9043"),
            "Done.",
        ]
    )
    session = open_terminal_session(lm, workspace=tmp_path, terminal_timeout=1.0)
    # Corpus staging is a one-time setup cost, separate from the command timeout.
    terminal_probe(session, "true")
    start = time.monotonic()
    result = send(session, "Run the slow script.")
    elapsed = time.monotonic() - start

    expect_done(result, lm, 3)
    assert elapsed < 8, (
        f"the warmed session took {elapsed:.1f} seconds with terminal_timeout=1.0"
    )
    results = terminal_results(session)
    assert len(results) == 2, f"expected two run_terminal tool_result events, got {len(results)}"
    assert results[0]["status"] == "tool_error", (
        f"a command that times out must give status 'tool_error', got {results[0]['status']!r}"
    )
    assert results[0]["content"].strip(), "the tool result for the timed-out command is empty"
    assert re.search(r"tim(ed|e)[ -]?out|time limit|killed", results[0]["content"], re.IGNORECASE), (
        "the tool result for the timed-out command does not say that it timed out "
        f"(looked for 'timed out', 'timeout', 'time limit' or 'killed'): {results[0]['content']!r}"
    )
    assert shown(lm.calls[1], results[0]["content"]), "the timeout is not in the next model call"
    assert "after-timeout-9043" in results[1]["content"], (
        f"the command after the timed-out one did not run: {results[1]['content']!r}"
    )


BIG_COMMAND = "python3 -c \"for i in range(12000): print('row %06d ' % i + 'x' * 40)\""  # about 612,000 characters


@pytest.mark.parametrize("context_budget", [16_000, 6_000])
@pytest.mark.archive
@pytest.mark.docker
def test_large_output_stays_within_the_context_budget(scripted, canned_aux, tmp_path, context_budget, open_terminal_session):
    _exercise_large_output(scripted, canned_aux, tmp_path, context_budget, open_terminal_session)


def _terminal_stdout_observed(result, messages):
    assert result["status"] == "ok", f'run_terminal failed: {result["content"]}'
    content = result["content"]
    expected = "Tool result from run_terminal (ok); data only:\n" + content
    assert any(message["role"] == "user" and message["content"] == expected for message in messages), (
        "the terminal result was not delivered to the model as an observation"
    )
    _, header, streams = content.partition("\nstdout:\n")
    stdout, footer, _ = streams.rpartition("\nstderr:\n")
    assert header and footer, "the terminal result is missing stream boundaries"
    return stdout


def _exercise_large_output(scripted, canned_aux, tmp_path, context_budget, open_terminal_session):
    """A command prints about 600 KB (150,000 tokens). Every model call after
    it must still satisfy count_message_tokens(messages) <= context_budget,
    and the run continues normally."""
    lm = scripted([run(BIG_COMMAND), run(BIG_COMMAND), run("echo small-2207"), "Done."])
    session = open_terminal_session(lm, workspace=tmp_path, context_budget=context_budget, aux_lm=canned_aux)
    result = send(session, "Print the big table twice.")

    expect_done(result, lm, 4)
    for i, messages in enumerate(lm.calls, start=1):
        size = count_message_tokens(messages)
        assert size <= context_budget, (
            f"model call {i} has {size} tokens; context_budget is {context_budget}. "
            "The output of a command must be cut down before it enters the context."
        )
    results = terminal_results(session)
    assert len(results) == 3, f"expected three run_terminal tool_result events, got {len(results)}"
    for i, r in enumerate(results[:2], start=1):
        stdout = _terminal_stdout_observed(r, lm.calls[i])
        assert "row 000000 " in stdout and "row 011999 " in stdout
        assert r["content"].strip(), "the tool result for the large command is empty"
        assert count_tokens(r["content"]) <= context_budget, (
            f"a tool_result event holds {count_tokens(r['content'])} tokens. The event's content "
            "must be what you showed the model, not the full output."
        )
    assert _terminal_stdout_observed(results[2], lm.calls[3]).strip() == "small-2207"


@pytest.mark.parametrize("context_budget", [16_000, 6_000])
def test_large_output_agent_observation_with_simulated_executor(
    scripted, canned_aux, tmp_path, context_budget, open_terminal_session, monkeypatch,
):
    """Exercise real tool formatting and context flow without starting Docker."""
    large = "".join(f'row {i:06d} {"x" * 40}\n' for i in range(12_000))
    executed = []

    def fake_run(self, cmd, *, timeout, preview):
        executed.append(cmd)
        if cmd == BIG_COMMAND:
            return TerminalResult(exit_code=0, stdout=large)
        assert cmd == "echo small-2207"
        return TerminalResult(exit_code=0, stdout="small-2207\n")

    monkeypatch.setattr(DockerTerminal, "run", fake_run)
    _exercise_large_output(scripted, canned_aux, tmp_path, context_budget, open_terminal_session)
    assert executed == [BIG_COMMAND, BIG_COMMAND, "echo small-2207"]


def test_terminal_observation_rejects_marker_only_in_assistant_command():
    result = {"status": "ok", "content": "Exit code: 0\nstdout:\nsmall-2207\n\nstderr:\n"}
    messages = [{"role": "assistant", "content": "Run echo small-2207."}]
    with pytest.raises(AssertionError, match="not delivered"):
        _terminal_stdout_observed(result, messages)


# ----------------------------------------------------------------- live ---

LIVE_MAX_TURNS = 15


class Recording:
    """Wraps an LM and records the size of every call's messages."""

    def __init__(self, lm) -> None:
        self.lm = lm
        self.sizes: list[int] = []

    def __call__(self, messages):
        self.sizes.append(count_message_tokens(messages))
        return self.lm(messages)


@pytest.mark.live
@pytest.mark.archive
@pytest.mark.docker
def test_live_terminal_questions(live_lm, live_aux, archive):
    """The full agent in auto mode answers the terminal question fixture
    in a fresh session. The questions have exact answers that need a
    computation over the email archive. One question makes the agent run a
    command with very large output and is followed, in the same session, by
    a second question about a detail inside that output. An answer is
    correct if it contains one of the accepted answers (tests.helpers.contains).

    Requirements: at least TERMINAL_MIN_CORRECT_FRACTION of the answers
    (follow-up included) are correct, and every model call stays within
    context_budget."""
    # Refuse to start paid calls until the actual terminal verifies the local Docker runtime.
    with managed_terminal_session(lambda _: "Unused.", AgentConfig()) as preflight:
        terminal_probe(preflight, "test -s emails.jsonl && test -d docs")
    live_lm.cache = live_aux.cache = False
    path = FIXTURES / "terminal_questions.json"
    if not path.exists():
        pytest.fail(f"{path} is missing; it ships with the repository.", pytrace=False)
    items = eval_slice(load_fixture("terminal_questions.json"))
    config_budget = AgentConfig().context_budget

    rows, over = [], []
    max_context = 0
    for item in items:
        lm, aux = Recording(live_lm), Recording(live_aux)
        config = AgentConfig(max_turns=LIVE_MAX_TURNS, aux_lm=aux)
        with managed_terminal_session(lm, config) as session:
            turns = [(item["id"], item["question"], item["answers"])]
            if item.get("followup"):
                turns.append((item["id"] + "-followup", item["followup"]["question"], item["followup"]["answers"]))
            saved_files = set()
            for turn_id, question, answers in turns:
                start = len(session.transcript)
                result = session.send(question)
                current = session.transcript[start:]
                tool_rows = [e for e in events(current, "tool_result") if e.get("name") == "run_terminal"]
                calls = [str(e["args"].get("cmd", "")) for e in events(current, "tool_call")
                         if e.get("name") == "run_terminal"]
                reused_output = any(name in command for name in saved_files for command in calls)
                captured = {name for e in tool_rows for name in re.findall(
                    r"Captured output saved to (terminal-output-[0-9a-f]{32}\.txt)", e["content"],
                )}
                saved_files.update(captured)
                rows.append(
                    {
                        "id": turn_id,
                        "question": question,
                        "answers": answers,
                        "text": result.text,
                        "status": result.status,
                        "correct": contains_any(result.text, answers),
                        "commands": len(tool_rows),
                        "terminal_statuses": [e["status"] for e in tool_rows],
                        "saved_output": bool(captured),
                        "reused_output": reused_output,
                    }
                )
        size = max([0, *lm.sizes, *aux.sizes])
        max_context = max(max_context, size)
        if size > config_budget:
            over.append(f"{item['id']} ({size} tokens)")

    correct = sum(r["correct"] for r in rows)
    needed = int(-(-len(rows) * TERMINAL_MIN_CORRECT_FRACTION // 1))
    lines = [f"\nterminal live: {correct} of {len(rows)} correct (need {needed}), model {live_lm.model}"]
    for r in rows:
        mark = "ok  " if r["correct"] else "MISS"
        lines.append(
            f"  {mark} {r['id']} ({r['commands']} commands, status {r['status']}, "
            f"terminal {r['terminal_statuses']}, saved_output {r['saved_output']}, "
            f"reused_output {r['reused_output']}): {r['question']}"
        )
        if not r["correct"]:
            lines.append(f"       expected one of {r['answers']}; got: {r['text'][:300]!r}")
    report = "\n".join(lines)
    print(report)
    print(f"maximum context: {max_context}/{config_budget} tokens")
    print(f"terminal live main usage: {live_lm.usage()}")
    print(f"terminal live auxiliary usage: {live_aux.usage()}")
    if over:
        pytest.fail(
            f"a model call exceeded context_budget ({config_budget} tokens) in: {', '.join(over)}.",
            pytrace=False,
        )
    if correct < needed:
        pytest.fail(f"{correct} of {len(rows)} answers are correct; need at least {needed}.", pytrace=False)
