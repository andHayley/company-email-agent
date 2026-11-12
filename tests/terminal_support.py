"""Run fixed test probes through the managed terminal tool."""

from contextlib import ExitStack, contextmanager

import pytest

from company_email_agent import adapters
from company_email_agent.types import AgentConfig, ToolCall


@contextmanager
def managed_terminal_session(lm, config):
    session = adapters.run_agent_session(lm, None, config)
    try:
        yield session
    finally:
        assert session.close(), "Docker terminal session cleanup could not be confirmed"


@pytest.fixture
def open_terminal_session(canned_aux):
    with ExitStack() as stack:
        def create(lm, **config):
            config.setdefault("aux_lm", canned_aux)
            return stack.enter_context(managed_terminal_session(lm, AgentConfig(**config)))

        yield create


def terminal_probe(session, command):
    """Run a fixed setup or observation command with the session's own policy."""
    result = session._run_tool(ToolCall("run_terminal", {"cmd": command}))
    assert result.status == "ok", result.content
    _, start, rest = result.content.partition("\nstdout:\n")
    stdout, end, _ = rest.rpartition("\nstderr:\n")
    assert start and end, result.content
    return stdout


def workspace_names(session):
    return set(terminal_probe(session, "ls -1").splitlines())
