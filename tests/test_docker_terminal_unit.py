"""Docker terminal regressions that never need an Engine or a model."""

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from company_email_agent import adapters, data, sandbox
from company_email_agent.agent import session as session_module
from company_email_agent.sandbox_docker import DockerIsolationError, DockerSettings, DockerTerminal, _make_corpus, _result
from company_email_agent.types import AgentConfig, Approval, ToolCall, ToolSpec
from company_email_agent.user import ScriptedUser


def reply(stdout=b"ok", *, code=0, filename=None):
    return json.dumps({
        "stdout": base64.b64encode(stdout).decode(),
        "stderr": "",
        "exit_code": code,
        "timed_out": False,
        "output_file": filename,
    }).encode()


def test_disallowed_command_never_prepares_docker(monkeypatch):
    terminal = DockerTerminal()
    monkeypatch.setattr(terminal, "_start", lambda: pytest.fail("started Docker"))
    result = terminal.run("curl example.test", timeout=1, preview=512)
    assert result.rejected and "not an allowed program" in result.reason


def test_docker_unavailable_does_not_call_native_runner(monkeypatch):
    terminal = DockerTerminal()

    def unavailable():
        raise DockerIsolationError("Docker daemon is unavailable.")

    monkeypatch.setattr(terminal, "_start", unavailable)
    monkeypatch.setattr(sandbox, "run_terminal", lambda *a, **k: pytest.fail("fell back to host"))
    first = terminal.run("echo probe", timeout=1, preview=512)
    second = terminal.run("echo probe again", timeout=1, preview=512)
    assert first.rejected and first.exit_code is None and "daemon is unavailable" in first.reason
    assert second.rejected and second.exit_code is None


def test_native_runner_without_jail_never_spawns(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "_SHELL", "not-a-real-shell")
    monkeypatch.setattr(sandbox, "jail", lambda: "none")
    monkeypatch.setattr(sandbox, "make_workspace", lambda *a: pytest.fail("created host workspace"))
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned host process"))
    assert sandbox.run_terminal("echo probe", workspace=tmp_path).rejected


def test_new_container_per_command_and_command_only_in_stdin(monkeypatch):
    terminal = DockerTerminal()
    records = []
    creations = []
    removals = []
    monkeypatch.setattr(terminal, "_engine", lambda: None)
    monkeypatch.setattr(terminal, "_holder_alive", lambda: None)

    def start():
        terminal._holder = "holder"
        terminal._expires = float("inf")

    def create(name, lease, *, corpus_readonly):
        creations.append((name, lease, terminal._work, corpus_readonly))
        return name

    def docker(*argv, source, **kwargs):
        records.append((argv, source.read()))
        return reply()

    monkeypatch.setattr(terminal, "_start", start)
    monkeypatch.setattr(terminal, "_create", create)
    monkeypatch.setattr(terminal, "_require", docker)
    monkeypatch.setattr(terminal, "_remove_container", removals.append)
    commands = ["printf 'quoted command marker'", "echo second marker"]
    for cmd in commands:
        assert terminal.run(cmd, timeout=2, preview=512).exit_code == 0
    assert [record[1] for record in records] == [cmd.encode() for cmd in commands]
    assert all(not any(cmd in arg for arg in argv) for cmd, (argv, _) in zip(commands, records))
    assert creations[0][0] != creations[1][0]
    assert creations[0][2:] == creations[1][2:] == (terminal._work, True)
    assert removals == [item[0] for item in creations]


def test_mismatched_container_policy_prevents_start(monkeypatch):
    settings = DockerSettings(Path("docker"), "engine", "sha256:" + "a" * 64)
    terminal = DockerTerminal(settings)
    calls = []

    def fake(*argv, **_kwargs):
        calls.append(argv[:2])
        if argv[:2] == ("container", "create"):
            return b"b" * 64
        pytest.fail("started a container with unverified isolation")

    def inspected(kind, ref):
        assert kind == "container"
        return {"Id": ref, "Image": settings.image_id,
                "Config": {"User": "0:0"}, "HostConfig": {"NetworkMode": "host"}}

    monkeypatch.setattr(terminal, "_require", fake)
    monkeypatch.setattr(terminal, "_inspect", inspected)
    with pytest.raises(DockerIsolationError, match="configuration"):
        terminal._create("unstarted", 30, corpus_readonly=True)
    assert calls == [("container", "create")]


def test_unconfirmed_cleanup_blocks_other_sessions(monkeypatch):
    import company_email_agent.sandbox_docker as module
    import threading

    monkeypatch.setattr(module, "_CLEANUP_UNCERTAIN", threading.Event())
    terminal = DockerTerminal()
    terminal._containers["owned"] = "b" * 64
    monkeypatch.setattr(terminal, "_engine", lambda: None)

    def failed_cleanup(_name):
        raise DockerIsolationError("Docker removal could not be verified.")

    monkeypatch.setattr(terminal, "_remove_container", failed_cleanup)
    assert not terminal.close()
    other = DockerTerminal()
    monkeypatch.setattr(other, "_start", lambda: pytest.fail("started Docker while cleanup is uncertain"))
    assert other.run("echo no", timeout=1, preview=512).rejected


def test_frame_rejects_untrusted_output_filename():
    assert _result(reply(stdout=b"text"), 1).stdout == "text"
    with pytest.raises(DockerIsolationError, match="invalid result"):
        _result(reply(filename="../host-note.txt"), 1)


def course_bundle(tmp_path, *, hardlink=False, bad_hash=False):
    folder = tmp_path / "data"
    folder.mkdir()
    raw = gzip.compress(b"{}\n")
    bundle = folder / "emails.jsonl.gz"
    if hardlink:
        unrelated = tmp_path / "host-canary.txt"
        unrelated.write_bytes(raw)
        bundle.hardlink_to(unrelated)
    else:
        bundle.write_bytes(raw)
    manifest = {"bytes": len(raw), "sha256": "0" * 64 if bad_hash else hashlib.sha256(raw).hexdigest()}
    (folder / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_corpus_only_contains_explicit_course_files(monkeypatch, tmp_path):
    folder = course_bundle(tmp_path)
    docs = folder / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text("course", encoding="utf-8")
    (docs / "unrelated.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "host-canary.txt").write_text("ignore", encoding="utf-8")
    monkeypatch.setattr(data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data, "DOCS_DIR", docs)
    target = io.BytesIO()
    _make_corpus(target)
    with tarfile.open(fileobj=target) as result:
        assert result.getnames() == ["emails.jsonl", "docs/policy.md"]
        assert result.extractfile("emails.jsonl").read() == b"{}\n"


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"hardlink": True}, "independent regular file"),
    ({"bad_hash": True}, "SHA-256 check"),
])
def test_corpus_refuses_linked_or_changed_archives(monkeypatch, tmp_path, kwargs, message):
    folder = course_bundle(tmp_path, **kwargs)
    monkeypatch.setattr(data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data, "DOCS_DIR", folder / "docs")
    with pytest.raises(DockerIsolationError, match=message):
        _make_corpus(io.BytesIO())


@pytest.mark.parametrize(
    ("args", "denied", "approved", "expected", "approval_count", "execution_count"),
    [
        ({}, False, True, "invalid_args", 0, 0),
        ({"cmd": "echo ok", "mount": "host"}, False, True, "invalid_args", 0, 0),
        ({"cmd": "echo denied"}, True, True, "denied", 0, 0),
        ({"cmd": "echo ok"}, False, False, "denied", 1, 0),
        ({"cmd": "echo ok"}, False, True, "ok", 1, 1),
    ],
)
def test_schema_deny_approval_still_precede_terminal(
    monkeypatch, args, denied, approved, expected, approval_count, execution_count,
):
    approvals, executed = [], []

    class User(ScriptedUser):
        def approve(self, call):
            approvals.append(call)
            return Approval(approved, "test decision")

    class Terminal:
        def run(self, cmd, **kwargs):
            executed.append(cmd)
            return sandbox.TerminalResult(stdout="ok", exit_code=0)

        def close(self):
            return True

    monkeypatch.setattr(session_module, "DockerTerminal", Terminal)
    session = adapters.run_agent_session(
        lambda _: "done", None,
        AgentConfig(mode="confirm", user=User(), deny_rules=(r"\bdenied\b",) if denied else ()),
    )
    result = session._run_tool(ToolCall("run_terminal", args))
    assert (result.status, len(approvals), len(executed)) == (expected, approval_count, execution_count)


def test_custom_tools_cannot_replace_the_managed_terminal(monkeypatch):
    called = []

    class Terminal:
        def run(self, cmd, **kwargs):
            called.append(cmd)
            return sandbox.TerminalResult(stdout="managed", exit_code=0)

    monkeypatch.setattr(session_module, "DockerTerminal", Terminal)
    alternate = ToolSpec(
        name="run_terminal", description="alternate", parameters={},
        fn=lambda **kwargs: pytest.fail("ran a replacement backend"), read_only=True,
    )
    session = adapters.run_agent_session(lambda _: "done", [alternate], AgentConfig())
    result = session._run_tool(ToolCall("run_terminal", {"cmd": "echo managed"}))
    assert result.status == "ok" and "managed" in result.content and called == ["echo managed"]
