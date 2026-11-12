"""Opt-in Docker Desktop probes. Fixed local commands, no model or network target."""

import os
from pathlib import Path
import re
import shlex

import pytest

from company_email_agent import sandbox_docker as docker
from company_email_agent.sandbox_docker import DockerSettings, DockerTerminal


pytestmark = [pytest.mark.docker, pytest.mark.skipif(
    os.environ.get("CARDINAL_DOCKER_PROBES") != "1",
    reason="requires an explicitly selected local Docker Desktop Engine and pinned test image",
)]


@pytest.fixture
def settings():
    cli = Path(os.environ["LOCALAPPDATA"]) / "Programs/DockerDesktop/resources/bin/docker.exe"
    return DockerSettings(cli.resolve(strict=True), os.environ["CARDINAL_PROBE_ENGINE"], os.environ["CARDINAL_PROBE_IMAGE"])


@pytest.fixture
def terminal(settings):
    runner = DockerTerminal(settings)
    try:
        yield runner
    finally:
        assert runner.close()


def run(terminal, cmd, *, timeout=8, preview=1024):
    result = terminal.run(cmd, timeout=timeout, preview=preview)
    assert result.exit_code == 0, result
    return result.stdout


def py(code):
    return "python3 -c " + shlex.quote(code)


@pytest.mark.archive
def test_workspace_and_read_only_course_corpus(terminal, tmp_path):
    host_file = tmp_path / "host-canary.txt"
    host_file.write_text("local canary", encoding="utf-8")
    windows_path = host_file.as_posix()
    linux_host_guess = "/mnt/" + windows_path[0].lower() + windows_path[2:]
    cmd = "test -s emails.jsonl && test -d docs && printf 'workspace ok\\n'"
    assert run(terminal, cmd).strip() == "workspace ok"
    code = (
        "from pathlib import Path\n"
        f"assert not Path({linux_host_guess!r}).exists()\n"
        "assert not Path('/host_mnt').exists()\n"
        "assert not Path('/workspace/.env').exists()\n"
        "assert not Path('/var/run/docker.sock').exists()\n"
        "assert not Path('/run/host-services').exists()\n"
        "try:\n"
        " Path('/corpus/docs/terminal-probe-file').write_text('synthetic probe')\n"
        "except OSError:\n"
        " print('read-only corpus')\n"
        "else:\n"
        " raise AssertionError('corpus writable')\n"
    )
    assert run(terminal, py(code)).strip() == "read-only corpus"
    assert host_file.read_text(encoding="utf-8") == "local canary"


@pytest.mark.archive
def test_sandbox_identity_network_and_limits(terminal, monkeypatch):
    monkeypatch.setenv("CARDINAL_DOCKER_TEST_SECRET", "synthetic-host-only-canary")
    code = """
import os
from pathlib import Path
import socket
import resource
assert os.getuid() == os.getgid() == 10001
assert 'CARDINAL_DOCKER_TEST_SECRET' not in os.environ
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert status['NoNewPrivs'].strip() == '1'
assert status['Seccomp'].strip() == '2'
assert all(int(status[key], 16) == 0 for key in ('CapEff', 'CapBnd', 'CapPrm', 'CapInh', 'CapAmb'))
assert {name for _, name in socket.if_nameindex()} == {'lo'}
assert not Path('/proc/net/route').read_text().splitlines()[1:]
with socket.socket() as sock:
    sock.settimeout(0.5)
    assert sock.connect_ex(('192.0.2.1', 443)) != 0
cg = Path('/sys/fs/cgroup')
for name, value in {'pids.max':'64', 'memory.max':'536870912', 'memory.swap.max':'0', 'cpu.max':'100000 100000'}.items():
    assert (cg / name).read_text().strip() == value
fs = os.statvfs('/workspace')
assert fs.f_blocks * fs.f_frsize == 32 * 1024 * 1024
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (256, 256)
assert os.statvfs('/').f_flag & os.ST_RDONLY
print('uid, capabilities, seccomp, network, cgroups, workspace quota: verified')
"""
    assert run(terminal, py(code)).strip().endswith("workspace quota: verified")


@pytest.mark.archive
def test_independent_command_containers_share_only_their_session(terminal, settings, monkeypatch):
    ids = []
    original = terminal._create

    def capture(name, lease, *, corpus_readonly):
        cid = original(name, lease, corpus_readonly=corpus_readonly)
        if corpus_readonly:
            ids.append(cid)
        return cid

    monkeypatch.setattr(terminal, "_create", capture)
    assert run(terminal, "echo session-value > result.txt").strip() == ""
    assert run(terminal, "cat result.txt").strip() == "session-value"
    assert len(set(ids)) == 2
    live_ids = terminal._require("container", "ls", "-a", "--no-trunc", "--format", "{{.ID}}").decode().split()
    assert not set(ids).intersection(live_ids)
    other = DockerTerminal(settings)
    try:
        assert run(other, "test ! -e result.txt && echo separate").strip() == "separate"
    finally:
        assert other.close()


@pytest.mark.archive
def test_timeout_truncation_and_follow_up(terminal):
    result = terminal.run(py("import time; time.sleep(20)"), timeout=0.5, preview=512)
    assert result.timed_out and result.exit_code is None, result
    assert list(terminal._containers.values()) == [terminal._holder]
    listing = "import sys; print('first\\n'); print(*range(3, 5001), sep='\\n'); print('diagnostic', file=sys.stderr)"
    long = terminal.run(py(listing), timeout=4, preview=512)
    assert long.exit_code == 0 and len(long.stdout) > 12000
    assert re.fullmatch(r"terminal-output-[0-9a-f]{32}\.txt", long.output_file or "")
    assert re.fullmatch(r"terminal-stderr-[0-9a-f]{32}\.txt", long.stderr_file or "")
    assert run(terminal, "sed -n '1,2p;4321p' " + long.output_file) == "first\n\n4321\n"
    assert run(terminal, "cat " + long.stderr_file) == "diagnostic\n"


@pytest.mark.archive
def test_session_close_removes_owned_resources(terminal):
    run(terminal, "echo cleanup")
    names = (terminal._work, terminal._corpus)
    label = f"label={docker.LABEL}={terminal._key}"
    # Check after close via another connected holder-free client.
    observer = DockerTerminal(terminal._settings)
    observer._client = docker.tempfile.TemporaryDirectory()
    try:
        assert terminal.close()
        assert not observer._require("container", "ls", "-a", "--filter", label, "--format", "{{.ID}}").strip()
        volumes = observer._require("volume", "ls", "--format", "{{.Name}}").decode().split()
        assert not set(names).intersection(volumes)
    finally:
        assert observer.close()


def test_unavailable_pinned_image_fails_closed(settings):
    runner = DockerTerminal(DockerSettings(settings.cli, settings.engine_id, "sha256:" + "0" * 64))
    try:
        result = runner.run("echo harmless", timeout=1, preview=512)
        assert result.rejected and result.exit_code is None
        assert not runner._volumes and not runner._containers
    finally:
        assert runner.close()


@pytest.mark.archive
def test_agent_terminal_routes_approved_calls_to_the_container(settings, monkeypatch, tmp_path):
    from company_email_agent import adapters
    from company_email_agent.llm import ScriptedLM
    from company_email_agent.types import AgentConfig, Approval, ToolCall
    from company_email_agent.user import ScriptedUser

    approvals = []

    class User(ScriptedUser):
        def approve(self, call):
            approvals.append(call.args["cmd"])
            return Approval(True)

    monkeypatch.setattr(DockerSettings, "load", classmethod(lambda cls: settings))
    first = "echo agent-container-probe > agent.txt"
    second = "cat agent.txt"
    lm = ScriptedLM([
        adapters.run_format_tool_call("", ToolCall("run_terminal", {"cmd": first})),
        adapters.run_format_tool_call("", ToolCall("run_terminal", {"cmd": second})),
        "Done.",
    ])
    session = adapters.run_agent_session(
        lm, None, AgentConfig(mode="confirm", user=User(), workspace=tmp_path),
    )
    try:
        result = session.send("Use the session workspace to write and read a test marker.")
        records = [event for event in result.transcript if event["type"] == "tool_result"]
        assert result.status == "done"
        assert approvals == [first, second]
        assert [event["status"] for event in records] == ["ok", "ok"], records
        assert "agent-container-probe" in records[-1]["content"]
        assert not (tmp_path / "agent.txt").exists()
    finally:
        assert session.close()
