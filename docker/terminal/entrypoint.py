"""Trusted entry for one isolated terminal command or a short-lived volume holder."""

import base64
import ctypes
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import tarfile
import time
import uuid

WORK = Path("/workspace")
CORPUS = Path("/corpus")
UID = 10001
MEMORY = 512 * 1024 * 1024
LIMITS = {WORK: 32 * 1024 * 1024, CORPUS: 96 * 1024 * 1024, Path("/tmp"): 16 * 1024 * 1024}
OUTPUT_LIMIT = 4_000_000
COMMAND_LIMIT = 65_536
DOC = re.compile(r"docs/[a-zA-Z0-9][a-zA-Z0-9_.-]*\.md")
ENV = {
    "PATH": "/opt/terminal-bin",
    "HOME": "/workspace",
    "TMPDIR": "/tmp",
    "LC_ALL": "C",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def require(condition):
    if not condition:
        raise RuntimeError("Terminal container did not pass its runtime checks.")


def protect_supervisor():
    # Protect the trusted process' file descriptors and memory from the command's UID.
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(4, 0, 0, 0, 0) == 0)  # PR_SET_DUMPABLE


def verify(corpus_readonly):
    require(os.getuid() == UID and os.getgid() == UID)
    proc = Path("/proc/self/status").read_text()
    status = dict(line.split(":", 1) for line in proc.splitlines() if ":" in line)
    for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        require(int(status[name].strip(), 16) == 0)
    require(status["NoNewPrivs"].strip() == "1" and status["Seccomp"].strip() == "2")
    require({p.name for p in Path("/sys/class/net").iterdir()} == {"lo"})
    require(not Path("/var/run/docker.sock").exists() and not Path("/run/host-services").exists())
    require(os.statvfs("/").f_flag & os.ST_RDONLY)
    mounts = {}
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mounts[left.split()[4]] = right.split()[0]
    for path, limit in LIMITS.items():
        require(mounts.get(str(path)) == "tmpfs")
        fs = os.statvfs(path)
        require(fs.f_blocks * fs.f_frsize == limit and 0 < fs.f_files <= 4096)
        require(bool(fs.f_flag & os.ST_RDONLY) == (path == CORPUS and corpus_readonly))
        require(bool(fs.f_flag & os.ST_NOSUID) and bool(fs.f_flag & os.ST_NODEV))
        require(bool(fs.f_flag & os.ST_NOEXEC))
    cgroup = Path("/sys/fs/cgroup")
    for key, value in {
        "memory.max": str(MEMORY), "memory.swap.max": "0",
        "pids.max": "64", "cpu.max": "100000 100000",
    }.items():
        require((cgroup / key).read_text().strip() == value)


def seed():
    require(not any(WORK.iterdir()) and not any(CORPUS.iterdir()))
    (CORPUS / "docs").mkdir(mode=0o700)
    seen = set()
    total = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
        for member in archive:
            name = member.name
            total += member.size
            require(member.isfile() and name not in seen and len(seen) < 256)
            require(name == "emails.jsonl" or DOC.fullmatch(name))
            require(0 <= member.size and total <= 90 * 1024 * 1024)
            seen.add(name)
            with (CORPUS / name).open("xb") as out, archive.extractfile(member) as source:
                while chunk := source.read(65536):
                    out.write(chunk)
    for name in ("emails.jsonl", "docs"):
        if (CORPUS / name).exists():
            (WORK / name).symlink_to(CORPUS / name)
    # Drain archive padding before returning to the host writer.
    while sys.stdin.buffer.read(65536):
        pass
    print("seeded", flush=True)


def collect(command, seconds):
    buffers = [bytearray(), bytearray()]
    cut = [False, False]
    deadline = time.monotonic() + seconds
    with subprocess.Popen(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        cwd=WORK, env=ENV, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    ) as child, selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ, 0)
        selector.register(child.stderr, selectors.EVENT_READ, 1)
        expired = False
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("terminal command", seconds)
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    index = key.data
                    room = OUTPUT_LIMIT - len(buffers[index])
                    buffers[index].extend(chunk[:room])
                    cut[index] |= len(chunk) > room
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            expired = True
        finally:
            # Container exit also stops descendants that changed their process group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=1)
    return None if expired else child.returncode, expired, tuple(map(bytes, buffers)), cut


def save_stream(prefix, capture_id, stream):
    filename = f"terminal-{prefix}-{capture_id}.txt"
    try:
        fd = os.open(WORK / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as target:
            target.write(stream)
    except OSError:
        return None
    return filename


def execute(seconds, preview):
    payload = sys.stdin.buffer.read(COMMAND_LIMIT + 1)
    require(0 < len(payload) <= COMMAND_LIMIT and b"\0" not in payload)
    command = payload.decode("utf-8")
    code, expired, (out, err), (out_cut, err_cut) = collect(command, seconds)
    output_file = stderr_file = None
    if len(out) + len(err) > preview:
        capture_id = uuid.uuid4().hex
        if out:
            output_file = save_stream("output", capture_id, out)
        if err:
            stderr_file = save_stream("stderr", capture_id, err)
    # Keep the notice in the tool preview. Saved stream bytes retain their line positions.
    notice = f"\n[sandbox: output cut off after {OUTPUT_LIMIT} bytes]".encode()
    if out_cut:
        out += notice
    if err_cut:
        err += notice
    print(json.dumps({
        "exit_code": code, "timed_out": expired, "output_file": output_file, "stderr_file": stderr_file,
        "stdout": base64.b64encode(out).decode("ascii"), "stderr": base64.b64encode(err).decode("ascii"),
    }), flush=True)


def main():
    protect_supervisor()
    action = sys.argv[1]
    if action == "hold":
        lease = int(sys.argv[3])
        require(1 <= lease <= 900)
        signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
        signal.alarm(lease)
        verify(sys.argv[2] == "ro")
        print("ready", flush=True)
        while True:
            signal.pause()
    elif action == "verify":
        verify(sys.argv[2] == "ro")
        print("ready", flush=True)
    elif action == "seed":
        verify(False)
        seed()
    elif action == "execute":
        verify(True)
        seconds, preview = float(sys.argv[2]), int(sys.argv[3])
        require(0 < seconds <= 120 and 128 <= preview <= 7200)
        execute(seconds, preview)
    else:
        raise RuntimeError("Unknown terminal container action.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Use a fixed diagnostic. Command output is returned only by the framed result.
        sys.stderr.write("Docker terminal entrypoint refused or could not complete its fixed action.\n")
        sys.exit(70)
