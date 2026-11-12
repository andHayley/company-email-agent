"""Fail-closed Windows Docker Desktop backend for the built-in terminal tool."""

from __future__ import annotations

import atexit
import base64
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO
import uuid

from company_email_agent import data, sandbox

ENDPOINT = "npipe:////./pipe/dockerDesktopLinuxEngine"
CONFIG = data.REPO_ROOT / ".lm_cache" / "docker-terminal.json"
ENTRYPOINT = ["/usr/bin/python3", "-I", "/opt/terminal-entrypoint.py"]
LABEL = "org.cardinal.terminal.session"
POLICY_LABEL = "org.cardinal.terminal.policy"
UID = "10001:10001"
IMAGE_ENV = ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"]
MEMORY = 512 * 1024 * 1024
LEASE = 900
COMMAND_LIMIT = 65_536
CORPUS_LIMIT = 90 * 1024 * 1024
WORK_SIZE = "32m"
CORPUS_SIZE = "96m"
TMP_OPTIONS = "rw,nosuid,nodev,noexec,size=16m,nr_inodes=1024,mode=1777"
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_OUTPUT_FILE = re.compile(r"terminal-output-[0-9a-f]{32}\.txt")
_STDERR_FILE = re.compile(r"terminal-stderr-[0-9a-f]{32}\.txt")
_DOC = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\.md")
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: set[str] = set()
_CLEANUP_UNCERTAIN = threading.Event()


class DockerIsolationError(RuntimeError):
    pass


class DockerTransportTimeout(DockerIsolationError):
    pass


@dataclass(frozen=True)
class DockerSettings:
    cli: Path
    engine_id: str
    image_id: str

    @classmethod
    def load(cls) -> DockerSettings:
        local = os.environ.get("LOCALAPPDATA")
        if os.name != "nt" or not local:
            raise DockerIsolationError("The configured backend requires local Windows Docker Desktop.")
        cli = Path(local) / "Programs/DockerDesktop/resources/bin/docker.exe"
        try:
            with CONFIG.open("rb") as source:
                settings = json.loads(source.read(1025))
            image, engine = settings["image_id"], settings["engine_id"]
            if set(settings) != {"engine_id", "image_id"} or not isinstance(engine, str):
                raise ValueError
            if not re.fullmatch(r"[a-zA-Z0-9:-]{16,128}", engine) or not _IMAGE_ID.fullmatch(image):
                raise ValueError
            if not cli.is_absolute() or not cli.is_file():
                raise ValueError
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DockerIsolationError("No verified local Docker terminal image is configured.") from exc
        return cls(cli.resolve(), engine, image)


def _volume_options(size: str) -> dict[str, str]:
    return {
        "type": "tmpfs", "device": "tmpfs",
        "o": f"size={size},nr_inodes=4096,uid=10001,gid=10001,mode=0700,nosuid,nodev,noexec",
    }


def _regular_corpus_file(path: Path) -> os.stat_result:
    relative = path.relative_to(data.REPO_ROOT)
    ancestor = data.REPO_ROOT
    for part in relative.parts[:-1]:
        ancestor /= part
        info = ancestor.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise DockerIsolationError("A course corpus directory is a link or is not a directory.")
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise DockerIsolationError("A course corpus source is not an independent regular file.")
    return info


def _identity(info: os.stat_result) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink


def _bundle_email_archive(target: BinaryIO) -> int:
    # The repository ships the digest-checked source independently of old
    # native workspaces, which may hard-link the decompressed archive.
    root = data.REPO_ROOT / "data"
    metadata = root / "source.json"
    _regular_corpus_file(metadata)
    with metadata.open("rb") as manifest:
        expected = json.loads(manifest.read(4097))
    bundled = root / "emails.jsonl.gz"
    info = _regular_corpus_file(bundled)
    if info.st_size != expected["bytes"] or not 0 < info.st_size <= CORPUS_LIMIT:
        raise DockerIsolationError("The bundled course archive size does not match its manifest.")
    with bundled.open("rb") as compressed:
        if _identity(os.fstat(compressed.fileno())) != _identity(info):
            raise DockerIsolationError("The bundled course archive changed before it was opened.")
        if hashlib.file_digest(compressed, "sha256").hexdigest() != expected["sha256"]:
            raise DockerIsolationError("The bundled course archive failed its SHA-256 check.")
        compressed.seek(0)
        size = 0
        with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
            while chunk := source.read(65536):
                size += len(chunk)
                if size > CORPUS_LIMIT:
                    raise DockerIsolationError("The decompressed course archive exceeds its size limit.")
                target.write(chunk)
        if _identity(os.fstat(compressed.fileno())) != _identity(info):
            raise DockerIsolationError("The bundled course archive changed while it was copied.")
    target.seek(0)
    return size


def _make_corpus(target: BinaryIO) -> None:
    """Copy the digest-checked archive and direct Markdown files to a bounded tar."""
    paths = [
        (path, f"docs/{path.name}") for path in sorted(data.DOCS_DIR.glob("*.md"))
        if _DOC.fullmatch(path.name)
    ]
    if len(paths) > 255:
        raise DockerIsolationError("Too many files in the course corpus.")
    with tempfile.TemporaryFile(mode="w+b") as mail, tarfile.open(
        fileobj=target, mode="w", format=tarfile.USTAR_FORMAT,
    ) as archive:
        total = _bundle_email_archive(mail)
        member = tarfile.TarInfo("emails.jsonl")
        member.size = total
        archive.addfile(member, mail)
        for path, name in paths:
            before = _regular_corpus_file(path)
            total += before.st_size
            if total > CORPUS_LIMIT:
                raise DockerIsolationError("The course corpus exceeds its size limit.")
            with path.open("rb") as source:
                if _identity(os.fstat(source.fileno())) != _identity(before):
                    raise DockerIsolationError("A course corpus file changed before it was copied.")
                member = tarfile.TarInfo(name)
                member.size = before.st_size
                archive.addfile(member, source)
                if _identity(os.fstat(source.fileno())) != _identity(before):
                    raise DockerIsolationError("A course corpus file changed while it was copied.")
    target.seek(0)


def _result(payload: bytes, seconds: float) -> sandbox.TerminalResult:
    try:
        body = json.loads(payload)
        code, expired, filename = body["exit_code"], body["timed_out"], body["output_file"]
        if type(expired) is not bool or (code is not None if expired else type(code) is not int):
            raise ValueError
        stderr_file = body.get("stderr_file")
        if ((filename is not None and not _OUTPUT_FILE.fullmatch(filename))
                or (stderr_file is not None and not _STDERR_FILE.fullmatch(stderr_file))):
            raise ValueError
        out, err = (base64.b64decode(body[key], validate=True) for key in ("stdout", "stderr"))
        if max(len(out), len(err)) > sandbox.MAX_OUTPUT_BYTES + 100:
            raise ValueError
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise DockerIsolationError("The terminal container returned an invalid result.") from exc
    return sandbox.TerminalResult(
        stdout=out.decode("utf-8", "replace"), stderr=err.decode("utf-8", "replace"),
        exit_code=code, timed_out=expired, output_file=filename, stderr_file=stderr_file,
        reason=f"Timed out after {seconds:g} seconds; the command was killed." if expired else "",
    )


class DockerTerminal:
    """One lazily initialized, bounded workspace shared by separate command containers."""

    def __init__(self, settings: DockerSettings | None = None):
        self._settings = settings
        self._lock = threading.RLock()
        self._client: tempfile.TemporaryDirectory | None = None
        self._closed = False
        self._failed = False
        self._key = uuid.uuid4().hex
        self._work = f"cardinal-work-{self._key}"
        self._corpus = f"cardinal-corpus-{self._key}"
        self._containers: dict[str, str | None] = {}
        self._volumes: set[str] = set()
        self._holder: str | None = None
        self._holder_started: str | None = None
        self._expires = 0.0

    def _invoke(
        self, *args: str, source: BinaryIO | None = None, timeout: float = 8,
        limit: int = 256_000,
    ) -> tuple[int, bytes]:
        assert self._settings is not None and self._client is not None
        directory = self._client.name
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
        env.update({
            "PATH": str(self._settings.cli.parent), "TEMP": directory, "TMP": directory,
            "HOME": directory, "USERPROFILE": directory, "DOCKER_CLI_HINTS": "false",
        })
        argv = [
            str(self._settings.cli), "--config", directory, "--host", ENDPOINT, *args,
        ]
        with subprocess.Popen(
            argv, shell=False, cwd=directory, env=env,
            stdin=subprocess.PIPE if source is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as child:
            stdout = bytearray()
            overflow = threading.Event()
            broken_input = threading.Event()

            def drain(pipe, keep: bool) -> None:
                try:
                    while chunk := pipe.read1(65536):
                        if keep:
                            room = limit - len(stdout)
                            if len(chunk) > room:
                                overflow.set()
                            stdout.extend(chunk[:room])
                except OSError:
                    overflow.set()

            def feed() -> None:
                try:
                    with child.stdin as pipe:
                        while chunk := source.read(65536):
                            pipe.write(chunk)
                except (BrokenPipeError, OSError):
                    broken_input.set()

            workers = [
                threading.Thread(target=drain, args=(child.stdout, True), daemon=True),
                threading.Thread(target=drain, args=(child.stderr, False), daemon=True),
            ]
            if source is not None:
                workers.append(threading.Thread(target=feed, daemon=True))
            for worker in workers:
                worker.start()
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise DockerTransportTimeout("The Docker client exceeded its operation deadline.") from exc
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=3)
                for worker in workers:
                    worker.join(timeout=2)
            if overflow.is_set() or any(worker.is_alive() for worker in workers):
                raise DockerIsolationError("Docker CLI output exceeded its verified bounds.")
            if broken_input.is_set() and child.returncode == 0:
                raise DockerIsolationError("Docker did not accept the complete terminal input.")
            return child.returncode, bytes(stdout)

    def _require(self, *args: str, **kwargs) -> bytes:
        status, output = self._invoke(*args, **kwargs)
        if status:
            raise DockerIsolationError(f"Docker {args[0]} {args[1]} failed before a usable result was received.")
        return output

    def _engine(self) -> None:
        output = self._require(
            "system", "info", "--format", "{{json .ID}}|{{.OSType}}|{{.CgroupVersion}}|{{json .SecurityOptions}}",
        ).decode().strip()
        engine, os_type, cgroup, security = output.split("|", 3)
        if (json.loads(engine) != self._settings.engine_id or os_type != "linux"
                or cgroup != "2" or "name=seccomp,profile=builtin" not in (json.loads(security) or [])):
            raise DockerIsolationError("Docker Engine identity or Linux isolation capabilities do not match.")

    def _inspect(self, kind: str, reference: str) -> dict:
        output = self._require(kind, "inspect", reference, "--format", "{{json .}}")
        entry = json.loads(output)
        if not isinstance(entry, dict):
            raise DockerIsolationError("Docker inspect did not return an object.")
        return entry

    def _create_volume(self, name: str, size: str) -> None:
        options = _volume_options(size)
        args = ["volume", "create", "--driver", "local", "--label", f"{LABEL}={self._key}"]
        for key, value in options.items():
            args.extend(("--opt", f"{key}={value}"))
        self._volumes.add(name)
        returned = self._require(*args, name).decode().strip()
        self._verify_volume(name, size)
        if returned != name:
            raise DockerIsolationError("Docker returned an unexpected volume name.")

    def _verify_volume(self, name: str, size: str) -> None:
        record = self._inspect("volume", name)
        if (record.get("Name") != name or record.get("Driver") != "local"
                or record.get("Scope") != "local" or record.get("Options") != _volume_options(size)
                or (record.get("Labels") or {}).get(LABEL) != self._key):
            raise DockerIsolationError("Docker did not confirm the bounded session volume.")

    def _create(self, name: str, lease: int, *, corpus_readonly: bool) -> str:
        mode = "ro" if corpus_readonly else "rw"
        corpus_mount = f"type=volume,src={self._corpus},dst=/corpus,volume-nocopy"
        if corpus_readonly:
            corpus_mount += ",readonly"
        self._containers[name] = None
        output = self._require(
            "container", "create", "--name", name, "--label", f"{LABEL}={self._key}",
            "--rm", "--pull=never", "--network=none", "--ipc=none", "--cgroupns=private",
            "--user", UID, "--workdir", "/workspace", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", "--no-healthcheck", "--log-driver=none",
            "--cpus=1", f"--memory={MEMORY}", f"--memory-swap={MEMORY}", "--pids-limit=64",
            "--ulimit=nofile=256:256", "--ulimit=core=0:0", "--stop-timeout=1",
            "--tmpfs", f"/tmp:{TMP_OPTIONS}",
            "--mount", f"type=volume,src={self._work},dst=/workspace,volume-nocopy",
            "--mount", corpus_mount, self._settings.image_id, "hold", mode, str(lease),
        ).decode().strip()
        if not _CONTAINER_ID.fullmatch(output):
            raise DockerIsolationError("Docker did not return a container ID.")
        self._containers[name] = output
        self._verify_container(output, mode, lease)
        self._require("container", "start", output)
        ready = self._require("container", "exec", output, *ENTRYPOINT, "verify", mode)
        if ready.strip() != b"ready":
            raise DockerIsolationError("The Docker runtime did not confirm its isolation.")
        return output

    def _verify_container(self, cid: str, mode: str, lease: int) -> None:
        info = self._inspect("container", cid)
        config, host = info["Config"], info["HostConfig"]
        limits = {
            "AutoRemove": True, "Privileged": False, "ReadonlyRootfs": True,
            "NetworkMode": "none", "IpcMode": "none", "CgroupnsMode": "private",
            "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges:true"],
            "Memory": MEMORY, "MemorySwap": MEMORY, "NanoCpus": 1_000_000_000,
            "PidsLimit": 64, "Tmpfs": {"/tmp": TMP_OPTIONS}, "PidMode": "",
        }
        forbidden = ("Binds", "Devices", "DeviceRequests", "CapAdd", "Links", "VolumesFrom",
                     "PortBindings", "PublishAllPorts", "ExtraHosts")
        if (info.get("Id") != cid or info.get("Image") != self._settings.image_id
                or config.get("User") != UID or config.get("WorkingDir") != "/workspace"
                or config.get("Entrypoint") != ENTRYPOINT or config.get("Tty")
                or config.get("Env") != IMAGE_ENV
                or config.get("Cmd") != ["hold", mode, str(lease)]
                or config.get("Volumes") or config.get("ExposedPorts")
                or (config.get("Labels") or {}).get(LABEL) != self._key
                or (host.get("LogConfig") or {}).get("Type") != "none"
                or any(host.get(key) != val for key, val in limits.items())
                or any(host.get(key) for key in forbidden)):
            raise DockerIsolationError("Docker container configuration does not match the terminal policy.")
        expected = {"/workspace": (self._work, True), "/corpus": (self._corpus, mode == "rw")}
        mounts = info.get("Mounts", [])
        # Inspect may list a tmpfs mount in Mounts as well as in HostConfig.Tmpfs.
        seen = {}
        for mount in mounts:
            destination = mount.get("Destination")
            if destination == "/tmp" and mount.get("Type") == "tmpfs":
                continue
            if destination in seen or mount.get("Type") != "volume" or mount.get("Driver") != "local":
                raise DockerIsolationError("Docker reported an unexpected terminal mount.")
            seen[destination] = (mount.get("Name"), mount.get("RW"))
        if seen != expected:
            raise DockerIsolationError("Docker did not confirm the terminal mount boundaries.")

    def _start(self) -> None:
        if self._settings is None:
            self._settings = DockerSettings.load()
        with _ACTIVE_LOCK:
            if len(_ACTIVE) >= 3:
                raise DockerIsolationError("The process already has three Docker terminal sessions.")
            _ACTIVE.add(self._key)
        atexit.register(self.close)
        self._client = tempfile.TemporaryDirectory(prefix="cardinal-docker-")
        self._engine()
        image = self._inspect("image", self._settings.image_id)
        config = image.get("Config") or {}
        if (image.get("Id") != self._settings.image_id or image.get("Os") != "linux"
                or config.get("Entrypoint") != ENTRYPOINT or config.get("User") != UID
                or config.get("Env") != IMAGE_ENV
                or config.get("Volumes") or config.get("ExposedPorts")
                or (config.get("Labels") or {}).get(POLICY_LABEL) != "2"):
            raise DockerIsolationError("The pinned Docker terminal image does not match the policy.")
        with tempfile.TemporaryFile(mode="w+b") as archive:
            _make_corpus(archive)
            self._create_volume(self._work, WORK_SIZE)
            self._create_volume(self._corpus, CORPUS_SIZE)
            name = f"cardinal-hold-{self._key}"
            self._expires = time.monotonic() + LEASE
            holder = self._create(name, LEASE, corpus_readonly=False)
            seeded = self._require("container", "exec", "-i", holder, *ENTRYPOINT, "seed",
                                   source=archive, timeout=45)
            if seeded.strip() != b"seeded":
                raise DockerIsolationError("Docker did not confirm the course corpus was staged.")
            state = self._inspect("container", holder)["State"]
            if not state.get("Running") or not state.get("StartedAt"):
                raise DockerIsolationError("The Docker workspace holder is not running.")
            self._holder_started = state["StartedAt"]
            self._holder = holder

    def _holder_alive(self) -> None:
        state = self._inspect("container", self._holder)["State"]
        if not state.get("Running") or state.get("StartedAt") != self._holder_started:
            raise DockerIsolationError("The session workspace is no longer available.")

    def _remove_container(self, name: str) -> None:
        matches = self._require(
            "container", "ls", "-a", "--no-trunc", "--filter", f"name=^/{name}$",
            "--format", "{{.ID}}",
        ).decode().split()
        if matches:
            if len(matches) != 1 or not _CONTAINER_ID.fullmatch(matches[0]):
                raise DockerIsolationError("Docker returned ambiguous cleanup state.")
            cid = matches[0]
            expected = self._containers[name]
            if expected is not None and cid != expected:
                raise DockerIsolationError("Docker returned an unexpected container during cleanup.")
            info = self._inspect("container", cid)
            if (info["Config"].get("Labels") or {}).get(LABEL) != self._key:
                raise DockerIsolationError("Docker container cleanup ownership could not be verified.")
            self._invoke("container", "rm", "--force", cid)
            remaining = self._require(
                "container", "ls", "-a", "--filter", f"name=^/{name}$", "--format", "{{.ID}}",
            )
            if remaining.strip():
                raise DockerIsolationError("Docker could not verify command container removal.")
        self._containers.pop(name, None)

    def _close(self) -> bool:
        try:
            if self._containers or self._volumes:
                self._engine()
            for name in reversed(tuple(self._containers)):
                self._remove_container(name)
            for name in tuple(self._volumes):
                volumes = self._require("volume", "ls", "--format", "{{.Name}}").decode().splitlines()
                if name in volumes:
                    size = WORK_SIZE if name == self._work else CORPUS_SIZE
                    self._verify_volume(name, size)
                    self._require("volume", "rm", name)
                    remaining = self._require("volume", "ls", "--format", "{{.Name}}").decode().splitlines()
                    if name in remaining:
                        raise DockerIsolationError("Docker could not verify session volume removal.")
                self._volumes.remove(name)
            if self._client is not None:
                self._client.cleanup()
                self._client = None
        except (DockerIsolationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            _CLEANUP_UNCERTAIN.set()
            return False
        atexit.unregister(self.close)
        with _ACTIVE_LOCK:
            _ACTIVE.discard(self._key)
        return True

    def close(self) -> bool:
        """Release owned resources. A failed cleanup can be retried at process exit."""
        with self._lock:
            self._closed = True
            return self._close()

    def run(self, cmd: str, *, timeout: float, preview: int) -> sandbox.TerminalResult:
        reason = sandbox.check_command(cmd)
        if reason:
            return sandbox.TerminalResult(rejected=True, reason=reason)
        try:
            encoded = cmd.encode("utf-8")
            seconds = float(timeout)
            if not 0 < seconds <= 120 or not 128 <= preview <= 7200:
                raise ValueError
            if len(encoded) > COMMAND_LIMIT or b"\0" in encoded:
                raise ValueError
        except (ValueError, TypeError):
            return sandbox.TerminalResult(rejected=True, reason="The terminal input or limits are invalid.")
        with self._lock:
            if self._closed or self._failed or _CLEANUP_UNCERTAIN.is_set():
                return sandbox.TerminalResult(rejected=True, reason="Docker terminal is closed or isolation is unavailable.")
            name = f"cardinal-cmd-{uuid.uuid4().hex}"
            try:
                if self._holder is None:
                    self._start()
                self._engine()
                if time.monotonic() + seconds + 25 >= self._expires:
                    raise DockerIsolationError("The bounded Docker workspace lease has expired. Open a new session.")
                self._holder_alive()
                cid = self._create(name, math.ceil(seconds) + 20, corpus_readonly=True)
                try:
                    if _CLEANUP_UNCERTAIN.is_set():
                        raise DockerIsolationError("Docker terminal cleanup is uncertain.")
                    output = self._require(
                        "container", "exec", "-i", cid, *ENTRYPOINT, "execute", str(seconds), str(preview),
                        source=io.BytesIO(encoded), timeout=seconds + 8, limit=11_000_000,
                    )
                    result = _result(output, seconds)
                finally:
                    self._remove_container(name)
                self._holder_alive()
                return result
            except (DockerIsolationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                self._failed = True
                cleaned = self._close()
                reason = str(exc) if isinstance(exc, DockerIsolationError) else "Docker isolation could not be verified."
                if not cleaned:
                    reason += " Cleanup could not be confirmed. Further Docker terminal commands are blocked."
                return sandbox.TerminalResult(rejected=True, reason=reason)
            except BaseException:
                self._failed = True
                self._close()
                raise
