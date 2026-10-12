"""Sandboxed execution for model-generated simulation and analysis code.

What the source plan called a sandbox was ``subprocess.run(["python3", path])``
with a timeout. That is not a sandbox in any sense: the child inherits the
parent's environment, filesystem, network and privileges, and a timeout is the
only constraint. Model-generated code that runs with the model's own
credentials is the single highest-severity component in this design, and
calling an unconstrained subprocess "an isolated POSIX micro-VM" is how that
gets shipped.

What is actually enforced here:

* a **separate process group**, killed as a group on timeout so that orphaned
  grandchildren cannot outlive the call;
* ``RLIMIT_CPU``, ``RLIMIT_AS``, ``RLIMIT_FSIZE``, ``RLIMIT_NPROC`` and
  ``RLIMIT_CORE`` applied in the child *before* ``exec``;
* a **scrubbed environment**: nothing inherited except an explicit allowlist;
* a private temporary working directory, removed afterwards;
* **network isolation via a new network namespace** when ``unshare`` is
  available, and an explicit, recorded ``network_isolated: False`` when it is
  not — never a silent assumption of isolation.

What is *not* enforced, stated plainly because a partial sandbox described as a
complete one is worse than none: there is no seccomp filter, no filesystem
namespace, and no user namespace by default. Read access to the host
filesystem is still possible. For untrusted code in production this needs
gVisor, Firecracker or an equivalent; ``SandboxResult.isolation`` records
exactly which guarantees were actually obtained so a caller can refuse to
trust a result that was produced with fewer than it required.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "PYTHONHASHSEED")


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 10
    wall_seconds: float = 20.0
    address_space_mb: int = 2048
    file_size_mb: int = 64
    max_processes: int = 64


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    isolation: dict[str, bool] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def require(self, **guarantees: bool) -> "SandboxResult":
        """Raise unless the run actually obtained the named guarantees."""
        for name, needed in guarantees.items():
            if needed and not self.isolation.get(name, False):
                raise PermissionError(
                    f"sandbox did not provide {name!r}; refusing to use the result"
                )
        return self


def _unshare_available() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if shutil.which("unshare") is None:
        return False
    try:
        probe = subprocess.run(
            ["unshare", "--net", "--map-root-user", "true"],
            capture_output=True, timeout=5,
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_UNSHARE = None


def unshare_available() -> bool:
    global _UNSHARE
    if _UNSHARE is None:
        _UNSHARE = _unshare_available()
    return _UNSHARE


class Sandbox:
    """Runs a Python program under resource limits and reports what it got."""

    def __init__(
        self,
        limits: SandboxLimits | None = None,
        env_allowlist: Sequence[str] = DEFAULT_ENV_ALLOWLIST,
        allow_network: bool = False,
    ) -> None:
        self.limits = limits or SandboxLimits()
        self.env_allowlist = tuple(env_allowlist)
        self.allow_network = allow_network

    def _preexec(self):
        limits = self.limits

        def apply() -> None:
            os.setsid()
            resource.setrlimit(
                resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds)
            )
            mem = limits.address_space_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            size = limits.file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
            resource.setrlimit(
                resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes)
            )
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return apply

    def _environment(self, workdir: Path) -> dict[str, str]:
        env = {k: os.environ[k] for k in self.env_allowlist if k in os.environ}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["HOME"] = str(workdir)
        env["TMPDIR"] = str(workdir)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def run(self, code: str, *, timeout: Optional[float] = None) -> SandboxResult:
        """Execute ``code``. It may write a JSON dict to ``$IRIDIUM_RESULT``."""
        import time

        wall = timeout if timeout is not None else self.limits.wall_seconds
        network_isolated = not self.allow_network and unshare_available()
        isolation = {
            "separate_process_group": True,
            "resource_limits": True,
            "scrubbed_environment": True,
            "private_workdir": True,
            "network_isolated": network_isolated,
            "filesystem_namespace": False,
            "seccomp": False,
        }

        workdir = Path(tempfile.mkdtemp(prefix="iridium-sandbox-"))
        try:
            script = workdir / "program.py"
            result_path = workdir / "result.json"
            script.write_text(_PRELUDE + code, encoding="utf-8")
            env = self._environment(workdir)
            env["IRIDIUM_RESULT"] = str(result_path)

            # -I isolates the interpreter from environment variables and the
            # user site directory. -S is deliberately NOT used: it would also
            # drop site-packages, and generated physics code needs numpy. The
            # isolation that matters comes from the limits and the namespace,
            # not from crippling the import system.
            argv = [sys.executable, "-I", str(script)]
            if network_isolated:
                argv = ["unshare", "--net", "--map-root-user", *argv]

            started = time.monotonic()
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(workdir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._preexec(),
                )
            except OSError as exc:
                return SandboxResult(
                    ok=False, error=f"spawn failed: {exc}", isolation=isolation
                )
            try:
                stdout, stderr = proc.communicate(timeout=wall)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                stdout, stderr = proc.communicate()
                return SandboxResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    error=f"exceeded {wall}s wall clock",
                    isolation=isolation,
                    duration_seconds=time.monotonic() - started,
                )
            duration = time.monotonic() - started

            payload: dict[str, Any] = {}
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    return SandboxResult(
                        ok=False, stdout=stdout, stderr=stderr,
                        returncode=proc.returncode,
                        error=f"result file is not JSON: {exc}",
                        isolation=isolation, duration_seconds=duration,
                    )
            return SandboxResult(
                ok=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode,
                payload=payload,
                error="" if proc.returncode == 0 else f"exit {proc.returncode}",
                isolation=isolation,
                duration_seconds=duration,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


_PRELUDE = '''"""Injected by iridium.runtime.sandbox."""
import json as _json, os as _os

def emit(payload):
    """Return a JSON-serializable result to the caller."""
    target = _os.environ.get("IRIDIUM_RESULT")
    if target:
        with open(target, "w", encoding="utf-8") as handle:
            _json.dump(payload, handle)

'''
