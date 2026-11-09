"""
Persistent Python sandbox for the CodeAct solver: one worker subprocess per question.

Variables survive between exec() calls, so a DataFrame loaded in turn 1 is still there in
turn 5. A per-exec timeout kills the worker (its state is lost) and the next exec starts a
fresh worker. The worker is started as ``<venv python> -I sandbox_worker.py`` with cwd set to
the shop's data directory and a minimal environment (no OPENROUTER_API_KEY).

Protocol: see the docstring of sandbox_worker.py.

Not a security boundary: the code runs with the user's permissions and may use the network
and read or write any file. The sandbox isolates state, the environment and run time only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

WORKER_PATH = Path(__file__).with_name("sandbox_worker.py")
DEFAULT_EXEC_TIMEOUT_S = 60.0
STARTUP_TIMEOUT_S = 120.0
OUTPUT_LIMIT = 6000
_ENV_PASSTHROUGH = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "LANG", "LC_ALL",
)


class SandboxError(RuntimeError):
    """The worker could not be started or is unusable (infrastructure failure, not a model error)."""


def minimal_env(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Only the variables the interpreter needs. Secrets such as OPENROUTER_API_KEY are never passed."""
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if extra:
        env.update(extra)
    return env


def truncate_output(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Keeps the head (3/5 of the budget) and the tail (the rest, where tracebacks end)."""
    if len(text) <= limit:
        return text
    head = limit * 3 // 5
    tail = limit - head - 60  # room for the marker
    omitted = len(text) - head - tail
    return text[:head] + f"\n...[{omitted} characters omitted]...\n" + text[len(text) - tail:]


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None          # traceback or error message; None when the cell ran cleanly
    final: Optional[Dict[str, Any]] = None  # set when final_answer() was called in this cell
    timed_out: bool = False
    crashed: bool = False
    elapsed_s: float = 0.0


def _pump_stdout(stream, lines: "queue.Queue[Optional[bytes]]") -> None:
    try:
        while True:
            line = stream.readline()
            if not line:
                break
            lines.put(line)
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)  # EOF sentinel


def _pump_stderr(stream, tail: Deque[str]) -> None:
    try:
        for raw in iter(stream.readline, b""):
            tail.append(raw.decode("utf-8", "replace").rstrip("\n"))
    except (OSError, ValueError):
        pass


class PythonSandbox:
    """Context manager around one worker process. Not shared between questions."""

    def __init__(
        self,
        data_dir: Path,
        *,
        as_of_date: str,
        calendar_path: Optional[Path] = None,
        rulebook_path: Optional[Path] = None,
        timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
        startup_timeout_s: float = STARTUP_TIMEOUT_S,
        python: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data dir not found: {self.data_dir}")
        dt.date.fromisoformat(as_of_date)  # validates YYYY-MM-DD early
        self._init_msg = {
            "op": "init",
            "data_dir": str(self.data_dir),
            "as_of_date": as_of_date,
            "calendar_path": str(calendar_path) if calendar_path and Path(calendar_path).is_file() else None,
            "rulebook_path": str(rulebook_path) if rulebook_path and Path(rulebook_path).is_file() else None,
        }
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.python = python or sys.executable
        self.env = minimal_env(env)
        self.restarts = 0
        self.execs = 0
        self.closed = False
        self._proc: Optional[subprocess.Popen] = None
        self._lines: Optional["queue.Queue[Optional[bytes]]"] = None
        self._stderr_tail: Deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()

    # -- public -------------------------------------------------------------

    def __enter__(self) -> "PythonSandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def exec(self, code: str, timeout_s: Optional[float] = None) -> ExecResult:
        if self.closed:
            raise SandboxError("sandbox is closed")
        timeout = self.timeout_s if timeout_s is None else timeout_s
        with self._lock:
            t0 = time.perf_counter()
            if self._proc is not None and self._proc.poll() is not None:
                self._kill()  # worker died between execs
            if self._proc is None:
                self._start()
            self.execs += 1
            self._stderr_tail.clear()
            status, payload = self._request({"op": "exec", "code": code}, timeout)
            elapsed = time.perf_counter() - t0

            if status == "timeout":
                self._kill()
                self.restarts += 1
                return ExecResult(
                    error=(f"TimeoutError: the code did not finish within {timeout:g} seconds and was stopped. "
                           "The Python session was reset: variables from earlier turns are gone, "
                           "so reload any data you need."),
                    timed_out=True, elapsed_s=elapsed)
            if status == "crashed":
                exit_code = self._exit_code()
                stderr_tail = "\n".join(list(self._stderr_tail)[-20:])
                self._kill()
                self.restarts += 1
                msg = (f"ProcessCrashed: the Python worker exited unexpectedly (exit code {exit_code}). "
                       "The session was reset: variables from earlier turns are gone.")
                if stderr_tail:
                    msg += f"\n{stderr_tail}"
                return ExecResult(error=truncate_output(msg), crashed=True, elapsed_s=elapsed)

            stdout = truncate_output(str(payload.get("stdout") or ""))
            stderr = truncate_output(str(payload.get("stderr") or ""))
            error = payload.get("error")
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                error=truncate_output(error) if error else None,
                final=payload.get("final"),
                elapsed_s=elapsed,
            )

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.stdin is not None:
                        self._proc.stdin.write(b'{"op": "shutdown"}\n')
                        self._proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                self._kill()
            self.closed = True

    # -- internals ----------------------------------------------------------

    def _start(self) -> None:
        try:
            proc = subprocess.Popen(
                [self.python, "-I", str(WORKER_PATH)],
                cwd=str(self.data_dir),
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            raise SandboxError(f"could not launch {self.python!r}: {e}") from e
        lines: "queue.Queue[Optional[bytes]]" = queue.Queue()
        threading.Thread(target=_pump_stdout, args=(proc.stdout, lines), daemon=True,
                         name="sandbox-stdout").start()
        threading.Thread(target=_pump_stderr, args=(proc.stderr, self._stderr_tail), daemon=True,
                         name="sandbox-stderr").start()
        self._proc, self._lines = proc, lines
        status, payload = self._request(self._init_msg, self.startup_timeout_s)
        if status != "ok" or not payload or not payload.get("ok"):
            detail = (payload or {}).get("error") or "\n".join(list(self._stderr_tail)[-20:])
            self._kill()
            raise SandboxError(f"python worker failed to start ({status}): {detail}")

    def _request(self, msg: Dict[str, Any], timeout: float) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Returns ("ok", payload) | ("timeout", None) | ("crashed", None)."""
        proc, lines = self._proc, self._lines
        assert proc is not None and lines is not None
        try:
            proc.stdin.write((json.dumps(msg, ensure_ascii=True) + "\n").encode("ascii"))
            proc.stdin.flush()
        except (OSError, ValueError):
            return "crashed", None
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            return "timeout", None
        if line is None:
            return "crashed", None
        try:
            payload = json.loads(line.decode("utf-8"))
        except ValueError:
            return "crashed", None
        return "ok", payload

    def _exit_code(self) -> Any:
        proc = self._proc
        if proc is None:
            return None
        try:
            return proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return None

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        self._lines = None
        if proc is None:
            return
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


__all__ = ["PythonSandbox", "ExecResult", "SandboxError", "truncate_output", "minimal_env",
           "WORKER_PATH", "OUTPUT_LIMIT"]
