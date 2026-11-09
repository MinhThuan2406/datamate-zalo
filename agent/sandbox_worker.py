"""
Worker process for agent.sandbox.PythonSandbox. Started as ``python -I sandbox_worker.py``.

Reads JSON-line requests from the original stdin and writes JSON-line responses to the
original stdout (both ASCII, one message per line). After startup, fd 0 points to
os.devnull and fd 1 points to stderr, so user code that reads stdin gets EOF and user
code that writes to fd 1 cannot corrupt the protocol. Python-level print output is
captured per exec.

The namespace (``_G``) lives for the whole worker lifetime, so variables survive between
execs. ``final_answer`` raises ``_FinalSignal`` (a BaseException, so ``except Exception``
in user code does not swallow it) to stop the current cell and report the answer.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import datetime as _datetime
import decimal
import io
import json
import linecache
import os
import sys
import time
import traceback

AGENT_FILENAME = "<agent>"
CAPTURE_LIMIT = 2_000_000  # characters kept per stream per exec; the host truncates further
ANSWER_TYPES = ("number", "percent", "date", "boolean", "string", "list", "table")
RECORD_TYPES = ANSWER_TYPES + ("unanswerable", "abstain")

PREAMBLE = """
import pandas as pd
import numpy as np
import duckdb
import sqlite3
import json
import re
import datetime as dt
import os
import glob
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.max_rows', 100)
"""


class _FinalSignal(BaseException):
    """Raised by final_answer() to end the current cell."""


_G: dict = {"__name__": "__main__", "__builtins__": builtins}
_STATE: dict = {"final": None}


class _Capture(io.TextIOBase):
    """Keeps up to CAPTURE_LIMIT characters; counts what was dropped."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list = []
        self._size = 0
        self._dropped = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        s = s if isinstance(s, str) else str(s)
        room = CAPTURE_LIMIT - self._size
        if room > 0:
            chunk = s[:room]
            self._parts.append(chunk)
            self._size += len(chunk)
            self._dropped += len(s) - len(chunk)
        else:
            self._dropped += len(s)
        return len(s)

    def getvalue(self) -> str:
        text = "".join(self._parts)
        if self._dropped:
            text += f"\n...[{self._dropped} more characters dropped]"
        return text


# -----------------------------------------------------------------------------
# Answer conversion
# -----------------------------------------------------------------------------


def _jsonable(obj, where: str = "answer"):
    """Converts numpy / pandas / Decimal / date values to plain JSON types; rejects NaN and inf."""
    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not (value == value and value not in (float("inf"), float("-inf"))):
            raise ValueError(f"{where}: NaN/inf is not a valid answer; use abstain or fix the computation")
        return value
    if isinstance(obj, decimal.Decimal):
        if not obj.is_finite():
            raise ValueError(f"{where}: NaN/inf is not a valid answer")
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if obj is pd.NA or obj is pd.NaT:
        raise ValueError(f"{where}: missing value (NA/NaT) is not a valid answer")
    if isinstance(obj, _datetime.datetime):  # includes pd.Timestamp
        return obj.date().isoformat()
    if isinstance(obj, _datetime.date):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return {
            "columns": [str(c) for c in obj.columns],
            "rows": [[_jsonable(v, f"{where}.rows[{i}][{j}]") for j, v in enumerate(row)]
                     for i, row in enumerate(obj.itertuples(index=False, name=None))],
        }
    if isinstance(obj, (pd.Series, np.ndarray)):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(obj.tolist())]
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, f"{where}[{k!r}]") for k, v in obj.items()}
    raise TypeError(f"{where}: unsupported type {builtins.type(obj).__name__}; "
                    "convert it to str/int/float/bool/list/dict first")


def final_answer(type, answer=None, reason=None):  # noqa: A002 - name fixed by the protocol
    """Records the final answer and ends the episode. type: number, percent, date, boolean,
    string, list, table, unanswerable or abstain."""
    if not isinstance(type, str) or type not in RECORD_TYPES:
        raise ValueError(f"final_answer: type must be one of {', '.join(RECORD_TYPES)}; got {type!r}")
    if type == "abstain":
        record = {"type": "abstain"}
    elif type == "unanswerable":
        record = {"type": "unanswerable"}
        if reason is not None:
            record["reason"] = str(reason)
    else:
        if answer is None:
            raise ValueError(f"final_answer: type {type!r} needs an answer (use 'abstain' to give up)")
        record = {"type": type, "answer": _jsonable(answer)}
    json.dumps(record, ensure_ascii=True, allow_nan=False)  # final guard
    _STATE["final"] = record
    raise _FinalSignal()


# -----------------------------------------------------------------------------
# Cell execution
# -----------------------------------------------------------------------------


def _run_cell(code: str) -> None:
    """Executes a cell in _G. A trailing bare expression is evaluated and its repr printed."""
    linecache.cache[AGENT_FILENAME] = (len(code), None, code.splitlines(True), AGENT_FILENAME)
    tree = ast.parse(code, filename=AGENT_FILENAME, mode="exec")
    trailing = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = ast.Expression(body=tree.body.pop().value)
    exec(compile(tree, AGENT_FILENAME, "exec"), _G)
    if trailing is not None:
        value = eval(compile(trailing, AGENT_FILENAME, "eval"), _G)
        if value is not None:
            print(repr(value))


def _format_exc(e: BaseException) -> str:
    tb = e.__traceback__
    # Drop the worker's own frames above the user code, keep everything from the cell onwards.
    while tb is not None and tb.tb_frame.f_code.co_filename != AGENT_FILENAME:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(e), e, tb))


def _exec(code: str) -> dict:
    _STATE["final"] = None
    out, err = _Capture(), _Capture()
    error = None
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            _run_cell(code)
        except _FinalSignal:
            pass
        except BaseException as e:  # noqa: BLE001 - everything from user code is reported
            error = _format_exc(e)
    return {
        "ok": True,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "error": error,
        "final": _STATE["final"],
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def _init(req: dict) -> dict:
    try:
        exec(compile(PREAMBLE, "<preamble>", "exec"), _G)
        _G.update({
            "final_answer": final_answer,
            "DATA_DIR": req["data_dir"],
            "AS_OF_DATE": req["as_of_date"],
            "AS_OF": _datetime.date.fromisoformat(req["as_of_date"]),
            "CALENDAR_PATH": req.get("calendar_path"),
            "RULEBOOK_PATH": req.get("rulebook_path"),
        })
        return {"ok": True}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "error": _format_exc(e)}


def main() -> None:
    proto_in = os.fdopen(os.dup(0), "r", encoding="utf-8", newline="\n")
    proto_out = os.fdopen(os.dup(1), "w", encoding="ascii", newline="\n")
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.close(null_fd)
    os.dup2(2, 1)  # anything written to fd 1 from now on goes to stderr
    sys.stdin = open(os.devnull, "r", encoding="utf-8")

    for raw in proto_in:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            resp = {"ok": False, "error": "malformed request"}
        else:
            op = req.get("op")
            if op == "init":
                resp = _init(req)
            elif op == "exec":
                resp = _exec(str(req.get("code", "")))
            elif op == "shutdown":
                break
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        proto_out.write(json.dumps(resp, ensure_ascii=True) + "\n")
        proto_out.flush()


if __name__ == "__main__":
    main()
