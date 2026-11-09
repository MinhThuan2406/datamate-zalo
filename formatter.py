"""
Submission record format checks (rulebook §7 / §9).

A solver returns a record *without* id, e.g. ``{"type": "number", "answer": 5}``.
``validate_format`` checks it against the single §9 format of its declared type
and downgrades anything invalid to ``{"type": "abstain"}`` -- a wrong format is
scored -1, an abstain 0, so emitting a malformed answer is never worth it.

This only checks *format*, not value: it cannot know whether a number should
be an integer VND amount or whether a list has the right length.

    number / percent  JSON number (int/float, not bool), finite
    date              "YYYY-MM-DD" (a real calendar date) or "YYYY-MM"
    boolean           true / false
    string            str
    list              JSON array of scalars (str / finite number / bool)
    table             {"columns": [...], "rows": [[...], ...]}, each row a list of
                      len(columns) scalars; numeric cells are finite numbers
    unanswerable      optional "reason", which must be a str if present
    abstain           nothing else needed
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from typing import Any, Dict, Optional, Tuple

__all__ = ["ANSWER_TYPES", "RECORD_TYPES", "ABSTAIN", "check_format", "validate_format"]

ANSWER_TYPES = ("number", "percent", "date", "boolean", "string", "list", "table")
RECORD_TYPES = ANSWER_TYPES + ("unanswerable", "abstain")
ABSTAIN: Dict[str, Any] = {"type": "abstain"}

_DATE_YMD = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
_DATE_YM = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, bool)) or _is_number(x)


def _check_answer(rtype: str, ans: Any) -> Optional[str]:
    if rtype in ("number", "percent"):
        if isinstance(ans, bool) or not isinstance(ans, (int, float)):
            return f"{rtype} answer must be a JSON number, got {type(ans).__name__}: {ans!r}"
        if not math.isfinite(ans):
            return f"{rtype} answer must be finite, got {ans!r}"
        return None

    if rtype == "date":
        if not isinstance(ans, str):
            return f"date answer must be a string, got {type(ans).__name__}"
        if _DATE_YM.match(ans):
            return None
        if _DATE_YMD.match(ans):
            try:
                _dt.date.fromisoformat(ans)
            except ValueError:
                return f"date answer is not a real calendar date: {ans!r}"
            return None
        return f"date answer must be YYYY-MM-DD or YYYY-MM, got {ans!r}"

    if rtype == "boolean":
        return None if isinstance(ans, bool) else f"boolean answer must be true/false, got {ans!r}"

    if rtype == "string":
        return None if isinstance(ans, str) else f"string answer must be a str, got {type(ans).__name__}"

    if rtype == "list":
        if not isinstance(ans, list):
            return f"list answer must be a JSON array, got {type(ans).__name__}"
        for i, el in enumerate(ans):
            if not _is_scalar(el):
                return f"list element {i} must be a string or finite number, got {el!r}"
        return None

    if rtype == "table":
        if not isinstance(ans, dict):
            return f"table answer must be an object, got {type(ans).__name__}"
        cols, rows = ans.get("columns"), ans.get("rows")
        if not isinstance(cols, list) or not isinstance(rows, list):
            return "table answer needs list 'columns' and list 'rows'"
        if not cols:
            return "table 'columns' must not be empty"
        for r_idx, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != len(cols):
                return f"table row {r_idx} must be a list of {len(cols)} cells, got {row!r}"
            for c_idx, cell in enumerate(row):
                if not _is_scalar(cell):
                    return f"table cell ({r_idx}, {c_idx}) must be a string or finite number, got {cell!r}"
        return None

    return f"unknown answer type {rtype!r}"


def check_format(record: Any) -> Optional[str]:
    """Returns None if `record` is a valid §9 record for its declared type, else a reason."""
    if not isinstance(record, dict):
        return f"record must be a dict, got {type(record).__name__}"
    rtype = record.get("type")
    if rtype not in RECORD_TYPES:
        return f"unknown type {rtype!r} (expected one of {', '.join(RECORD_TYPES)})"
    if rtype == "abstain":
        return None
    if rtype == "unanswerable":
        if "reason" in record and not isinstance(record["reason"], str):
            return "unanswerable reason must be a string when present"
        return None
    if "answer" not in record:
        return f"{rtype} record is missing 'answer'"
    err = _check_answer(rtype, record["answer"])
    if err:
        return err
    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as e:
        return f"record is not JSON-serialisable: {e}"
    return None


def validate_format(record: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Returns (clean_record, reason).

    Valid records come back as a minimal copy holding only the keys that are
    scored ({"type", "answer"} / {"type", "reason"?} / {"type"}), without "id".
    Invalid records come back as {"type": "abstain"} with the reason string.
    """
    reason = check_format(record)
    if reason is not None:
        return dict(ABSTAIN), reason
    rtype = record["type"]
    if rtype == "abstain":
        return dict(ABSTAIN), None
    if rtype == "unanswerable":
        out: Dict[str, Any] = {"type": "unanswerable"}
        if "reason" in record:
            out["reason"] = record["reason"]
        return out, None
    return {"type": rtype, "answer": record["answer"]}, None
