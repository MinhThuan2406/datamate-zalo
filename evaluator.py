"""
DataMate Evaluator — Section 9 & Section 7 of Rulebook
Deterministic evaluation module for DataMate submissions against ground truth answers.

Rules summary:
  - Correct: +1
  - Incorrect: -1
  - Abstain: 0 (explicit {"type": "abstain"} or missing row)
  - unanswerable:
      - Valid if reason is omitted or is a string.
      - If reason is present but not a string: treated as unreadable / abstain (0 points).
      - If GT is unanswerable and pred is valid unanswerable: +1.
      - If GT is unanswerable and pred answers something else: -1 (should_be_unanswerable).
      - If GT is answerable and pred answers unanswerable: -1 (false_unanswerable).
  - Type & Format validation before value comparison:
      - number: JSON number (int/float, not bool), exact match (float tolerance 1e-9 / 1e-6).
      - percent: JSON number (int/float, not bool), exact to 2 decimal places.
      - date: "YYYY-MM-DD" or "YYYY-MM".
      - boolean: True or False (not bool string or 1/0).
      - string: Case-insensitive, whitespace/punctuation-insensitive; accents must match; aliases accepted.
      - list: Ordered JSON array; all elements match; per-element aliases accepted.
      - table: {"columns": [...], "rows": [[...], ...]}; numeric cells are JSON numbers;
               order of rows does not matter (1-to-1 matching); column matching by name or position.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# The intent classifier lives in intent.py; re-exported here so that
# `from evaluator import classify_intent` keeps working.
from intent import classify_intent  # noqa: F401


# -----------------------------------------------------------------------------
# String Normalization & Matching
# -----------------------------------------------------------------------------

# ASCII symbols (Unicode category S*, not P*) that are also treated as punctuation.
# Every character whose category starts with "P" (ASCII or Unicode: – — “ ” ‘ ’ … « » ·)
# is stripped as well; see normalize_text.
_EXTRA_STRIP_CHARS = frozenset("+*~^=|<>#@&/\\`")


def _is_strippable(ch: str) -> bool:
    return ch in _EXTRA_STRIP_CHARS or unicodedata.category(ch).startswith("P")


def normalize_text(s: str) -> str:
    """
    Normalizes a string for §9 comparison: case-, whitespace- and
    punctuation-insensitive, but Vietnamese diacritics must still match.

    Steps: NFC (so NFD-encoded input compares equal to NFC ground truth),
    casefold, NFC again (casefold can emit decomposed sequences), replace every
    Unicode punctuation char (category P*) and a few ASCII symbols with a space,
    collapse whitespace. Diacritics are never removed ("dầu" != "dâu").
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", unicodedata.normalize("NFC", s).casefold())
    cleaned = "".join(" " if _is_strippable(ch) else ch for ch in s)
    return " ".join(cleaned.split())


def match_string(pred: Any, gt: Any, aliases: Optional[Sequence[Any]] = None) -> bool:
    """
    Matches a string prediction against ground truth and candidate aliases.
    Strict on Vietnamese accents, insensitive to case, punctuation, and extra whitespace.
    """
    if not isinstance(pred, str):
        return False

    nfc_pred = unicodedata.normalize("NFC", pred)
    norm_pred = normalize_text(pred)
    candidates = [gt]
    if aliases:
        candidates.extend(aliases)

    for c in candidates:
        if isinstance(c, str):
            if nfc_pred == unicodedata.normalize("NFC", c) or norm_pred == normalize_text(c):
                return True
        elif isinstance(c, (int, float)) and not isinstance(c, bool):
            # Rare case where alias might be numeric representation
            if norm_pred == str(c):
                return True
    return False


# -----------------------------------------------------------------------------
# Maximum Bipartite Matching for Tables
# -----------------------------------------------------------------------------

def max_bipartite_matching(graph: List[List[int]]) -> int:
    """
    Computes maximum bipartite matching size using DFS augmenting paths.
    `graph[u]` contains indices of `v` nodes connected to `u`.
    """
    n = len(graph)
    match_v: Dict[int, int] = {}

    def dfs(u: int, visited: Set[int]) -> bool:
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                if v not in match_v or dfs(match_v[v], visited):
                    match_v[v] = u
                    return True
        return False

    result = 0
    for u in range(n):
        if dfs(u, set()):
            result += 1
    return result


def get_column_mapping(gt_cols: List[str], pred_cols: List[str]) -> List[int]:
    """
    Maps each GT column index to the corresponding Pred column index.
    Prioritizes column name matching (case/whitespace/underscore-insensitive).
    Falls back to positional mapping if names do not match bijectively.
    """
    if len(gt_cols) != len(pred_cols):
        return []

    def norm_c(col: Any) -> str:
        return re.sub(r'[\s_]+', '', unicodedata.normalize("NFC", str(col)).casefold())

    norm_pred_cols = [norm_c(c) for c in pred_cols]
    norm_gt_cols = [norm_c(c) for c in gt_cols]

    col_map: List[int] = []
    used_pred: Set[int] = set()
    name_match_possible = True

    for gt_c in norm_gt_cols:
        matched_idx = None
        for j, pred_c in enumerate(norm_pred_cols):
            if j not in used_pred and gt_c == pred_c:
                matched_idx = j
                used_pred.add(j)
                break
        if matched_idx is None:
            name_match_possible = False
            break
        col_map.append(matched_idx)

    if name_match_possible and len(col_map) == len(gt_cols):
        return col_map
    else:
        # Fallback to positional mapping
        return list(range(len(gt_cols)))


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

ERROR_TYPES = (
    "wrong_type",
    "format_error",
    "rounding_error",          # numeric miss within rounding: number |diff| <= 1, percent |diff| <= 0.01
    "close_value",             # numeric miss < 1% relative (or percent |diff| <= 0.05)
    "value_mismatch",
    "should_be_unanswerable",
    "false_unanswerable",
)

_SEVERITY = {"rounding_error": 0, "close_value": 1, "value_mismatch": 2}
_FLOAT_EPS = 1e-9


def numbers_match(pred: Any, gt: Any) -> bool:
    """Exact numeric match, ignoring only floating-point representation error."""
    return pred == gt or math.isclose(pred, gt, rel_tol=1e-9, abs_tol=1e-6)


def numeric_error_label(pred: float, gt: float, kind: str = "number") -> str:
    """
    Classifies a numeric miss (pred != gt) for the error taxonomy.

      kind="number":  |diff| <= 1                     -> rounding_error
                      |diff| / |gt| < 1%              -> close_value
      kind="percent": |diff| <= 0.01                  -> rounding_error
                      |diff| <= 0.05 or rel < 1%      -> close_value
      otherwise                                       -> value_mismatch

    For list elements / table cells the GT value decides the kind: an integer GT
    uses "number" thresholds (money, counts), a non-integer float GT uses
    "percent" thresholds (2-decimal percentages and averages).
    """
    diff = abs(pred - gt)
    rel = diff / abs(gt) if gt else math.inf
    if kind == "percent":
        if diff <= 0.01 + _FLOAT_EPS:
            return "rounding_error"
        if diff <= 0.05 + _FLOAT_EPS or rel < 0.01:
            return "close_value"
        return "value_mismatch"
    if diff <= 1.0 + _FLOAT_EPS:
        return "rounding_error"
    if rel < 0.01:
        return "close_value"
    return "value_mismatch"


def _element_kind(gt: Any) -> str:
    return "percent" if isinstance(gt, float) and not float(gt).is_integer() else "number"


def _is_json_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _worst_label(labels: Iterable[str]) -> str:
    labels = list(labels)
    if not labels:
        return "value_mismatch"
    return max(labels, key=lambda l: _SEVERITY.get(l, 2))


@dataclass
class EvalRowResult:
    id: str
    status: str                    # "correct", "incorrect", "abstain"
    score: int                     # +1, -1, 0
    gt_type: str
    pred_type: Optional[str] = None
    error_type: Optional[str] = None  # None if correct or abstain; otherwise one of
                                      # ERROR_TYPES (see numeric_error_label for the
                                      # rounding_error / close_value split)
    details: str = ""
    shop: str = "unknown"
    intent: str = "khac"


@dataclass
class EvalReport:
    total_questions: int
    correct: int
    incorrect: int
    abstain: int
    total_score: int
    max_possible_score: int
    coverage_rate: float
    accuracy_on_answered: float
    expected_score_per_answered: float
    by_shop: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_intent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_intent_shop_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_taxonomy: Dict[str, int] = field(default_factory=dict)
    # Problems found while loading the submission; empty when clean. Keys (only
    # present when non-empty): "duplicate_ids" (sorted list; last row wins),
    # "unknown_ids" (sorted list of ids not in ground truth, ignored),
    # "invalid_lines" (count of unparsable / non-object / id-less rows) and
    # "invalid_line_numbers" (first 20 1-based line or list positions).
    submission_warnings: Dict[str, Any] = field(default_factory=dict)
    row_results: List[EvalRowResult] = field(default_factory=list)

    def to_dict(self, include_rows: bool = True) -> Dict[str, Any]:
        """JSON-serialisable dict of the full report (single source of truth for report JSON)."""
        data = asdict(self)
        if not include_rows:
            data.pop("row_results", None)
        return data

    def summary_table(self) -> str:
        """Returns human-readable text report."""
        lines = []
        lines.append("=" * 78)
        lines.append("                     DATAMATE EVALUATION REPORT")
        lines.append("=" * 78)
        lines.append(f"Total Questions:              {self.total_questions:8d}")
        lines.append(f"Correct (+1):                 {self.correct:8d} ({self.correct/max(self.total_questions,1)*100:6.2f}%)")
        lines.append(f"Incorrect (-1):               {self.incorrect:8d} ({self.incorrect/max(self.total_questions,1)*100:6.2f}%)")
        lines.append(f"Abstain (0):                  {self.abstain:8d} ({self.abstain/max(self.total_questions,1)*100:6.2f}%)")
        lines.append(f"Total Score:                  {self.total_score:8d} / {self.max_possible_score}")
        lines.append(f"Coverage Rate:                {self.coverage_rate * 100:8.2f}%")
        lines.append(f"Accuracy on Answered (p):     {self.accuracy_on_answered * 100:8.2f}%")
        lines.append(f"Expected Score/Answer (2p-1): {self.expected_score_per_answered:8.4f}")
        lines.append("-" * 78)

        if self.submission_warnings:
            lines.append("\n[ WARNINGS ]")
            lines.extend(f"  - {w}" for w in format_submission_warnings(self.submission_warnings))
            lines.append("-" * 78)

        # Error Taxonomy
        lines.append("\n[ ERROR TAXONOMY ]")
        total_errors = sum(self.error_taxonomy.values())
        lines.append(f"{'Error Type':<35} | {'Count':<8} | {'% of Errors':<12}")
        lines.append("-" * 60)
        for err_k, err_v in sorted(self.error_taxonomy.items(), key=lambda x: -x[1]):
            pct = (err_v / max(total_errors, 1)) * 100
            lines.append(f"{err_k:<35} | {err_v:<8d} | {pct:6.2f}%")
        if total_errors == 0:
            lines.append("No errors! Perfect score.")
        lines.append("-" * 60)

        # Breakdown by Type
        lines.append("\n[ BREAKDOWN BY TYPE ]")
        lines.append(f"{'Type':<15} | {'Total':<6} | {'Correct':<7} | {'Wrong':<6} | {'Abstain':<7} | {'Score':<6} | {'Accuracy':<8}")
        lines.append("-" * 68)
        for t, d in sorted(self.by_type.items()):
            acc_str = f"{d['accuracy']*100:6.2f}%" if (d['correct'] + d['incorrect']) > 0 else "N/A"
            lines.append(f"{t:<15} | {d['total']:<6d} | {d['correct']:<7d} | {d['incorrect']:<6d} | {d['abstain']:<7d} | {d['score']:<6d} | {acc_str:<8}")

        # Breakdown by Intent
        lines.append("\n[ BREAKDOWN BY INTENT ]")
        lines.append(f"{'Intent':<22} | {'Total':<6} | {'Correct':<7} | {'Wrong':<6} | {'Abstain':<7} | {'Score':<6} | {'Accuracy':<8}")
        lines.append("-" * 75)
        for intent, d in sorted(self.by_intent.items()):
            acc_str = f"{d['accuracy']*100:6.2f}%" if (d['correct'] + d['incorrect']) > 0 else "N/A"
            lines.append(f"{intent:<22} | {d['total']:<6d} | {d['correct']:<7d} | {d['incorrect']:<6d} | {d['abstain']:<7d} | {d['score']:<6d} | {acc_str:<8}")

        # Breakdown by Shop (Top 5 lowest scoring & summary)
        lines.append("\n[ BREAKDOWN BY SHOP (30 Shops summary) ]")
        lines.append(f"{'Shop':<12} | {'Total':<6} | {'Correct':<7} | {'Wrong':<6} | {'Abstain':<7} | {'Score':<6} | {'Accuracy':<8}")
        lines.append("-" * 65)
        for shop, d in sorted(self.by_shop.items()):
            acc_str = f"{d['accuracy']*100:6.2f}%" if (d['correct'] + d['incorrect']) > 0 else "N/A"
            lines.append(f"{shop:<12} | {d['total']:<6d} | {d['correct']:<7d} | {d['incorrect']:<6d} | {d['abstain']:<7d} | {d['score']:<6d} | {acc_str:<8}")
        lines.append("=" * 78)
        return "\n".join(lines)


def format_submission_warnings(warnings: Dict[str, Any], max_ids: int = 5) -> List[str]:
    """Short human-readable lines describing submission_warnings."""
    out: List[str] = []

    def ids_preview(ids: Sequence[str]) -> str:
        head = ", ".join(ids[:max_ids])
        return head + (f", ... (+{len(ids) - max_ids} more)" if len(ids) > max_ids else "")

    if warnings.get("duplicate_ids"):
        ids = warnings["duplicate_ids"]
        out.append(f"{len(ids)} duplicate id(s) in submission (last row wins): {ids_preview(ids)}")
    if warnings.get("unknown_ids"):
        ids = warnings["unknown_ids"]
        out.append(f"{len(ids)} id(s) not in ground truth (ignored): {ids_preview(ids)}")
    if warnings.get("invalid_lines"):
        nums = ", ".join(str(n) for n in warnings.get("invalid_line_numbers", [])[:max_ids])
        out.append(f"{warnings['invalid_lines']} invalid row(s) skipped (bad JSON / not an object / no id)"
                   + (f" at: {nums}" if nums else ""))
    return out


def save_report_json(report: EvalReport, path: Union[str, Path], include_rows: bool = True) -> Path:
    """Writes the report as JSON (used by both evaluator.py and run.py)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(include_rows=include_rows), f, ensure_ascii=False, indent=2)
    return out_path


# -----------------------------------------------------------------------------
# Evaluator Engine
# -----------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates predictions against ground truth according to Section 7 & Section 9 of DataMate rulebook.
    """

    def __init__(
        self,
        answers: Union[str, Path, Sequence[Dict[str, Any]]],
        questions: Optional[Union[str, Path, Sequence[Dict[str, Any]]]] = None,
    ):
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        self.question_meta: Dict[str, Dict[str, Any]] = {}

        # 1. Load answers
        if isinstance(answers, (str, Path)):
            ans_path = Path(answers)
            if not ans_path.is_file():
                raise FileNotFoundError(f"Answers file not found: {ans_path}")
            with open(ans_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        self.ground_truth[record["id"]] = record
            
            # If questions not explicitly given, look in same directory
            if questions is None:
                sibling_q = ans_path.parent / "questions.jsonl"
                if sibling_q.is_file():
                    questions = sibling_q
        else:
            for record in answers:
                self.ground_truth[record["id"]] = record

        # 2. Load questions metadata if available
        if questions is not None:
            if isinstance(questions, (str, Path)):
                q_path = Path(questions)
                if q_path.is_file():
                    with open(q_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                r = json.loads(line)
                                self.question_meta[r["id"]] = {
                                    "business_id": r.get("business_id", "unknown"),
                                    "question": r.get("question", ""),
                                    "intent": classify_intent(r.get("question", "")),
                                }
            else:
                for r in questions:
                    self.question_meta[r["id"]] = {
                        "business_id": r.get("business_id", "unknown"),
                        "question": r.get("question", ""),
                        "intent": classify_intent(r.get("question", "")),
                    }

    def evaluate_row(
        self,
        q_id: str,
        pred_dict: Optional[Dict[str, Any]],
        gt_dict: Dict[str, Any],
    ) -> EvalRowResult:
        """Evaluates a single question prediction against ground truth."""
        meta = self.question_meta.get(q_id, {})
        shop = meta.get("business_id", "unknown")
        intent = meta.get("intent", "khac")

        gt_type = gt_dict["type"]
        gt_ans = gt_dict.get("answer")
        gt_alias = gt_dict.get("alias", [])

        # Case 1: Missing prediction or explicit abstain
        if pred_dict is None or pred_dict.get("type") == "abstain":
            return EvalRowResult(
                id=q_id,
                status="abstain",
                score=0,
                gt_type=gt_type,
                pred_type="abstain" if pred_dict else None,
                details="Abstain or missing row",
                shop=shop,
                intent=intent,
            )

        pred_type = pred_dict.get("type")

        # Case 2: Prediction claims unanswerable
        if pred_type == "unanswerable":
            # Section 7 Rule:
            # "Trường reason không bắt buộc và không chấm tự động. Nếu có, reason phải là chuỗi;
            #  dòng có reason không phải chuỗi không đọc được và tính như bỏ trống."
            if "reason" in pred_dict and not isinstance(pred_dict["reason"], str):
                return EvalRowResult(
                    id=q_id,
                    status="abstain",
                    score=0,
                    gt_type=gt_type,
                    pred_type=pred_type,
                    error_type=None,
                    details="Unanswerable row with non-string reason is unreadable, counts as abstain (0 pts)",
                    shop=shop,
                    intent=intent,
                )

            # Valid unanswerable prediction
            if gt_type == "unanswerable":
                return EvalRowResult(
                    id=q_id,
                    status="correct",
                    score=1,
                    gt_type=gt_type,
                    pred_type=pred_type,
                    details="Correctly identified as unanswerable",
                    shop=shop,
                    intent=intent,
                )
            else:
                return EvalRowResult(
                    id=q_id,
                    status="incorrect",
                    score=-1,
                    gt_type=gt_type,
                    pred_type=pred_type,
                    error_type="false_unanswerable",
                    details=f"Predicted unanswerable but ground truth was answerable ({gt_type})",
                    shop=shop,
                    intent=intent,
                )

        # Case 3: Ground truth is unanswerable, but prediction answered something else
        if gt_type == "unanswerable":
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type=gt_type,
                pred_type=pred_type,
                error_type="should_be_unanswerable",
                details=f"Predicted {pred_type} but ground truth is unanswerable",
                shop=shop,
                intent=intent,
            )

        # Case 4: Answerable question - verify prediction type matches GT type
        if pred_type != gt_type:
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type=gt_type,
                pred_type=pred_type,
                error_type="wrong_type",
                details=f"Predicted type '{pred_type}' does not match GT type '{gt_type}'",
                shop=shop,
                intent=intent,
            )

        # Verify presence of 'answer' field
        if "answer" not in pred_dict:
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type=gt_type,
                pred_type=pred_type,
                error_type="format_error",
                details="Missing 'answer' field in prediction",
                shop=shop,
                intent=intent,
            )

        pred_val = pred_dict["answer"]

        # Case 5: Evaluate each specific type
        if gt_type == "number":
            return self._eval_number(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        elif gt_type == "percent":
            return self._eval_percent(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        elif gt_type == "date":
            return self._eval_date(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        elif gt_type == "boolean":
            return self._eval_boolean(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        elif gt_type == "string":
            return self._eval_string(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        elif gt_type == "list":
            return self._eval_list(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        elif gt_type == "table":
            return self._eval_table(q_id, pred_val, gt_ans, gt_alias, shop, intent)

        else:
            # Unknown type in dataset
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type=gt_type,
                pred_type=pred_type,
                error_type="wrong_type",
                details=f"Unrecognized ground truth type '{gt_type}'",
                shop=shop,
                intent=intent,
            )

    # -------------------------------------------------------------------------
    # Type-Specific Evaluators
    # -------------------------------------------------------------------------

    def _eval_number(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        # Format check: must be int or float, but NOT bool
        if not isinstance(pred_val, (int, float)) or isinstance(pred_val, bool) or math.isnan(pred_val) or math.isinf(pred_val):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="number",
                pred_type="number",
                error_type="format_error",
                details=f"Invalid number format: {repr(pred_val)} (type={type(pred_val).__name__})",
                shop=shop,
                intent=intent,
            )

        # Check exact equality or float tolerance
        matched = False
        candidates = [gt_ans] + list(gt_alias)
        for c in candidates:
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                if pred_val == c or math.isclose(pred_val, c, rel_tol=1e-9, abs_tol=1e-6):
                    matched = True
                    break

        if matched:
            return EvalRowResult(
                id=q_id,
                status="correct",
                score=1,
                gt_type="number",
                pred_type="number",
                details=f"Matched number {pred_val} == {gt_ans}",
                shop=shop,
                intent=intent,
            )

        # Not matched: classify the miss (rounding_error / close_value / value_mismatch)
        diff = abs(pred_val - gt_ans)
        err_type = numeric_error_label(pred_val, gt_ans, "number")

        return EvalRowResult(
            id=q_id,
            status="incorrect",
            score=-1,
            gt_type="number",
            pred_type="number",
            error_type=err_type,
            details=f"Number mismatch: predicted {pred_val}, expected {gt_ans} (diff={diff})",
            shop=shop,
            intent=intent,
        )

    def _eval_percent(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        # Format check: must be int or float, but NOT bool
        if not isinstance(pred_val, (int, float)) or isinstance(pred_val, bool) or math.isnan(pred_val) or math.isinf(pred_val):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="percent",
                pred_type="percent",
                error_type="format_error",
                details=f"Invalid percent format: {repr(pred_val)} (type={type(pred_val).__name__})",
                shop=shop,
                intent=intent,
            )

        # Percent check: exact match (only ignore floating-point representation imprecision).
        # Do NOT use Python round(pred_val, 2) here, as unrounded 3-decimal predictions
        # (e.g. 12.404 vs 12.4) must be rejected per Section 9.
        matched = False
        candidates = [gt_ans] + list(gt_alias)
        for c in candidates:
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                if pred_val == c or math.isclose(pred_val, c, rel_tol=1e-9, abs_tol=1e-6):
                    matched = True
                    break

        if matched:
            return EvalRowResult(
                id=q_id,
                status="correct",
                score=1,
                gt_type="percent",
                pred_type="percent",
                details=f"Matched percent {pred_val} == {gt_ans}",
                shop=shop,
                intent=intent,
            )

        diff = abs(pred_val - gt_ans)
        err_type = numeric_error_label(pred_val, gt_ans, "percent")

        return EvalRowResult(
            id=q_id,
            status="incorrect",
            score=-1,
            gt_type="percent",
            pred_type="percent",
            error_type=err_type,
            details=f"Percent mismatch: predicted {pred_val}, expected {gt_ans} (diff={diff})",
            shop=shop,
            intent=intent,
        )

    def _eval_date(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        # Format check: string matching YYYY-MM-DD or YYYY-MM
        if not isinstance(pred_val, str):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="date",
                pred_type="date",
                error_type="format_error",
                details=f"Date is not string: {repr(pred_val)}",
                shop=shop,
                intent=intent,
            )

        is_ymd = re.match(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$', pred_val)
        is_ym = re.match(r'^\d{4}-(?:0[1-9]|1[0-2])$', pred_val)
        if not (is_ymd or is_ym):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="date",
                pred_type="date",
                error_type="format_error",
                details=f"Invalid date format (must be YYYY-MM-DD or YYYY-MM): '{pred_val}'",
                shop=shop,
                intent=intent,
            )

        candidates = [gt_ans] + list(gt_alias)
        if pred_val in candidates:
            return EvalRowResult(
                id=q_id,
                status="correct",
                score=1,
                gt_type="date",
                pred_type="date",
                details=f"Matched date {pred_val} == {gt_ans}",
                shop=shop,
                intent=intent,
            )

        return EvalRowResult(
            id=q_id,
            status="incorrect",
            score=-1,
            gt_type="date",
            pred_type="date",
            error_type="value_mismatch",
            details=f"Date mismatch: predicted '{pred_val}', expected '{gt_ans}'",
            shop=shop,
            intent=intent,
        )

    def _eval_boolean(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        # Format check: must be boolean
        if not isinstance(pred_val, bool):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="boolean",
                pred_type="boolean",
                error_type="format_error",
                details=f"Invalid boolean format (must be True/False JSON boolean): {repr(pred_val)}",
                shop=shop,
                intent=intent,
            )

        if pred_val == gt_ans:
            return EvalRowResult(
                id=q_id,
                status="correct",
                score=1,
                gt_type="boolean",
                pred_type="boolean",
                details=f"Matched boolean {pred_val}",
                shop=shop,
                intent=intent,
            )

        return EvalRowResult(
            id=q_id,
            status="incorrect",
            score=-1,
            gt_type="boolean",
            pred_type="boolean",
            error_type="value_mismatch",
            details=f"Boolean mismatch: predicted {pred_val}, expected {gt_ans}",
            shop=shop,
            intent=intent,
        )

    def _eval_string(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        if not isinstance(pred_val, str):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="string",
                pred_type="string",
                error_type="format_error",
                details=f"String answer must be a string: {repr(pred_val)}",
                shop=shop,
                intent=intent,
            )

        if match_string(pred_val, gt_ans, gt_alias):
            return EvalRowResult(
                id=q_id,
                status="correct",
                score=1,
                gt_type="string",
                pred_type="string",
                details=f"Matched string '{pred_val}'",
                shop=shop,
                intent=intent,
            )

        return EvalRowResult(
            id=q_id,
            status="incorrect",
            score=-1,
            gt_type="string",
            pred_type="string",
            error_type="value_mismatch",
            details=f"String mismatch: predicted '{pred_val}', expected '{gt_ans}' (alias={gt_alias})",
            shop=shop,
            intent=intent,
        )

    def _eval_list(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        if not isinstance(pred_val, list):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="list",
                pred_type="list",
                error_type="format_error",
                details=f"List answer must be a JSON array: {repr(pred_val)}",
                shop=shop,
                intent=intent,
            )

        if not isinstance(gt_ans, list):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="list",
                pred_type="list",
                error_type="format_error",
                details="Ground truth answer is not a list",
                shop=shop,
                intent=intent,
            )

        # Rulebook §9: "Danh sách: đúng cả tập và thứ tự."
        if len(pred_val) != len(gt_ans):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="list",
                pred_type="list",
                error_type="value_mismatch",
                details=f"List length mismatch: predicted len={len(pred_val)}, expected len={len(gt_ans)}",
                shop=shop,
                intent=intent,
            )

        # Per-element comparison. Format errors and non-numeric mismatches fail
        # immediately; numeric mismatches are collected so the error label is the
        # worst of rounding_error / close_value / value_mismatch across elements.
        numeric_misses: List[Tuple[int, Any, Any, str]] = []
        for i in range(len(gt_ans)):
            gt_item = gt_ans[i]
            pred_item = pred_val[i]
            item_aliases = []
            if isinstance(gt_alias, list) and i < len(gt_alias) and isinstance(gt_alias[i], list):
                item_aliases = gt_alias[i]

            if isinstance(gt_item, (int, float)) and not isinstance(gt_item, bool):
                if not isinstance(pred_item, (int, float)) or isinstance(pred_item, bool):
                    return EvalRowResult(
                        id=q_id,
                        status="incorrect",
                        score=-1,
                        gt_type="list",
                        pred_type="list",
                        error_type="format_error",
                        details=f"List element {i} must be number: {repr(pred_item)}",
                        shop=shop,
                        intent=intent,
                    )
                if not math.isfinite(pred_item):
                    return EvalRowResult(
                        id=q_id,
                        status="incorrect",
                        score=-1,
                        gt_type="list",
                        pred_type="list",
                        error_type="format_error",
                        details=f"List element {i} must be a finite number: {repr(pred_item)}",
                        shop=shop,
                        intent=intent,
                    )
                if not numbers_match(pred_item, gt_item):
                    numeric_misses.append(
                        (i, pred_item, gt_item, numeric_error_label(pred_item, gt_item, _element_kind(gt_item)))
                    )
            elif isinstance(gt_item, str):
                if not isinstance(pred_item, str):
                    return EvalRowResult(
                        id=q_id,
                        status="incorrect",
                        score=-1,
                        gt_type="list",
                        pred_type="list",
                        error_type="format_error",
                        details=f"List element {i} must be string: {repr(pred_item)}",
                        shop=shop,
                        intent=intent,
                    )
                if not match_string(pred_item, gt_item, item_aliases):
                    return EvalRowResult(
                        id=q_id,
                        status="incorrect",
                        score=-1,
                        gt_type="list",
                        pred_type="list",
                        error_type="value_mismatch",
                        details=f"List element {i} mismatch: predicted '{pred_item}', expected '{gt_item}' (alias={item_aliases})",
                        shop=shop,
                        intent=intent,
                    )
            else:
                if pred_item != gt_item:
                    return EvalRowResult(
                        id=q_id,
                        status="incorrect",
                        score=-1,
                        gt_type="list",
                        pred_type="list",
                        error_type="value_mismatch",
                        details=f"List element {i} mismatch: predicted {pred_item}, expected {gt_item}",
                        shop=shop,
                        intent=intent,
                    )

        if numeric_misses:
            i, p_item, g_item, _ = numeric_misses[0]
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="list",
                pred_type="list",
                error_type=_worst_label(m[3] for m in numeric_misses),
                details=(f"List numeric mismatch at {len(numeric_misses)} element(s); first: "
                         f"element {i} predicted {p_item}, expected {g_item}"),
                shop=shop,
                intent=intent,
            )

        return EvalRowResult(
            id=q_id,
            status="correct",
            score=1,
            gt_type="list",
            pred_type="list",
            details="All list elements matched in order",
            shop=shop,
            intent=intent,
        )

    def _eval_table(
        self,
        q_id: str,
        pred_val: Any,
        gt_ans: Any,
        gt_alias: List[Any],
        shop: str,
        intent: str,
    ) -> EvalRowResult:
        # Format checks for table
        if not isinstance(pred_val, dict):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="table",
                pred_type="table",
                error_type="format_error",
                details="Table must be a JSON object with 'columns' and 'rows'",
                shop=shop,
                intent=intent,
            )

        if "columns" not in pred_val or "rows" not in pred_val:
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="table",
                pred_type="table",
                error_type="format_error",
                details="Table missing 'columns' or 'rows' keys",
                shop=shop,
                intent=intent,
            )

        pred_cols = pred_val["columns"]
        pred_rows = pred_val["rows"]

        if not isinstance(pred_cols, list) or not isinstance(pred_rows, list):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="table",
                pred_type="table",
                error_type="format_error",
                details="'columns' and 'rows' must be lists",
                shop=shop,
                intent=intent,
            )

        gt_cols = gt_ans.get("columns", [])
        gt_rows = gt_ans.get("rows", [])

        if len(pred_cols) != len(gt_cols):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="table",
                pred_type="table",
                error_type="value_mismatch",
                details=f"Table column count mismatch: predicted {len(pred_cols)}, expected {len(gt_cols)}",
                shop=shop,
                intent=intent,
            )

        # Check that each row is a list of matching column length
        for r_idx, row in enumerate(pred_rows):
            if not isinstance(row, list) or len(row) != len(pred_cols):
                return EvalRowResult(
                    id=q_id,
                    status="incorrect",
                    score=-1,
                    gt_type="table",
                    pred_type="table",
                    error_type="format_error",
                    details=f"Row {r_idx} must be a list of {len(pred_cols)} items: {repr(row)}",
                    shop=shop,
                    intent=intent,
                )

        # Check row count
        if len(pred_rows) != len(gt_rows):
            return EvalRowResult(
                id=q_id,
                status="incorrect",
                score=-1,
                gt_type="table",
                pred_type="table",
                error_type="value_mismatch",
                details=f"Table row count mismatch: predicted {len(pred_rows)}, expected {len(gt_rows)}",
                shop=shop,
                intent=intent,
            )

        # Map columns by name or position
        col_map = get_column_mapping(gt_cols, pred_cols)

        # Check cell format: numeric columns in GT must have JSON number cells in Pred
        for gt_c_idx in range(len(gt_cols)):
            pred_c_idx = col_map[gt_c_idx]
            # Check if this column is numeric in GT
            if gt_rows and isinstance(gt_rows[0][gt_c_idx], (int, float)) and not isinstance(gt_rows[0][gt_c_idx], bool):
                for r_idx, row in enumerate(pred_rows):
                    cell = row[pred_c_idx]
                    if not isinstance(cell, (int, float)) or isinstance(cell, bool) or math.isnan(cell) or math.isinf(cell):
                        return EvalRowResult(
                            id=q_id,
                            status="incorrect",
                            score=-1,
                            gt_type="table",
                            pred_type="table",
                            error_type="format_error",
                            details=f"Table numeric cell at row {r_idx}, col {pred_c_idx} must be JSON number (got {repr(cell)})",
                            shop=shop,
                            intent=intent,
                        )

        # Row matching (order independent, 1-to-1 matching via bipartite graph)
        # Build graph: gt_row_i -> list of matching pred_row_j
        graph: List[List[int]] = []
        for i, gt_row in enumerate(gt_rows):
            row_alias = gt_alias[i] if (isinstance(gt_alias, list) and i < len(gt_alias) and isinstance(gt_alias[i], list)) else []
            matching_j: List[int] = []

            for j, pred_row in enumerate(pred_rows):
                # Check all columns
                row_match = True
                for gt_c_idx in range(len(gt_cols)):
                    pred_c_idx = col_map[gt_c_idx]
                    gt_val = gt_row[gt_c_idx]
                    pred_cell = pred_row[pred_c_idx]

                    if gt_c_idx == 0:
                        # First column is entity (string)
                        if isinstance(gt_val, str):
                            if not match_string(pred_cell, gt_val, row_alias):
                                row_match = False
                                break
                        else:
                            if pred_cell != gt_val:
                                row_match = False
                                break
                    else:
                        # Other columns (metrics)
                        if isinstance(gt_val, (int, float)) and not isinstance(gt_val, bool):
                            if not isinstance(pred_cell, (int, float)) or isinstance(pred_cell, bool):
                                row_match = False
                                break
                            if not numbers_match(pred_cell, gt_val):
                                row_match = False
                                break
                        elif isinstance(gt_val, str):
                            if not match_string(pred_cell, gt_val, []):
                                row_match = False
                                break
                        else:
                            if pred_cell != gt_val:
                                row_match = False
                                break

                if row_match:
                    matching_j.append(j)

            graph.append(matching_j)

        matching_size = max_bipartite_matching(graph)
        if matching_size == len(gt_rows):
            return EvalRowResult(
                id=q_id,
                status="correct",
                score=1,
                gt_type="table",
                pred_type="table",
                details=f"All {len(gt_rows)} table rows matched bijectively",
                shop=shop,
                intent=intent,
            )

        return EvalRowResult(
            id=q_id,
            status="incorrect",
            score=-1,
            gt_type="table",
            pred_type="table",
            error_type=self._table_error_label(gt_rows, pred_rows, col_map, gt_alias, graph),
            details=f"Table row matching failed: matched {matching_size}/{len(gt_rows)} rows",
            shop=shop,
            intent=intent,
        )

    @staticmethod
    def _table_error_label(
        gt_rows: List[List[Any]],
        pred_rows: List[List[Any]],
        col_map: List[int],
        gt_alias: Any,
        graph: List[List[int]],
    ) -> str:
        """
        Error label for a failed table match. For every GT row without an exact
        match, look for a pred row with the same entity (first column) whose
        non-numeric cells all match; its numeric cell misses give the label
        (closest candidate per row, worst across rows). A GT row with no such
        pred row -> value_mismatch.
        """
        labels: List[str] = []
        for i, gt_row in enumerate(gt_rows):
            if graph[i]:
                continue
            row_alias = gt_alias[i] if (isinstance(gt_alias, list) and i < len(gt_alias) and isinstance(gt_alias[i], list)) else []
            best: Optional[str] = None
            for pred_row in pred_rows:
                ent_gt, ent_pred = gt_row[0], pred_row[col_map[0]]
                same_entity = match_string(ent_pred, ent_gt, row_alias) if isinstance(ent_gt, str) else ent_pred == ent_gt
                if not same_entity:
                    continue
                cell_labels: List[str] = []
                ok = True
                for c in range(1, len(gt_row)):
                    g, p = gt_row[c], pred_row[col_map[c]]
                    if _is_json_number(g):
                        if not _is_json_number(p):
                            ok = False
                            break
                        if not numbers_match(p, g):
                            cell_labels.append(numeric_error_label(p, g, _element_kind(g)))
                    elif isinstance(g, str):
                        if not match_string(p, g, []):
                            ok = False
                            break
                    elif p != g:
                        ok = False
                        break
                if ok and cell_labels:
                    cand = _worst_label(cell_labels)
                    if best is None or _SEVERITY[cand] < _SEVERITY[best]:
                        best = cand
            if best is None:
                return "value_mismatch"
            labels.append(best)
        # Every GT row had an exact candidate (pure matching conflict) -> value_mismatch
        return _worst_label(labels)

    # -------------------------------------------------------------------------
    # Batch Submission Evaluation
    # -------------------------------------------------------------------------

    def load_submission(
        self,
        submission: Union[str, Path, Sequence[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Loads a submission into {id: record} and collects loading problems.

        Duplicate ids: the LAST row for an id wins (matches a JSONL reader that
        overwrites), but every duplicated id is reported. Ids not present in the
        ground truth are ignored and reported. Rows that are not valid JSON, not a
        JSON object, or lack a string "id" are skipped and counted.

        Returns (sub_map, warnings) where warnings only contains non-empty entries
        (see EvalReport.submission_warnings).
        """
        sub_map: Dict[str, Dict[str, Any]] = {}
        duplicates: Set[str] = set()
        invalid_positions: List[int] = []

        def add(record: Any, pos: int) -> None:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                invalid_positions.append(pos)
                return
            rid = record["id"]
            if rid in sub_map:
                duplicates.add(rid)
            sub_map[rid] = record  # last row wins

        if isinstance(submission, (str, Path)):
            sub_path = Path(submission)
            if not sub_path.is_file():
                raise FileNotFoundError(f"Submission file not found: {sub_path}")
            with open(sub_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_positions.append(line_idx)
                        continue
                    add(record, line_idx)
        elif isinstance(submission, dict):
            for pos, (rid, record) in enumerate(submission.items(), start=1):
                if not isinstance(record, dict):
                    invalid_positions.append(pos)
                    continue
                sub_map[rid] = record
        else:
            for pos, record in enumerate(submission, start=1):
                add(record, pos)

        warnings: Dict[str, Any] = {}
        if duplicates:
            warnings["duplicate_ids"] = sorted(duplicates)
        unknown = sorted(rid for rid in sub_map if rid not in self.ground_truth)
        if unknown:
            warnings["unknown_ids"] = unknown
        if invalid_positions:
            warnings["invalid_lines"] = len(invalid_positions)
            warnings["invalid_line_numbers"] = invalid_positions[:20]
        return sub_map, warnings

    def evaluate_submission(
        self,
        submission: Union[str, Path, Sequence[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        ids: Optional[Iterable[str]] = None,
        warn: bool = True,
    ) -> EvalReport:
        """
        Evaluates a submission against the initialized ground truth.
        Supports file path, list of dicts, or dict of {id: pred_dict}.

        ids:  optional subset of ground-truth ids to score (e.g. the questions a
              run actually selected with --limit/--shops). Other ground-truth ids
              are left out of the report entirely (not counted as abstain), and
              submission rows for them are silently ignored. Ids in `ids` that are
              not in the ground truth raise ValueError.
        warn: print submission loading warnings to stderr (they are always
              recorded on report.submission_warnings).
        """
        sub_map, sub_warnings = self.load_submission(submission)
        if warn:
            for w in format_submission_warnings(sub_warnings):
                print(f"Warning: {w}", file=sys.stderr)

        if ids is None:
            scope = self.ground_truth
        else:
            id_list = list(dict.fromkeys(ids))
            missing = [i for i in id_list if i not in self.ground_truth]
            if missing:
                raise ValueError(f"{len(missing)} requested id(s) not in ground truth, e.g. {missing[:3]}")
            scope = {i: self.ground_truth[i] for i in id_list}

        row_results: List[EvalRowResult] = []
        correct = 0
        incorrect = 0
        abstain = 0
        total_score = 0
        error_taxonomy: Counter[str] = Counter()

        by_shop: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "abstain": 0, "score": 0})
        by_type: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "abstain": 0, "score": 0})
        by_intent: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "abstain": 0, "score": 0})
        by_intent_shop_type: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "abstain": 0, "score": 0})

        for q_id, gt_dict in scope.items():
            pred_dict = sub_map.get(q_id)
            row_res = self.evaluate_row(q_id, pred_dict, gt_dict)
            row_results.append(row_res)

            total_score += row_res.score
            if row_res.status == "correct":
                correct += 1
            elif row_res.status == "incorrect":
                incorrect += 1
                if row_res.error_type:
                    error_taxonomy[row_res.error_type] += 1
            else:
                abstain += 1

            # Stats by shop
            s_stat = by_shop[row_res.shop]
            s_stat["total"] += 1
            s_stat[row_res.status] += 1
            s_stat["score"] += row_res.score

            # Stats by type
            t_stat = by_type[row_res.gt_type]
            t_stat["total"] += 1
            t_stat[row_res.status] += 1
            t_stat["score"] += row_res.score

            # Stats by intent
            i_stat = by_intent[row_res.intent]
            i_stat["total"] += 1
            i_stat[row_res.status] += 1
            i_stat["score"] += row_res.score

            # Stats by intent x shop x type
            ist_key = f"{row_res.intent} | {row_res.shop} | {row_res.gt_type}"
            ist_stat = by_intent_shop_type[ist_key]
            ist_stat["total"] += 1
            ist_stat[row_res.status] += 1
            ist_stat["score"] += row_res.score

        # Compute accuracies
        def enrich_stats(stats_dict: Dict[str, Dict[str, Any]]) -> None:
            for k, d in stats_dict.items():
                ans_count = d["correct"] + d["incorrect"]
                d["accuracy"] = (d["correct"] / ans_count) if ans_count > 0 else 0.0

        enrich_stats(by_shop)
        enrich_stats(by_type)
        enrich_stats(by_intent)
        enrich_stats(by_intent_shop_type)

        total_q = len(scope)
        answered = correct + incorrect
        coverage_rate = answered / max(total_q, 1)
        accuracy = (correct / answered) if answered > 0 else 0.0
        expected_score_per_answered = (2 * accuracy - 1) if answered > 0 else 0.0

        return EvalReport(
            total_questions=total_q,
            correct=correct,
            incorrect=incorrect,
            abstain=abstain,
            total_score=total_score,
            max_possible_score=total_q,
            coverage_rate=coverage_rate,
            accuracy_on_answered=accuracy,
            expected_score_per_answered=expected_score_per_answered,
            by_shop=dict(by_shop),
            by_type=dict(by_type),
            by_intent=dict(by_intent),
            by_intent_shop_type=dict(by_intent_shop_type),
            error_taxonomy=dict(error_taxonomy),
            submission_warnings=sub_warnings,
            row_results=row_results,
        )


def evaluate_submission(
    submission: Union[str, Path, Sequence[Dict[str, Any]]],
    answers: Union[str, Path, Sequence[Dict[str, Any]]],
    questions: Optional[Union[str, Path, Sequence[Dict[str, Any]]]] = None,
    ids: Optional[Iterable[str]] = None,
) -> EvalReport:
    """Helper function to run evaluation in a single line."""
    evaluator = Evaluator(answers=answers, questions=questions)
    return evaluator.evaluate_submission(submission, ids=ids)


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DataMate Submission Evaluator (Section 7 & 9)")
    parser.add_argument("--submission", "-s", required=True, help="Path to submission.jsonl")
    parser.add_argument("--answers", "-a", default="train/answers.jsonl", help="Path to ground truth answers.jsonl")
    parser.add_argument("--questions", "-q", default=None, help="Path to questions.jsonl (for shop/intent breakdown)")
    parser.add_argument("--output-json", "-o", default=None, help="Optional path to save full report JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print individual row results")

    args = parser.parse_args()

    evaluator = Evaluator(answers=args.answers, questions=args.questions)
    report = evaluator.evaluate_submission(args.submission)

    print(report.summary_table())

    if args.verbose:
        print("\n[ DETAILED ROW RESULTS (First 50) ]")
        for row in report.row_results[:50]:
            print(f"{row.id:<18} | {row.status:<9} | score={row.score:+2d} | gt={row.gt_type:<10} | pred={str(row.pred_type):<10} | err={str(row.error_type):<30} | {row.details}")

    if args.output_json:
        out_path = save_report_json(report, args.output_json)
        print(f"\nSaved detailed evaluation JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
