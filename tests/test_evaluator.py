"""
Unit Tests for DataMate Evaluator (tests/test_evaluator.py)
Validates all scoring criteria from Section 7 and Section 9 of the Rulebook.
"""

import contextlib
import io
import json
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path

# Project root, independent of the current working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator import (  # noqa: E402
    EvalReport,
    Evaluator,
    classify_intent,
    evaluate_submission,
    match_string,
    normalize_text,
    numeric_error_label,
    save_report_json,
)

TRAIN_ANSWERS = ROOT / "train" / "answers.jsonl"
TRAIN_QUESTIONS = ROOT / "train" / "questions.jsonl"


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        # Base dummy ground truths for testing each type
        self.dummy_gt = [
            {"id": "q_num", "type": "number", "answer": 4820000, "alias": []},
            {"id": "q_pct", "type": "percent", "answer": 12.4, "alias": []},
            {"id": "q_date_ymd", "type": "date", "answer": "2026-08-15", "alias": []},
            {"id": "q_date_ym", "type": "date", "answer": "2025-09", "alias": []},
            {"id": "q_bool_t", "type": "boolean", "answer": True, "alias": []},
            {"id": "q_bool_f", "type": "boolean", "answer": False, "alias": []},
            {
                "id": "q_str",
                "type": "string",
                "answer": "Sting dâu 330ml",
                "alias": ["SP001", "Sting dâu lon 330ml"],
            },
            {
                "id": "q_str_phone",
                "type": "string",
                "answer": "Vương Phương Loan",
                "alias": ["KH0185", "Vương Phương Loan - 0973 189 069"],
            },
            {
                "id": "q_list_str",
                "type": "list",
                "answer": ["Cà phê sữa (M)", "Trà đào cam sả (M)", "Cà phê đen (M)"],
                "alias": [["SP0003"], ["SP0017"], ["SP0001"]],
            },
            {
                "id": "q_list_num",
                "type": "list",
                "answer": [3, 10],
                "alias": [],
            },
            {
                "id": "q_table",
                "type": "table",
                "answer": {
                    "columns": ["khach_hang", "so_tien_no"],
                    "rows": [["Chị Lan", 3200000], ["Anh Tuấn", 1500000]],
                },
                "alias": [["KH0001", "Chi Lan"], ["KH0002"]],
            },
            {"id": "q_unans", "type": "unanswerable", "answer": None, "alias": []},
        ]
        self.evaluator = Evaluator(answers=self.dummy_gt)

    # -------------------------------------------------------------------------
    # 1. Number Evaluation
    # -------------------------------------------------------------------------
    def test_number_evaluation(self):
        # Exact integer match
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": 4820000}, self.dummy_gt[0])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Float representation of exact integer
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": 4820000.0}, self.dummy_gt[0])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Float tolerance (tiny epsilon)
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": 4820000.0000000001}, self.dummy_gt[0])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Format error: String number
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": "4820000"}, self.dummy_gt[0])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Format error: Unit text in number
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": "4,82 triệu"}, self.dummy_gt[0])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Format error: Boolean True (which is subclass of int in Python)
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": True}, self.dummy_gt[0])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Rounding error: off by 1 unit
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": 4820001}, self.dummy_gt[0])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "rounding_error")

        # Close value: off by 0.2% (< 1%) but far more than rounding
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": 4810000}, self.dummy_gt[0])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "close_value")

        # Large mismatch
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "number", "answer": 100000}, self.dummy_gt[0])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

    # -------------------------------------------------------------------------
    # 2. Percent Evaluation
    # -------------------------------------------------------------------------
    def test_percent_evaluation(self):
        # Exact percent
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 12.4}, self.dummy_gt[1])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Float representation matching 2 decimals
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 12.400000000000002}, self.dummy_gt[1])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Format error: string with % sign
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": "12,4%"}, self.dummy_gt[1])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Proportion error (0.124 instead of 12.4) -> value mismatch
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 0.124}, self.dummy_gt[1])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

        # Rounding error (12.41 vs 12.40)
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 12.41}, self.dummy_gt[1])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "rounding_error")

        # Unrounded 3-decimal percent (12.404 vs 12.4) must also be rejected (-1)
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 12.404}, self.dummy_gt[1])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "rounding_error")

        # Close value: 12.45 vs 12.4 (|diff| <= 0.05)
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 12.45}, self.dummy_gt[1])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.error_type, "close_value")

        # Value mismatch: 12.6 vs 12.4
        res = self.evaluator.evaluate_row("q_pct", {"id": "q_pct", "type": "percent", "answer": 12.6}, self.dummy_gt[1])
        self.assertEqual(res.error_type, "value_mismatch")

    # -------------------------------------------------------------------------
    # 3. Date Evaluation
    # -------------------------------------------------------------------------
    def test_date_evaluation(self):
        # YYYY-MM-DD
        res = self.evaluator.evaluate_row("q_date_ymd", {"id": "q_date_ymd", "type": "date", "answer": "2026-08-15"}, self.dummy_gt[2])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # YYYY-MM
        res = self.evaluator.evaluate_row("q_date_ym", {"id": "q_date_ym", "type": "date", "answer": "2025-09"}, self.dummy_gt[3])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Format error: DD/MM/YYYY
        res = self.evaluator.evaluate_row("q_date_ymd", {"id": "q_date_ymd", "type": "date", "answer": "15/08/2026"}, self.dummy_gt[2])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Format error: "tháng 9/2025"
        res = self.evaluator.evaluate_row("q_date_ym", {"id": "q_date_ym", "type": "date", "answer": "tháng 9/2025"}, self.dummy_gt[3])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Date value mismatch
        res = self.evaluator.evaluate_row("q_date_ymd", {"id": "q_date_ymd", "type": "date", "answer": "2026-08-16"}, self.dummy_gt[2])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

    # -------------------------------------------------------------------------
    # 4. Boolean Evaluation
    # -------------------------------------------------------------------------
    def test_boolean_evaluation(self):
        # Exact True / False
        res = self.evaluator.evaluate_row("q_bool_t", {"id": "q_bool_t", "type": "boolean", "answer": True}, self.dummy_gt[4])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        res = self.evaluator.evaluate_row("q_bool_f", {"id": "q_bool_f", "type": "boolean", "answer": False}, self.dummy_gt[5])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Format errors: "có", 1, "true"
        for bad_bool in ["có", 1, "true", "True"]:
            res = self.evaluator.evaluate_row("q_bool_t", {"id": "q_bool_t", "type": "boolean", "answer": bad_bool}, self.dummy_gt[4])
            self.assertEqual(res.status, "incorrect", f"Failed on {bad_bool}")
            self.assertEqual(res.error_type, "format_error")

        # Value mismatch
        res = self.evaluator.evaluate_row("q_bool_t", {"id": "q_bool_t", "type": "boolean", "answer": False}, self.dummy_gt[4])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.error_type, "value_mismatch")

    # -------------------------------------------------------------------------
    # 5. String & Alias Evaluation
    # -------------------------------------------------------------------------
    def test_string_and_alias_evaluation(self):
        # Exact string
        res = self.evaluator.evaluate_row("q_str", {"id": "q_str", "type": "string", "answer": "Sting dâu 330ml"}, self.dummy_gt[6])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Case-insensitive
        res = self.evaluator.evaluate_row("q_str", {"id": "q_str", "type": "string", "answer": "sting dâu 330ml"}, self.dummy_gt[6])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Punctuation & extra space insensitive
        res = self.evaluator.evaluate_row("q_str", {"id": "q_str", "type": "string", "answer": "  Sting  dâu   330ml.  "}, self.dummy_gt[6])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Diacritics check: "Sting dầu 330ml" (wrong accent) must FAIL
        res = self.evaluator.evaluate_row("q_str", {"id": "q_str", "type": "string", "answer": "Sting dầu 330ml"}, self.dummy_gt[6])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

        # Alias matching: Product code "SP001"
        res = self.evaluator.evaluate_row("q_str", {"id": "q_str", "type": "string", "answer": "SP001"}, self.dummy_gt[6])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Alias matching: Name with phone number in alias
        res = self.evaluator.evaluate_row("q_str_phone", {"id": "q_str_phone", "type": "string", "answer": "KH0185"}, self.dummy_gt[7])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        res = self.evaluator.evaluate_row("q_str_phone", {"id": "q_str_phone", "type": "string", "answer": "Vương Phương Loan - 0973 189 069"}, self.dummy_gt[7])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Format error: String inside list
        res = self.evaluator.evaluate_row("q_str", {"id": "q_str", "type": "string", "answer": ["Sting dâu 330ml"]}, self.dummy_gt[6])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

    # -------------------------------------------------------------------------
    # 6. List Evaluation
    # -------------------------------------------------------------------------
    def test_list_evaluation(self):
        # Exact list match
        pred_list = ["Cà phê sữa (M)", "Trà đào cam sả (M)", "Cà phê đen (M)"]
        res = self.evaluator.evaluate_row("q_list_str", {"id": "q_list_str", "type": "list", "answer": pred_list}, self.dummy_gt[8])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # List elements matching per-element aliases
        pred_alias_list = ["SP0003", "Trà đào cam sả (m)", "SP0001"]
        res = self.evaluator.evaluate_row("q_list_str", {"id": "q_list_str", "type": "list", "answer": pred_alias_list}, self.dummy_gt[8])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Wrong order in list must FAIL (Section 9: đúng cả tập và thứ tự)
        pred_wrong_order = ["Trà đào cam sả (M)", "Cà phê sữa (M)", "Cà phê đen (M)"]
        res = self.evaluator.evaluate_row("q_list_str", {"id": "q_list_str", "type": "list", "answer": pred_wrong_order}, self.dummy_gt[8])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

        # Length mismatch
        pred_short = ["Cà phê sữa (M)", "Trà đào cam sả (M)"]
        res = self.evaluator.evaluate_row("q_list_str", {"id": "q_list_str", "type": "list", "answer": pred_short}, self.dummy_gt[8])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

        # Format error: String instead of JSON array
        res = self.evaluator.evaluate_row("q_list_str", {"id": "q_list_str", "type": "list", "answer": "Cà phê sữa (M), Trà đào cam sả (M)"}, self.dummy_gt[8])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # List of numbers
        res = self.evaluator.evaluate_row("q_list_num", {"id": "q_list_num", "type": "list", "answer": [3, 10]}, self.dummy_gt[9])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

    # -------------------------------------------------------------------------
    # 7. Table Evaluation
    # -------------------------------------------------------------------------
    def test_table_evaluation(self):
        # Exact table
        pred_table = {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Chị Lan", 3200000], ["Anh Tuấn", 1500000]],
        }
        res = self.evaluator.evaluate_row("q_table", {"id": "q_table", "type": "table", "answer": pred_table}, self.dummy_gt[10])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Different row order (Section 9: Thứ tự các dòng không ảnh hưởng kết quả chấm)
        pred_table_rev = {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Anh Tuấn", 1500000], ["Chị Lan", 3200000]],
        }
        res = self.evaluator.evaluate_row("q_table", {"id": "q_table", "type": "table", "answer": pred_table_rev}, self.dummy_gt[10])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Different column names (fallback to position)
        pred_table_diff_cols = {
            "columns": ["customer", "debt"],
            "rows": [["Chị Lan", 3200000], ["Anh Tuấn", 1500000]],
        }
        res = self.evaluator.evaluate_row("q_table", {"id": "q_table", "type": "table", "answer": pred_table_diff_cols}, self.dummy_gt[10])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Entity alias in first column (e.g. "KH0001" and "KH0002")
        pred_table_alias = {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["KH0002", 1500000], ["KH0001", 3200000]],
        }
        res = self.evaluator.evaluate_row("q_table", {"id": "q_table", "type": "table", "answer": pred_table_alias}, self.dummy_gt[10])
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Format error: Numeric cell as string (Section 9: Sai: ô "3.200.000đ")
        pred_table_bad_cell = {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Chị Lan", "3.200.000đ"], ["Anh Tuấn", 1500000]],
        }
        res = self.evaluator.evaluate_row("q_table", {"id": "q_table", "type": "table", "answer": pred_table_bad_cell}, self.dummy_gt[10])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "format_error")

        # Value mismatch in numeric cell
        pred_table_wrong_num = {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Chị Lan", 9999999], ["Anh Tuấn", 1500000]],
        }
        res = self.evaluator.evaluate_row("q_table", {"id": "q_table", "type": "table", "answer": pred_table_wrong_num}, self.dummy_gt[10])
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "value_mismatch")

    # -------------------------------------------------------------------------
    # 8. Unanswerable Evaluation & Section 7 Rules
    # -------------------------------------------------------------------------
    def test_unanswerable_evaluation(self):
        # GT is unanswerable, pred is unanswerable with string reason -> +1
        res = self.evaluator.evaluate_row(
            "q_unans",
            {"id": "q_unans", "type": "unanswerable", "reason": "Dữ liệu không có thông tin địa chỉ"},
            self.dummy_gt[11],
        )
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # GT is unanswerable, pred is unanswerable without reason -> +1
        res = self.evaluator.evaluate_row(
            "q_unans",
            {"id": "q_unans", "type": "unanswerable"},
            self.dummy_gt[11],
        )
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.score, 1)

        # Section 7 Rule:
        # "dòng có reason không phải chuỗi không đọc được và tính như bỏ trống" (0 points)
        res = self.evaluator.evaluate_row(
            "q_unans",
            {"id": "q_unans", "type": "unanswerable", "reason": 12345},
            self.dummy_gt[11],
        )
        self.assertEqual(res.status, "abstain")
        self.assertEqual(res.score, 0)

        res = self.evaluator.evaluate_row(
            "q_unans",
            {"id": "q_unans", "type": "unanswerable", "reason": ["not_a_string"]},
            self.dummy_gt[11],
        )
        self.assertEqual(res.status, "abstain")
        self.assertEqual(res.score, 0)

        res = self.evaluator.evaluate_row(
            "q_unans",
            {"id": "q_unans", "type": "unanswerable", "reason": None},
            self.dummy_gt[11],
        )
        self.assertEqual(res.status, "abstain")
        self.assertEqual(res.score, 0)

        # GT is unanswerable, but pred answered a number -> -1 (should_be_unanswerable)
        res = self.evaluator.evaluate_row(
            "q_unans",
            {"id": "q_unans", "type": "number", "answer": 0},
            self.dummy_gt[11],
        )
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "should_be_unanswerable")

        # GT is answerable, but pred claimed unanswerable with string reason -> -1 (false_unanswerable)
        res = self.evaluator.evaluate_row(
            "q_num",
            {"id": "q_num", "type": "unanswerable", "reason": "Không tính được"},
            self.dummy_gt[0],
        )
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.score, -1)
        self.assertEqual(res.error_type, "false_unanswerable")

        # GT is answerable, but pred claimed unanswerable with non-string reason -> 0 (treated as abstain)
        res = self.evaluator.evaluate_row(
            "q_num",
            {"id": "q_num", "type": "unanswerable", "reason": 999},
            self.dummy_gt[0],
        )
        self.assertEqual(res.status, "abstain")
        self.assertEqual(res.score, 0)

    # -------------------------------------------------------------------------
    # 9. Abstain Evaluation
    # -------------------------------------------------------------------------
    def test_abstain_evaluation(self):
        # Explicit abstain
        res = self.evaluator.evaluate_row("q_num", {"id": "q_num", "type": "abstain"}, self.dummy_gt[0])
        self.assertEqual(res.status, "abstain")
        self.assertEqual(res.score, 0)

        # Missing prediction row (pred is None)
        res = self.evaluator.evaluate_row("q_num", None, self.dummy_gt[0])
        self.assertEqual(res.status, "abstain")
        self.assertEqual(res.score, 0)

    # -------------------------------------------------------------------------
    # 10. Self-Consistency on Full Train (3000 Questions)
    # -------------------------------------------------------------------------
    def test_oracle_consistency_full_train(self):
        train_ans_path = TRAIN_ANSWERS
        train_q_path = TRAIN_QUESTIONS

        self.assertTrue(train_ans_path.is_file(), "train/answers.jsonl must exist")
        evaluator = Evaluator(answers=train_ans_path, questions=train_q_path)

        # Evaluate train answers against themselves
        report = evaluator.evaluate_submission(train_ans_path)
        self.assertEqual(report.submission_warnings, {})

        self.assertEqual(report.total_questions, 3000)
        self.assertEqual(report.correct, 3000)
        self.assertEqual(report.incorrect, 0)
        self.assertEqual(report.abstain, 0)
        self.assertEqual(report.total_score, 3000)
        self.assertEqual(report.accuracy_on_answered, 1.0)
        self.assertEqual(len(report.error_taxonomy), 0)

    # -------------------------------------------------------------------------
    # 11. Baseline Abstain on Full Train (Score = 0)
    # -------------------------------------------------------------------------
    def test_baseline_abstain_full_train(self):
        train_ans_path = TRAIN_ANSWERS
        train_q_path = TRAIN_QUESTIONS

        evaluator = Evaluator(answers=train_ans_path, questions=train_q_path)

        # Generate abstain submission in memory
        abstain_sub = [{"id": q_id, "type": "abstain"} for q_id in evaluator.ground_truth.keys()]
        report = evaluator.evaluate_submission(abstain_sub)

        self.assertEqual(report.total_questions, 3000)
        self.assertEqual(report.correct, 0)
        self.assertEqual(report.incorrect, 0)
        self.assertEqual(report.abstain, 3000)
        self.assertEqual(report.total_score, 0)
        self.assertEqual(report.coverage_rate, 0.0)

    # -------------------------------------------------------------------------
    # 12. Evaluating a subset of ids
    # -------------------------------------------------------------------------
    def test_evaluate_subset_ids(self):
        sub = [{"id": "q_num", "type": "number", "answer": 4820000},
               {"id": "q_pct", "type": "percent", "answer": 1.0}]
        report = self.evaluator.evaluate_submission(sub, ids=["q_num", "q_bool_t"], warn=False)
        self.assertEqual(report.total_questions, 2)
        self.assertEqual(report.correct, 1)
        self.assertEqual(report.abstain, 1)    # q_bool_t missing
        self.assertEqual(report.incorrect, 0)  # q_pct is out of scope, ignored
        self.assertEqual(report.submission_warnings, {})
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_submission(sub, ids=["not_in_gt"], warn=False)


class TestStringNormalization(unittest.TestCase):

    def test_nfd_input_matches_nfc_ground_truth(self):
        gt = unicodedata.normalize("NFC", "Cà phê sữa")
        pred = unicodedata.normalize("NFD", "Cà phê sữa")
        self.assertNotEqual(pred, gt)  # really different code points
        self.assertTrue(match_string(pred, gt))
        self.assertEqual(normalize_text(pred), normalize_text(gt))
        # Through the evaluator too
        ev = Evaluator(answers=[{"id": "q", "type": "string", "answer": gt, "alias": []}])
        res = ev.evaluate_row("q", {"type": "string", "answer": pred}, ev.ground_truth["q"])
        self.assertEqual(res.status, "correct")
        # And inside lists
        ev = Evaluator(answers=[{"id": "l", "type": "list", "answer": [gt, "Bạc xỉu"], "alias": []}])
        res = ev.evaluate_row("l", {"type": "list", "answer": [pred, unicodedata.normalize("NFD", "Bạc xỉu")]},
                              ev.ground_truth["l"])
        self.assertEqual(res.status, "correct")

    def test_unicode_punctuation_is_stripped(self):
        self.assertEqual(normalize_text("Vương Phương Loan – 0973 189 069"),
                         normalize_text("Vương Phương Loan - 0973 189 069"))
        self.assertEqual(normalize_text("“Cà phê sữa”…"), "cà phê sữa")
        self.assertEqual(normalize_text("Trà đào — cam sả"), "trà đào cam sả")
        self.assertEqual(normalize_text("«Bánh mì»·(M)"), "bánh mì m")
        self.assertTrue(match_string("Vương Phương Loan – 0973 189 069", "Vương Phương Loan",
                                     ["Vương Phương Loan - 0973 189 069"]))
        # ASCII punctuation and symbols are still stripped
        self.assertEqual(normalize_text("a+b*c~d^e=f|g<h>i#j@k&l/m\\n`o"), "a b c d e f g h i j k l m n o")
        self.assertEqual(normalize_text("  Sting  dâu   330ml.  "), "sting dâu 330ml")

    def test_wrong_diacritics_still_fail(self):
        self.assertFalse(match_string("Sting dầu 330ml", "Sting dâu 330ml"))
        self.assertFalse(match_string(unicodedata.normalize("NFD", "Sting dầu 330ml"), "Sting dâu 330ml"))
        self.assertFalse(match_string("Sting dau 330ml", "Sting dâu 330ml"))
        self.assertFalse(match_string("Ca phe sua", "Cà phê sữa"))
        self.assertTrue(match_string("CÀ PHÊ SỮA", "Cà phê sữa"))  # case-insensitive keeps marks


class TestSubmissionWarnings(unittest.TestCase):

    def setUp(self):
        self.gt = [
            {"id": "a", "type": "number", "answer": 1, "alias": []},
            {"id": "b", "type": "number", "answer": 2, "alias": []},
        ]
        self.evaluator = Evaluator(answers=self.gt)

    def test_duplicate_and_unknown_ids_from_list(self):
        sub = [
            {"id": "a", "type": "number", "answer": 99},
            {"id": "a", "type": "number", "answer": 1},   # last wins -> correct
            {"id": "zzz", "type": "number", "answer": 1},
            {"type": "number", "answer": 1},              # no id -> invalid
            "not a dict",                                  # invalid
        ]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            report = self.evaluator.evaluate_submission(sub)
        w = report.submission_warnings
        self.assertEqual(w["duplicate_ids"], ["a"])
        self.assertEqual(w["unknown_ids"], ["zzz"])
        self.assertEqual(w["invalid_lines"], 2)
        self.assertEqual(w["invalid_line_numbers"], [4, 5])
        self.assertEqual(report.correct, 1)   # last row for "a" wins
        self.assertEqual(report.abstain, 1)   # "b" missing
        self.assertIn("duplicate", err.getvalue())
        self.assertIn("not in ground truth", err.getvalue())
        table = report.summary_table()
        self.assertIn("[ WARNINGS ]", table)
        self.assertIn("zzz", table)

    def test_warnings_from_file_with_invalid_json(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sub.jsonl"
            p.write_text(
                '{"id": "a", "type": "number", "answer": 1}\n'
                '{oops not json\n'
                '\n'
                '{"id": "b", "type": "number", "answer": 5}\n'
                '{"id": "b", "type": "number", "answer": 2}\n'
                '[1, 2]\n',
                encoding="utf-8",
            )
            with contextlib.redirect_stderr(io.StringIO()):
                report = self.evaluator.evaluate_submission(p)
        w = report.submission_warnings
        self.assertEqual(w["duplicate_ids"], ["b"])
        self.assertNotIn("unknown_ids", w)
        self.assertEqual(w["invalid_lines"], 2)
        self.assertEqual(w["invalid_line_numbers"], [2, 6])
        self.assertEqual(report.correct, 2)

    def test_clean_submission_has_no_warnings_section(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            report = self.evaluator.evaluate_submission(self.gt)
        self.assertEqual(report.submission_warnings, {})
        self.assertEqual(err.getvalue(), "")
        self.assertNotIn("WARNINGS", report.summary_table())


class TestErrorTaxonomy(unittest.TestCase):

    def test_numeric_error_label(self):
        self.assertEqual(numeric_error_label(4820001, 4820000, "number"), "rounding_error")
        self.assertEqual(numeric_error_label(4810000, 4820000, "number"), "close_value")
        self.assertEqual(numeric_error_label(4000000, 4820000, "number"), "value_mismatch")
        self.assertEqual(numeric_error_label(3, 0, "number"), "value_mismatch")
        self.assertEqual(numeric_error_label(12.41, 12.4, "percent"), "rounding_error")
        self.assertEqual(numeric_error_label(12.45, 12.4, "percent"), "close_value")
        self.assertEqual(numeric_error_label(-12.4, 12.4, "percent"), "value_mismatch")
        self.assertEqual(numeric_error_label(0.124, 12.4, "percent"), "value_mismatch")

    def test_list_numeric_labels(self):
        ev = Evaluator(answers=[
            {"id": "l", "type": "list", "answer": [4820000, 1500000], "alias": []},
            {"id": "p", "type": "list", "answer": [12.4, 3.5], "alias": []},
        ])
        gt_l, gt_p = ev.ground_truth["l"], ev.ground_truth["p"]
        self.assertEqual(ev.evaluate_row("l", {"type": "list", "answer": [4820001, 1500000]}, gt_l).error_type,
                         "rounding_error")
        self.assertEqual(ev.evaluate_row("l", {"type": "list", "answer": [4810000, 1500001]}, gt_l).error_type,
                         "close_value")
        # worst element decides
        self.assertEqual(ev.evaluate_row("l", {"type": "list", "answer": [4820001, 9]}, gt_l).error_type,
                         "value_mismatch")
        # non-integer GT elements use percent thresholds
        self.assertEqual(ev.evaluate_row("p", {"type": "list", "answer": [12.41, 3.5]}, gt_p).error_type,
                         "rounding_error")
        self.assertEqual(ev.evaluate_row("p", {"type": "list", "answer": [12.4, 3.9]}, gt_p).error_type,
                         "value_mismatch")

    def test_table_numeric_labels(self):
        ev = Evaluator(answers=[{
            "id": "t", "type": "table",
            "answer": {"columns": ["khach_hang", "so_tien_no"],
                       "rows": [["Chị Lan", 3200000], ["Anh Tuấn", 1500000]]},
            "alias": [["KH0001"], ["KH0002"]],
        }])
        gt = ev.ground_truth["t"]

        def label(rows):
            return ev.evaluate_row("t", {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                                                      "rows": rows}}, gt).error_type

        self.assertEqual(label([["Chị Lan", 3200001], ["Anh Tuấn", 1500000]]), "rounding_error")
        self.assertEqual(label([["KH0001", 3190000], ["Anh Tuấn", 1500000]]), "close_value")
        self.assertEqual(label([["Chị Lan", 9999999], ["Anh Tuấn", 1500000]]), "value_mismatch")
        self.assertEqual(label([["Chị Hoa", 3200000], ["Anh Tuấn", 1500000]]), "value_mismatch")


class TestReportJson(unittest.TestCase):

    def test_to_dict_and_save_report_json(self):
        ev = Evaluator(answers=[{"id": "a", "type": "number", "answer": 1, "alias": []}],
                       questions=[{"id": "a", "business_id": "biz_1", "question": "tổng dthu hôm nay"}])
        with contextlib.redirect_stderr(io.StringIO()):
            report = ev.evaluate_submission([{"id": "a", "type": "number", "answer": 1},
                                             {"id": "a", "type": "number", "answer": 1}])
        d = report.to_dict()
        for key in ("by_intent_shop_type", "submission_warnings", "row_results", "error_taxonomy", "by_intent"):
            self.assertIn(key, d)
        self.assertEqual(d["submission_warnings"]["duplicate_ids"], ["a"])
        self.assertNotIn("row_results", report.to_dict(include_rows=False))
        with tempfile.TemporaryDirectory() as td:
            out = save_report_json(report, Path(td) / "sub" / "r.json")
            loaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(loaded["correct"], 1)
        self.assertIn("doanh_thu | biz_1 | number", loaded["by_intent_shop_type"])
        self.assertIsInstance(report, EvalReport)

    def test_classify_intent_reexported(self):
        import intent
        self.assertIs(classify_intent, intent.classify_intent)


if __name__ == "__main__":
    unittest.main()
