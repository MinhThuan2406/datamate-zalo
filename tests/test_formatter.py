"""Tests for formatter.validate_format / check_format (§9 record formats)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formatter import check_format, validate_format  # noqa: E402

ABSTAIN = {"type": "abstain"}


class TestValidateFormat(unittest.TestCase):

    def assertValid(self, record):
        out, reason = validate_format(record)
        self.assertIsNone(reason, record)
        return out

    def assertDowngraded(self, record):
        out, reason = validate_format(record)
        self.assertEqual(out, ABSTAIN, record)
        self.assertIsInstance(reason, str)
        return reason

    def test_valid_records(self):
        for rec in [
            {"type": "number", "answer": 4820000},
            {"type": "number", "answer": 18.6},
            {"type": "number", "answer": -5},
            {"type": "percent", "answer": 12.4},
            {"type": "date", "answer": "2026-08-15"},
            {"type": "date", "answer": "2025-09"},
            {"type": "boolean", "answer": True},
            {"type": "boolean", "answer": False},
            {"type": "string", "answer": "Sting dâu 330ml"},
            {"type": "list", "answer": ["Cà phê sữa"]},
            {"type": "list", "answer": [3, 10]},
            {"type": "list", "answer": []},
            {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000]]}},
            {"type": "table", "answer": {"columns": ["a"], "rows": []}},
        ]:
            out = self.assertValid(rec)
            self.assertEqual(out, rec)

    def test_unanswerable_and_abstain(self):
        self.assertEqual(validate_format({"type": "unanswerable"}), ({"type": "unanswerable"}, None))
        self.assertEqual(validate_format({"type": "unanswerable", "reason": "không có dữ liệu"}),
                         ({"type": "unanswerable", "reason": "không có dữ liệu"}, None))
        self.assertEqual(validate_format({"type": "abstain"}), (ABSTAIN, None))
        self.assertEqual(validate_format({"type": "abstain", "answer": 1}), (ABSTAIN, None))
        self.assertDowngraded({"type": "unanswerable", "reason": 123})
        self.assertDowngraded({"type": "unanswerable", "reason": None})

    def test_strips_id_and_extra_keys(self):
        out, reason = validate_format({"id": "q1", "type": "number", "answer": 5, "debug": "x"})
        self.assertIsNone(reason)
        self.assertEqual(out, {"type": "number", "answer": 5})

    def test_number_and_percent_format_errors(self):
        for bad in ["4820000", "4,82 triệu", True, False, None, float("nan"), float("inf"), [5]]:
            self.assertDowngraded({"type": "number", "answer": bad})
            self.assertDowngraded({"type": "percent", "answer": bad})
        self.assertDowngraded({"type": "number"})  # missing answer

    def test_date_format_errors(self):
        for bad in ["15/08/2026", "tháng 9/2025", "2026-13-01", "2026-02-30", "2026-8-15", 20260815, "2026"]:
            self.assertDowngraded({"type": "date", "answer": bad})

    def test_boolean_string_list_errors(self):
        for bad in ["có", 1, 0, "true", None]:
            self.assertDowngraded({"type": "boolean", "answer": bad})
        self.assertDowngraded({"type": "string", "answer": ["Sting dâu 330ml"]})
        self.assertDowngraded({"type": "string", "answer": 5})
        self.assertDowngraded({"type": "list", "answer": "Cà phê sữa, Bạc xỉu"})
        self.assertDowngraded({"type": "list", "answer": [[1, 2]]})
        self.assertDowngraded({"type": "list", "answer": [float("nan")]})

    def test_table_errors(self):
        self.assertDowngraded({"type": "table", "answer": [["a", 1]]})
        self.assertDowngraded({"type": "table", "answer": {"columns": ["a", "b"]}})
        self.assertDowngraded({"type": "table", "answer": {"columns": "a,b", "rows": []}})
        self.assertDowngraded({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x"]]}})
        self.assertDowngraded({"type": "table", "answer": {"columns": ["a", "b"], "rows": ["x", 1]}})
        self.assertDowngraded({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x", float("inf")]]}})
        self.assertDowngraded({"type": "table", "answer": {"columns": [], "rows": []}})

    def test_bad_records(self):
        for bad in [None, "abstain", 5, [], {}, {"type": "money", "answer": 5}, {"answer": 5}]:
            self.assertDowngraded(bad)
        self.assertIsNone(check_format({"type": "number", "answer": 1}))
        self.assertIn("date", check_format({"type": "date", "answer": "x"}))


if __name__ == "__main__":
    unittest.main()
