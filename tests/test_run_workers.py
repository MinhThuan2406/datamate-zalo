"""Tests for run.py's thread pool (--workers), --resume with workers, and the new CLI validation."""

import contextlib
import io
import json
import random
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402
from run import SOLVERS, Solver, SolverContext, build_solver, run as run_all  # noqa: E402

QUESTIONS = [
    {"id": f"q{i:02d}", "business_id": "biz_x", "question": f"câu {i}", "as_of_date": "2026-01-01"}
    for i in range(12)
]


class _Slow(Solver):
    """Thread-safe, answers out of order, and records which threads ran it."""

    name = "slow_test"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.threads = set()
        self._lock = threading.Lock()

    def solve(self, question):
        with self._lock:
            self.threads.add(threading.get_ident())
        time.sleep(random.Random(question["id"]).random() * 0.02)
        if question["id"] == "q05":
            raise RuntimeError("boom in a worker thread")
        return {"type": "number", "answer": int(question["id"][1:])}


class TestThreadedRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.ctx = SolverContext(split="train", questions_path=self.dir / "q.jsonl", answers_path=None,
                                 base_dir=self.dir)
        self.solver = _Slow(self.ctx)

    def tearDown(self):
        self._tmp.cleanup()

    def _rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_threaded_run_writes_every_question_once(self):
        out = self.dir / "sub.jsonl"
        log = self.dir / "log.jsonl"
        stats = run_all(self.solver, QUESTIONS, out, mode="slow_test", log_path=log, workers=4)
        rows = self._rows(out)
        self.assertEqual(sorted(r["id"] for r in rows), sorted(q["id"] for q in QUESTIONS))
        self.assertEqual(stats["written"], 12)
        self.assertEqual(stats["errors"], 1)                      # q05 raised -> abstain row
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["q05"]["type"], "abstain")
        self.assertEqual(by_id["q07"], {"id": "q07", "type": "number", "answer": 7})
        self.assertEqual(len(self._rows(log)), 12)
        self.assertGreaterEqual(len(self.solver.threads), 2)     # really ran on several threads

    def test_resume_after_threaded_run_skips_done_ids(self):
        out = self.dir / "sub.jsonl"
        run_all(self.solver, QUESTIONS[:5], out, mode="slow_test", workers=3)
        stats = run_all(self.solver, QUESTIONS, out, mode="slow_test", resume=True, workers=3)
        self.assertEqual(stats["skipped"], 5)
        self.assertEqual(stats["written"], 7)
        rows = self._rows(out)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["id"] for r in rows}), 12)

    def test_sequential_path_is_unchanged(self):
        out = self.dir / "seq.jsonl"
        stats = run_all(self.solver, QUESTIONS, out, mode="slow_test", workers=1)
        self.assertEqual([r["id"] for r in self._rows(out)], [q["id"] for q in QUESTIONS])  # file order
        self.assertEqual(stats["written"], 12)

    def test_workers_flag_validation(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = run.main(["--mode", "abstain", "--workers", "0", "-o", str(self.dir / "x.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("--workers must be >= 1", err.getvalue())

    def test_max_turns_flag_validation(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = run.main(["--mode", "abstain", "--max-turns", "0", "-o", str(self.dir / "x.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("--max-turns must be >= 1", err.getvalue())

    def test_solver_stats_report_is_optional(self):
        self.assertIsNone(build_solver("abstain", self.ctx).stats_report())
        self.assertIsNone(Solver(self.ctx).stats_report())

    def test_context_defaults_are_backward_compatible(self):
        ctx = SolverContext(split="train", questions_path=self.dir / "q.jsonl", answers_path=None)
        self.assertIsNone(ctx.max_turns)
        self.assertEqual(ctx.run_name, "default")
        self.assertIn("react", SOLVERS)
        self.assertTrue(callable(SOLVERS["react"]))


if __name__ == "__main__":
    unittest.main()
