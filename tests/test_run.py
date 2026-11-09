"""Tests for the runner (run.py): solver registry, error handling, format gate, CLI options."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402
from run import (  # noqa: E402
    SOLVERS,
    AbstainSolver,
    OracleSolver,
    Solver,
    SolverContext,
    build_solver,
    register_solver,
    select_questions,
    solve_one,
)

QUESTIONS = [
    {"id": "q1", "business_id": "biz_101", "question": "doanh thu hôm nay", "as_of_date": "2026-01-01"},
    {"id": "q2", "business_id": "biz_102", "question": "tồn kho còn bao nhiêu", "as_of_date": "2026-01-01"},
    {"id": "q3", "business_id": "biz_101", "question": "khách nào nợ nhiều nhất", "as_of_date": "2026-01-01"},
    {"id": "q4", "business_id": "biz_101", "question": "nhân viên có tận tâm không", "as_of_date": "2026-01-01"},
]
ANSWERS = [
    {"id": "q1", "type": "number", "answer": 100, "alias": []},
    {"id": "q2", "type": "number", "answer": 7, "alias": []},
    {"id": "q3", "type": "string", "answer": "Chị Lan", "alias": ["KH0001"]},
    {"id": "q4", "type": "unanswerable", "answer": None, "alias": []},
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _Raising(Solver):
    name = "raising"

    def solve(self, question):
        if question["id"] == "q2":
            raise RuntimeError("boom")
        return {"type": "number", "answer": 100}


class _BadFormat(Solver):
    name = "bad"

    def solve(self, question):
        return {"type": "number", "answer": "100"}


class TestRunner(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.q_path = self.dir / "questions.jsonl"
        self.a_path = self.dir / "answers.jsonl"
        write_jsonl(self.q_path, QUESTIONS)
        write_jsonl(self.a_path, ANSWERS)
        self.ctx = SolverContext(split="train", questions_path=self.q_path, answers_path=self.a_path)

    def tearDown(self):
        self.tmp.cleanup()

    def quiet(self):
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return stack

    # --- registry -------------------------------------------------------------
    def test_registry_has_builtin_modes(self):
        self.assertIn("abstain", SOLVERS)
        self.assertIn("oracle_check", SOLVERS)
        self.assertIsInstance(build_solver("abstain", self.ctx), AbstainSolver)
        self.assertIsInstance(build_solver("oracle_check", self.ctx), OracleSolver)
        with self.assertRaises(ValueError):
            build_solver("nope", self.ctx)

    def test_register_custom_solver(self):
        mode = "_test_custom_mode"
        try:
            @register_solver(mode)
            class Custom(Solver):
                name = mode

                def solve(self, question):
                    return {"type": "boolean", "answer": True}

            self.assertIsInstance(build_solver(mode, self.ctx), Custom)
            with self.assertRaises(ValueError):
                register_solver(mode)(Custom)
        finally:
            SOLVERS.pop(mode, None)

    # --- solvers ---------------------------------------------------------------
    def test_oracle_solver_outputs(self):
        oracle = OracleSolver(self.ctx)
        self.assertEqual(oracle.solve(QUESTIONS[0]), {"type": "number", "answer": 100})
        self.assertEqual(oracle.solve(QUESTIONS[3]), {"type": "unanswerable", "reason": "oracle"})
        self.assertNotIn("answer", oracle.solve(QUESTIONS[3]))
        with self.assertRaises(FileNotFoundError):
            OracleSolver(SolverContext(split="x", questions_path=self.q_path, answers_path=None))

    # --- error handling / format gate -----------------------------------------
    def test_solver_exception_becomes_abstain(self):
        with self.quiet():
            res = solve_one(_Raising(self.ctx), QUESTIONS[1])
        self.assertEqual(res["record"], {"id": "q2", "type": "abstain"})
        self.assertIn("RuntimeError: boom", res["error"])
        self.assertIsNone(res["format_error"])

    def test_invalid_format_becomes_abstain(self):
        with self.quiet():
            res = solve_one(_BadFormat(self.ctx), QUESTIONS[0])
        self.assertEqual(res["record"], {"id": "q1", "type": "abstain"})
        self.assertIsNone(res["error"])
        self.assertIn("JSON number", res["format_error"])

    def test_run_writes_output_and_log(self):
        out, log = self.dir / "out.jsonl", self.dir / "log.jsonl"
        with self.quiet():
            stats = run.run(_Raising(self.ctx), QUESTIONS, out, mode="raising", log_path=log)
        self.assertEqual(stats["written"], 4)
        self.assertEqual(stats["errors"], 1)
        rows = read_jsonl(out)
        self.assertEqual([r["id"] for r in rows], ["q1", "q2", "q3", "q4"])
        self.assertEqual(rows[1], {"id": "q2", "type": "abstain"})
        logs = read_jsonl(log)
        self.assertEqual(len(logs), 4)
        self.assertEqual(logs[1]["mode"], "raising")
        self.assertIn("boom", logs[1]["error"])
        self.assertIsNone(logs[0]["error"])
        self.assertIsInstance(logs[0]["elapsed_s"], float)

    # --- selection / resume ------------------------------------------------------
    def test_select_questions(self):
        self.assertEqual([q["id"] for q in select_questions(QUESTIONS, shops={"biz_101"})], ["q1", "q3", "q4"])
        self.assertEqual([q["id"] for q in select_questions(QUESTIONS, shops={"biz_101"}, limit=2)], ["q1", "q3"])
        self.assertEqual([q["id"] for q in select_questions(QUESTIONS, limit=1)], ["q1"])
        self.assertEqual(len(select_questions(QUESTIONS)), 4)

    def test_resume_skips_done_ids_and_appends(self):
        out = self.dir / "out.jsonl"
        write_jsonl(out, [{"id": "q1", "type": "number", "answer": 100}])
        with self.quiet():
            stats = run.run(AbstainSolver(self.ctx), QUESTIONS, out, mode="abstain", resume=True)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["written"], 3)
        rows = read_jsonl(out)
        self.assertEqual([r["id"] for r in rows], ["q1", "q2", "q3", "q4"])
        self.assertEqual(rows[0], {"id": "q1", "type": "number", "answer": 100})

    # --- CLI end to end -----------------------------------------------------------
    def _main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run.main(["--questions", str(self.q_path), "--answers", str(self.a_path), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_oracle_full(self):
        out, report = self.dir / "o.jsonl", self.dir / "r.json"
        code, stdout, stderr = self._main("--mode", "oracle_check", "-o", str(out), "--evaluate",
                                          "--output-report-json", str(report))
        self.assertEqual(code, 0)
        self.assertRegex(stdout, r"Total Score:\s+4 / 4")
        self.assertNotIn("WARNINGS", stdout)
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["total_score"], 4)
        self.assertIn("by_intent_shop_type", data)
        self.assertEqual(data["submission_warnings"], {})

    def test_cli_filtered_evaluation_scores_only_selected(self):
        out = self.dir / "p.jsonl"
        code, stdout, _ = self._main("--mode", "oracle_check", "--shops", "biz_101", "--limit", "2",
                                     "-o", str(out), "--evaluate")
        self.assertEqual(code, 0)
        self.assertEqual([r["id"] for r in read_jsonl(out)], ["q1", "q3"])
        self.assertRegex(stdout, r"Total Questions:\s+2")
        self.assertRegex(stdout, r"Total Score:\s+2 / 2")
        self.assertIn("scoring only the 2 selected", stdout)

    def test_cli_abstain_and_log(self):
        out, log = self.dir / "a.jsonl", self.dir / "a.log.jsonl"
        code, stdout, _ = self._main("--mode", "abstain", "-o", str(out), "--log", str(log), "--evaluate")
        self.assertEqual(code, 0)
        self.assertRegex(stdout, r"Total Score:\s+0 / 4")
        self.assertEqual({r["type"] for r in read_jsonl(out)}, {"abstain"})
        self.assertEqual(len(read_jsonl(log)), 4)


if __name__ == "__main__":
    unittest.main()
