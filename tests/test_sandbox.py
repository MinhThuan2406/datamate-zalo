"""Tests for agent.sandbox against the real worker process (a few seconds in total)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.sandbox import (  # noqa: E402
    OUTPUT_LIMIT,
    PythonSandbox,
    SandboxError,
    minimal_env,
    truncate_output,
)


class SandboxTestCase(unittest.TestCase):
    """One worker per class, to keep the suite fast; workers are independent per question in real runs."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name)
        (cls.data_dir / "sales.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        cls.sb = PythonSandbox(cls.data_dir, as_of_date="2026-06-01", timeout_s=30)

    @classmethod
    def tearDownClass(cls):
        cls.sb.close()
        cls._tmp.cleanup()


class TestSandboxState(SandboxTestCase):
    def test_state_persists_between_execs(self):
        r1 = self.sb.exec("kept_value = 41\nimport numpy as np")
        self.assertIsNone(r1.error)
        r2 = self.sb.exec("print(kept_value + 1)")
        self.assertEqual(r2.stdout, "42\n")

    def test_trailing_expression_is_echoed(self):
        r = self.sb.exec("kept_value * 2")
        self.assertEqual(r.stdout, "82\n")

    def test_none_valued_trailing_expression_is_silent(self):
        r = self.sb.exec("print('x')\nNone")
        self.assertEqual(r.stdout, "x\n")

    def test_preamble_names_are_available(self):
        r = self.sb.exec(
            "print(DATA_DIR == " + repr(str(self.data_dir.resolve())) + ")\n"
            "print(AS_OF_DATE, type(AS_OF).__name__, AS_OF.isoformat())\n"
            "print(pd.__name__, np.__name__, duckdb.__name__, sqlite3.__name__, dt.__name__)\n"
            "print(CALENDAR_PATH, RULEBOOK_PATH)\n"
            "print(pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))['b'].sum())"
        )
        self.assertIsNone(r.error, r.error)
        lines = r.stdout.splitlines()
        self.assertEqual(lines[0], "True")
        self.assertEqual(lines[1], "2026-06-01 date 2026-06-01")
        self.assertEqual(lines[2], "pandas numpy duckdb sqlite3 datetime")
        self.assertEqual(lines[3], "None None")
        self.assertEqual(lines[4], "6")

    def test_traceback_names_the_cell_and_hides_worker_frames(self):
        r = self.sb.exec("def f():\n    return 1 / 0\nf()")
        self.assertIsNotNone(r.error)
        self.assertIn('File "<agent>", line 2', r.error)
        self.assertIn("ZeroDivisionError", r.error)
        self.assertNotIn("sandbox_worker", r.error)

    def test_syntax_error_is_reported_not_raised(self):
        r = self.sb.exec("def broken(:\n    pass")
        self.assertIsNotNone(r.error)
        self.assertIn("SyntaxError", r.error)
        self.assertIsNone(r.final)

    def test_user_code_cannot_corrupt_protocol_via_stdin_or_fd1(self):
        r = self.sb.exec("import sys, os\nprint(repr(sys.stdin.read()))\nos.write(1, b'raw-fd1\\n')\nprint('after')")
        self.assertIsNone(r.error, r.error)
        self.assertIn("''", r.stdout)
        self.assertIn("after", r.stdout)
        self.assertNotIn("raw-fd1", r.stdout)  # fd 1 points at stderr after startup

    def test_stderr_is_captured_separately(self):
        r = self.sb.exec("import sys\nprint('out')\nprint('err', file=sys.stderr)")
        self.assertEqual(r.stdout, "out\n")
        self.assertEqual(r.stderr, "err\n")


class TestFinalAnswer(SandboxTestCase):
    def test_numpy_scalar_becomes_plain_int(self):
        r = self.sb.exec("final_answer('number', np.int64(4820000))")
        self.assertEqual(r.final, {"type": "number", "answer": 4820000})
        self.assertIs(type(r.final["answer"]), int)
        self.assertIsNone(r.error)

    def test_numpy_float_and_list_and_bool(self):
        r = self.sb.exec("final_answer('list', [np.float64(12.5), np.int32(3), np.bool_(True)])")
        self.assertEqual(r.final, {"type": "list", "answer": [12.5, 3, True]})
        self.assertIs(type(r.final["answer"][0]), float)
        self.assertIs(type(r.final["answer"][2]), bool)

    def test_dataframe_becomes_table(self):
        r = self.sb.exec("final_answer('table', pd.DataFrame({'khach': ['Chị Lan'], 'so_tien': [np.int64(3200000)]}))")
        self.assertEqual(r.final, {"type": "table",
                                   "answer": {"columns": ["khach", "so_tien"], "rows": [["Chị Lan", 3200000]]}})

    def test_dates_and_unanswerable_reason(self):
        r = self.sb.exec("final_answer('date', '2026-08-15')")
        self.assertEqual(r.final, {"type": "date", "answer": "2026-08-15"})
        r = self.sb.exec("final_answer('unanswerable', reason='shop không ghi trạng thái huỷ')")
        self.assertEqual(r.final, {"type": "unanswerable", "reason": "shop không ghi trạng thái huỷ"})

    def test_abstain_needs_no_answer(self):
        r = self.sb.exec("final_answer('abstain')")
        self.assertEqual(r.final, {"type": "abstain"})

    def test_final_answer_stops_the_cell(self):
        r = self.sb.exec("final_answer('abstain')\nprint('after the answer')")
        self.assertEqual(r.final, {"type": "abstain"})
        self.assertNotIn("after the answer", r.stdout)

    def test_nan_is_rejected_as_an_error_so_the_model_can_fix_it(self):
        r = self.sb.exec("final_answer('number', float('nan'))")
        self.assertIsNone(r.final)
        self.assertIn("ValueError", r.error)
        self.assertIn("NaN", r.error)

    def test_numpy_nan_inside_table_is_rejected(self):
        r = self.sb.exec("final_answer('table', {'columns': ['a', 'b'], 'rows': [['x', np.float64('nan')]]})")
        self.assertIsNone(r.final)
        self.assertIn("NaN", r.error)

    def test_unknown_type_and_missing_answer_are_rejected(self):
        r = self.sb.exec("final_answer('money', 5)")
        self.assertIsNone(r.final)
        self.assertIn("type must be one of", r.error)
        r = self.sb.exec("final_answer('number')")
        self.assertIsNone(r.final)
        self.assertIn("needs an answer", r.error)

    def test_final_is_reset_by_next_exec(self):
        self.sb.exec("final_answer('number', 1)")
        r = self.sb.exec("print('no answer this time')")
        self.assertIsNone(r.final)


class TestOutput(SandboxTestCase):
    def test_stdout_is_truncated_head_and_tail_kept(self):
        r = self.sb.exec("print('H' * 9000 + 'TAIL-MARK')")
        self.assertLessEqual(len(r.stdout), OUTPUT_LIMIT)
        self.assertTrue(r.stdout.startswith("HHH"))
        self.assertIn("TAIL-MARK", r.stdout)
        self.assertIn("characters omitted", r.stdout)

    def test_truncate_output_unit(self):
        text = "a" * 10 + "b" * 10 + "c" * 10
        self.assertEqual(truncate_output(text, limit=100), text)
        cut = truncate_output("x" * 1000 + "END", limit=200)
        self.assertLessEqual(len(cut), 200)
        self.assertTrue(cut.endswith("END"))
        self.assertIn("omitted", cut)


class TestEnvironment(unittest.TestCase):
    def test_worker_env_has_no_api_key(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-not-real"}):
            with PythonSandbox(Path(tmp), as_of_date="2026-06-01", timeout_s=30) as sb:
                r = sb.exec("import os\nprint('OPENROUTER_API_KEY' in os.environ, "
                            "any('sk-test' in v for v in os.environ.values()))")
        self.assertIsNone(r.error, r.error)
        self.assertEqual(r.stdout.strip(), "False False")

    def test_minimal_env_drops_secrets_and_sets_utf8(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-x", "MY_SECRET": "y"}):
            env = minimal_env()
        self.assertNotIn("OPENROUTER_API_KEY", env)
        self.assertNotIn("MY_SECRET", env)
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_bad_inputs_fail_before_starting_a_worker(self):
        with self.assertRaises(FileNotFoundError):
            PythonSandbox(Path(tempfile.gettempdir()) / "definitely-missing-dir-xyz", as_of_date="2026-06-01")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                PythonSandbox(Path(tmp), as_of_date="2026-13-40")

    def test_unlaunchable_interpreter_is_a_sandbox_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = PythonSandbox(Path(tmp), as_of_date="2026-06-01", python=str(Path(tmp) / "no-such-python.exe"))
            with self.assertRaises(SandboxError):
                sb.exec("print(1)")
            sb.close()


class TestRestarts(unittest.TestCase):
    def test_timeout_and_crash_restart_the_worker_with_fresh_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with PythonSandbox(Path(tmp), as_of_date="2026-06-01", timeout_s=2) as sb:
                self.assertIsNone(sb.exec("kept = 5").error)

                slow = sb.exec("import time\ntime.sleep(60)")
                self.assertTrue(slow.timed_out)
                self.assertIn("TimeoutError", slow.error)
                self.assertIn("reset", slow.error)

                after_timeout = sb.exec("print('kept' in globals())")
                self.assertEqual(after_timeout.stdout.strip(), "False")
                self.assertEqual(sb.restarts, 1)

                crash = sb.exec("import os\nos._exit(3)")
                self.assertTrue(crash.crashed)
                self.assertIn("exit code 3", crash.error)

                alive = sb.exec("print('alive')")
                self.assertEqual(alive.stdout, "alive\n")
                self.assertFalse(alive.timed_out or alive.crashed)
                self.assertEqual(sb.restarts, 2)

    def test_close_is_idempotent_and_blocks_further_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = PythonSandbox(Path(tmp), as_of_date="2026-06-01", timeout_s=30)
            sb.exec("x = 1")
            sb.close()
            sb.close()
            with self.assertRaises(SandboxError):
                sb.exec("print(1)")


if __name__ == "__main__":
    unittest.main()
