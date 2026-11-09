"""Tests for agent.react: code extraction and the ReAct loop with a scripted LLM (real sandbox, no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import LLMError, LLMResult, LLMStats, Usage  # noqa: E402
from agent.prompts import MULTI_BLOCK_NOTE, NO_CODE_MESSAGE, format_observation  # noqa: E402
from agent.react import ReactSolver, extract_code_blocks  # noqa: E402
from agent.sandbox import ExecResult  # noqa: E402
from run import SOLVERS, SolverContext  # noqa: E402

CSV = "a,b\n1,10\n2,20\n3,30\n"
RULEBOOK = "# Rulebook test\n## 1. Doanh thu\nDoanh thu = tổng thành tiền.\n"
PY = "```python\n{}\n```"


class ScriptedLLM:
    """chat() returns the next scripted reply (text or exception). Records every messages list it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.stats = LLMStats()

    def chat(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        item = self.replies.pop(0) if self.replies else PY.format("print('filler')")
        if isinstance(item, BaseException):
            raise item
        return LLMResult(text=item, usage=Usage(prompt_tokens=10, completion_tokens=5),
                         provider="PrimeIntellect", model="fake/model", finish_reason="stop")

    def close(self):
        pass


class TestExtractCodeBlocks(unittest.TestCase):
    def test_single_block_variants(self):
        for fence in ("```python", "```py", "```", "```Python", "```python3"):
            text = f"Let me check.\n{fence}\nprint(1)\n```\nDone."
            self.assertEqual(extract_code_blocks(text), ["print(1)"], fence)

    def test_multiple_blocks_are_all_returned_in_order(self):
        text = PY.format("x = 1") + "\nand then\n" + PY.format("print(x)")
        self.assertEqual(extract_code_blocks(text), ["x = 1", "print(x)"])

    def test_no_block(self):
        self.assertEqual(extract_code_blocks("I think the answer is 3."), [])
        self.assertEqual(extract_code_blocks(""), [])
        self.assertEqual(extract_code_blocks(None), [])

    def test_unterminated_fence_is_not_run(self):
        self.assertEqual(extract_code_blocks("Reasoning\n```python\nprint(1)\n"), [])

    def test_empty_block_is_skipped(self):
        self.assertEqual(extract_code_blocks("```python\n\n```"), [])

    def test_think_block_is_ignored(self):
        text = "<think>```python\nprint('draft')\n```</think>\n" + PY.format("print('real')")
        self.assertEqual(extract_code_blocks(text), ["print('real')"])

    def test_crlf_line_endings(self):
        self.assertEqual(extract_code_blocks("```python\r\nprint(2)\r\n```"), ["print(2)"])

    def test_indented_code_keeps_relative_indentation(self):
        text = PY.format("for i in range(2):\n    print(i)")
        self.assertEqual(extract_code_blocks(text), ["for i in range(2):\n    print(i)"])


class TestObservationFormat(unittest.TestCase):
    def test_sections_and_budget_notes(self):
        ex = ExecResult(stdout="3\n", stderr="warn\n", error="ValueError: x")
        text = format_observation(ex, turn=2, max_turns=8)
        self.assertIn("Result of turn 2/8:", text)
        self.assertIn("stdout:\n3", text)
        self.assertIn("stderr:\nwarn", text)
        self.assertIn("error:\nValueError: x", text)
        self.assertNotIn("turns left", text)

    def test_last_turn_warning_and_no_output(self):
        text = format_observation(ExecResult(), turn=7, max_turns=8)
        self.assertIn("(no output)", text)
        self.assertIn("LAST turn", text)
        near_end = format_observation(ExecResult(), turn=6, max_turns=8)
        self.assertIn("2 turns left, including the next one.", near_end)
        self.assertNotIn("LAST turn", near_end)

    def test_timeout_message_is_passed_through(self):
        ex = ExecResult(error="TimeoutError: slow", timed_out=True)
        self.assertIn("TimeoutError: slow", format_observation(ex, turn=1, max_turns=4))

    def test_huge_output_is_capped_and_marked(self):
        text = format_observation(ExecResult(stdout="z" * 50000), turn=1, max_turns=4)
        self.assertLessEqual(len(text), 8000 + 100)
        self.assertIn("characters omitted", text)


class ReactTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.split = self.base / "split"
        (self.split / "data" / "biz_t").mkdir(parents=True)
        (self.split / "data" / "biz_t" / "data.csv").write_text(CSV, encoding="utf-8")
        (self.split / "data" / "biz_t" / "README.md").write_text("Shop test.\n", encoding="utf-8")
        (self.split / "rulebook.md").write_text(RULEBOOK, encoding="utf-8")
        (self.split / "calendar.csv").write_text("date\n2026-01-01\n", encoding="utf-8")
        (self.split / "questions.jsonl").write_text("", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def solver(self, replies, max_turns=4):
        ctx = SolverContext(split="train", questions_path=self.split / "questions.jsonl",
                            answers_path=None, base_dir=self.base, max_turns=max_turns, run_name="unit")
        llm = ScriptedLLM(replies)
        return ReactSolver(ctx, llm=llm, exec_timeout_s=30), llm

    def question(self, qid="q1", biz="biz_t"):
        return {"id": qid, "business_id": biz, "question": "Có bao nhiêu dòng dữ liệu?", "as_of_date": "2026-06-01"}

    def trace(self, qid="q1"):
        return json.loads((self.base / "logs" / "traces" / "unit" / f"{qid}.json").read_text(encoding="utf-8"))


class TestReactLoop(ReactTestCase):
    def test_react_is_registered_as_a_mode(self):
        self.assertIn("react", SOLVERS)

    def test_happy_path_returns_final_answer_and_writes_trace(self):
        code1 = ("import pandas as pd, os\n"
                 "df = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))\n"
                 "print(len(df))")
        code2 = "final_answer('number', int(len(df)))"
        solver, llm = self.solver([f"Count the rows first.\n{PY.format(code1)}",
                                   f"Now answer.\n{PY.format(code2)}"])
        try:
            record = solver.solve(self.question())
        finally:
            solver.close()
        self.assertEqual(record, {"type": "number", "answer": 3})
        self.assertEqual(len(llm.seen), 2)
        self.assertIn("stdout:\n3", llm.seen[1][-1]["content"])
        self.assertIn("Result of turn 1/4", llm.seen[1][-1]["content"])

        tr = self.trace()
        self.assertEqual(tr["outcome"], "final:number")
        self.assertEqual(len(tr["turns"]), 2)
        self.assertEqual(tr["turns"][0]["exec"]["stdout"], "3\n")
        self.assertEqual(tr["turns"][0]["provider"], "PrimeIntellect")
        self.assertEqual(tr["usage_total"]["prompt_tokens"], 20)
        self.assertEqual(tr["messages"][0]["role"], "system")
        self.assertIn("Question again", tr["messages"][1]["content"])
        self.assertEqual(tr["record"], {"type": "number", "answer": 3})

    def test_no_code_reply_gets_corrective_message_then_recovers(self):
        solver, llm = self.solver(["I believe it is 3.", PY.format("final_answer('number', 3)")])
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "number", "answer": 3})
        self.assertEqual(llm.seen[1][-1]["content"], NO_CODE_MESSAGE)
        self.assertEqual(llm.seen[1][-2]["role"], "assistant")
        self.assertEqual(self.trace()["turns"][0]["code"], None)

    def test_two_consecutive_no_code_replies_abstain(self):
        solver, llm = self.solver(["no code", "still none", PY.format("final_answer('number', 1)")])
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "abstain"})
        self.assertEqual(len(llm.seen), 2)
        self.assertEqual(self.trace()["outcome"], "no_code")

    def test_max_turns_abstains(self):
        solver, llm = self.solver([PY.format("print('still looking')")] * 3, max_turns=2)
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "abstain"})
        self.assertEqual(len(llm.seen), 2)
        tr = self.trace()
        self.assertEqual(tr["outcome"], "max_turns")
        self.assertIn("LAST turn", tr["turns"][0]["observation"])

    def test_llm_error_abstains_and_is_recorded(self):
        solver, llm = self.solver([LLMError("PrimeIntellect unavailable")])
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "abstain"})
        tr = self.trace()
        self.assertEqual(tr["outcome"], "llm_error")
        self.assertIn("PrimeIntellect unavailable", tr["turns"][0]["llm_error"])

    def test_invalid_final_is_rejected_and_model_can_retry(self):
        bad = PY.format("final_answer('number', '4,82 triệu')")
        good = PY.format("final_answer('number', 4820000)")
        solver, llm = self.solver([bad, good])
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "number", "answer": 4820000})
        feedback = llm.seen[1][-1]["content"]
        self.assertTrue(feedback.startswith("final_answer was rejected by the format check"), feedback)
        self.assertIn("JSON number", feedback)
        self.assertIn("format_error", self.trace()["turns"][0])

    def test_multiple_blocks_run_only_the_first(self):
        reply = PY.format("print('first block')") + "\n" + PY.format("final_answer('number', 2)")
        solver, llm = self.solver([reply, PY.format("final_answer('number', 2)")])
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "number", "answer": 2})
        obs = llm.seen[1][-1]["content"]
        self.assertIn(MULTI_BLOCK_NOTE.format(n=2), obs)
        self.assertIn("first block", obs)

    def test_unanswerable_with_reason(self):
        solver, _ = self.solver([PY.format("final_answer('unanswerable', reason='shop không ghi trạng thái huỷ')")])
        record = solver.solve(self.question())
        self.assertEqual(record, {"type": "unanswerable", "reason": "shop không ghi trạng thái huỷ"})
        self.assertEqual(self.trace()["outcome"], "final:unanswerable")

    def test_model_abstain_is_a_normal_outcome(self):
        solver, _ = self.solver([PY.format("final_answer('abstain')")])
        self.assertEqual(solver.solve(self.question()), {"type": "abstain"})
        self.assertEqual(self.trace()["outcome"], "final:abstain")

    def test_variables_persist_across_turns(self):
        solver, llm = self.solver([
            PY.format("import pandas as pd\ntotal_rows = len(pd.read_csv(DATA_DIR + '/data.csv'))"),
            PY.format("final_answer('number', total_rows * 10)"),
        ])
        self.assertEqual(solver.solve(self.question()), {"type": "number", "answer": 30})

    def test_missing_shop_folder_raises_and_writes_error_trace(self):
        solver, _ = self.solver([])
        with self.assertRaises(FileNotFoundError):
            solver.solve(self.question(biz="biz_missing"))
        self.assertEqual(self.trace()["outcome"], "error")

    def test_stats_report_lists_outcomes_and_turns(self):
        solver, _ = self.solver([PY.format("final_answer('number', 3)")])
        solver.solve(self.question())
        report = solver.stats_report()
        self.assertIn("Outcomes: final:number=1", report)
        self.assertIn("Avg LLM turns per question: 1.00", report)
        self.assertIn("Traces:", report)


if __name__ == "__main__":
    unittest.main()
