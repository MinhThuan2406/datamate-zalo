"""
DataMate Agentic Harness — Main Execution Runner.

Iterates questions, asks a pluggable *solver* for an answer record, validates
the record against the §9 format (formatter.validate_format), and writes the
submission JSONL. Optionally evaluates the result on train.

Adding a solver
---------------
Subclass ``Solver`` (or implement anything with ``name`` and ``solve``) and
register a factory under a mode name::

    @register_solver("react")
    class ReactSolver(Solver):
        name = "react"
        def __init__(self, ctx: SolverContext): ...
        def solve(self, question: dict) -> dict:
            return {"type": "number", "answer": 5}      # no "id"

``solve`` receives the raw question row (id, business_id, question,
as_of_date, ...) and returns a record without id. Exceptions are caught by
the runner and turned into ``{"type": "abstain"}``; malformed records are
downgraded to abstain as well. The factory receives a ``SolverContext``.

Examples
--------
    python run.py --mode abstain --evaluate -o out/a.jsonl
    python run.py --mode oracle_check --shops biz_101 --limit 20 --evaluate -o out/p.jsonl
    python run.py --mode react --resume --log out/react.log.jsonl -o out/react.jsonl
    python run.py --mode react --workers 4 --max-turns 8 --run-name smoke1 -o out/react.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set, runtime_checkable

from evaluator import Evaluator, save_report_json
from formatter import validate_format

if __name__ == "__main__":
    # agent/react.py does `from run import ...`. Make that reuse this running module instead of
    # executing run.py a second time, which would build a second solver registry.
    sys.modules.setdefault("run", sys.modules[__name__])

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Solver interface & registry
# -----------------------------------------------------------------------------


@dataclass
class SolverContext:
    """Everything a solver factory may need to build itself."""

    split: str
    questions_path: Path
    answers_path: Optional[Path]
    base_dir: Path = BASE_DIR
    max_turns: Optional[int] = None   # --max-turns (None: solver default)
    run_name: str = "default"         # --run-name (names per-run outputs such as trace dirs)

    @property
    def data_dir(self) -> Path:
        return self.questions_path.parent


@runtime_checkable
class SolverProtocol(Protocol):
    name: str

    def solve(self, question: Dict[str, Any]) -> Dict[str, Any]: ...


class Solver:
    """Base class for solvers. ``solve`` returns a §9 record without "id"."""

    name: str = "base"

    def __init__(self, ctx: SolverContext):
        self.ctx = ctx

    def solve(self, question: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (DB handles, clients). Called once after the run."""

    def stats_report(self) -> Optional[str]:
        """Optional text printed after the run (token usage, outcomes, ...)."""
        return None


SolverFactory = Callable[[SolverContext], SolverProtocol]
SOLVERS: Dict[str, SolverFactory] = {}


def register_solver(mode: str) -> Callable[[SolverFactory], SolverFactory]:
    """Decorator: make a solver class/factory available as ``--mode <mode>``."""

    def deco(factory: SolverFactory) -> SolverFactory:
        if mode in SOLVERS:
            raise ValueError(f"Solver mode {mode!r} already registered")
        SOLVERS[mode] = factory
        return factory

    return deco


def build_solver(mode: str, ctx: SolverContext) -> SolverProtocol:
    try:
        factory = SOLVERS[mode]
    except KeyError:
        raise ValueError(f"Unknown mode {mode!r}; available: {', '.join(sorted(SOLVERS))}") from None
    return factory(ctx)


@register_solver("abstain")
class AbstainSolver(Solver):
    """Baseline: abstain on everything (score 0)."""

    name = "abstain"

    def solve(self, question: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "abstain"}


@register_solver("oracle_check")
class OracleSolver(Solver):
    """Answers from ground truth; sanity check for the runner + evaluator (should score 100%)."""

    name = "oracle_check"

    def __init__(self, ctx: SolverContext):
        super().__init__(ctx)
        if ctx.answers_path is None or not ctx.answers_path.is_file():
            raise FileNotFoundError(f"oracle_check requires ground truth answers at {ctx.answers_path}")
        self.answers: Dict[str, Dict[str, Any]] = {r["id"]: r for r in read_jsonl(ctx.answers_path)}

    def solve(self, question: Dict[str, Any]) -> Dict[str, Any]:
        gt = self.answers[question["id"]]  # KeyError -> runner abstains
        if gt["type"] == "unanswerable":
            return {"type": "unanswerable", "reason": "oracle"}
        return {"type": gt["type"], "answer": gt.get("answer")}


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_done_ids(path: Path) -> Set[str]:
    """Ids already present in an existing output file (unparsable lines are ignored)."""
    done: Set[str] = set()
    if not path.is_file():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                done.add(rec["id"])
    return done


def select_questions(
    questions: Iterable[Dict[str, Any]],
    shops: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter by business_id (if shops given), then keep the first `limit` (file order)."""
    selected = [q for q in questions if not shops or q.get("business_id") in shops]
    if limit is not None:
        selected = selected[: max(limit, 0)]
    return selected


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
# Core loop
# -----------------------------------------------------------------------------


def solve_one(solver: SolverProtocol, question: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one question through the solver and the format gate.

    Returns {"record": <submission row with id>, "error": str|None,
             "format_error": str|None, "elapsed_s": float}.
    """
    qid = question["id"]
    t0 = time.perf_counter()
    error: Optional[str] = None
    try:
        raw = solver.solve(question)
    except Exception as e:  # noqa: BLE001 - any solver failure must become an abstain
        error = f"{type(e).__name__}: {e}"
        _log(f"[!] {qid}: solver raised {error}; abstaining")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__, limit=3)).rstrip())
        raw = {"type": "abstain"}
    elapsed = time.perf_counter() - t0

    record, format_error = validate_format(raw)
    if format_error:
        _log(f"[!] {qid}: invalid record downgraded to abstain: {format_error}")
    return {
        "record": {"id": qid, **record},
        "error": error,
        "format_error": format_error,
        "elapsed_s": elapsed,
    }


def run(
    solver: SolverProtocol,
    questions: List[Dict[str, Any]],
    output_path: Path,
    mode: str,
    resume: bool = False,
    log_path: Optional[Path] = None,
    workers: int = 1,
) -> Dict[str, int]:
    """Solves `questions` and writes the submission. Returns run statistics.

    workers > 1 solves questions in a thread pool; the solver must then be thread-safe.
    Results are written as they complete, under a lock, so the output stays resumable.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done_ids(output_path) if resume else set()
    todo = [q for q in questions if q["id"] not in done]
    stats = {"selected": len(questions), "skipped": len(questions) - len(todo), "written": 0,
             "errors": 0, "format_errors": 0}

    fmode = "a" if resume else "w"
    log_f = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, fmode, encoding="utf-8")
    write_lock = threading.Lock()

    def commit(q: Dict[str, Any], res: Dict[str, Any]) -> None:
        with write_lock:
            out_f.write(json.dumps(res["record"], ensure_ascii=False) + "\n")
            out_f.flush()  # keep the file resumable after a crash
            stats["written"] += 1
            stats["errors"] += res["error"] is not None
            stats["format_errors"] += res["format_error"] is not None
            if log_f is not None:
                log_f.write(json.dumps({
                    "id": q["id"],
                    "mode": mode,
                    "elapsed_s": round(res["elapsed_s"], 6),
                    "type": res["record"]["type"],
                    "error": res["error"],
                    "format_error": res["format_error"],
                }, ensure_ascii=False) + "\n")
                log_f.flush()

    try:
        with open(output_path, fmode, encoding="utf-8") as out_f:
            if workers <= 1:
                for q in todo:
                    commit(q, solve_one(solver, q))
            else:
                pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="solve")
                try:
                    futures = {pool.submit(solve_one, solver, q): q for q in todo}
                    for fut in as_completed(futures):
                        commit(futures[fut], fut.result())
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)  # cancel pending work on error
    finally:
        if log_f is not None:
            log_f.close()
    return stats


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DataMate Harness Runner (run.py)")
    parser.add_argument("--split", choices=["train", "public_test"], default="train",
                        help="Dataset split to run on (default: train)")
    parser.add_argument("--mode", choices=sorted(SOLVERS), default="abstain",
                        help="Solver to run (default: abstain). Register more with @register_solver.")
    parser.add_argument("--output", "-o", default="submission.jsonl",
                        help="Output submission path (default: submission.jsonl)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluator after generating the submission (needs answers.jsonl)")
    parser.add_argument("--questions", default=None, help="Custom questions.jsonl path (overrides --split default)")
    parser.add_argument("--answers", default=None, help="Custom answers.jsonl path (for evaluation or oracle_check)")
    parser.add_argument("--output-report-json", default=None, help="Optional path to save evaluation report as JSON")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N selected questions")
    parser.add_argument("--shops", default=None,
                        help="Comma-separated business_ids to run, e.g. biz_101,biz_102")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ids already in --output and append to it instead of overwriting")
    parser.add_argument("--log", default=None, help="Optional per-question JSONL log path")
    parser.add_argument("--workers", type=int, default=1,
                        help="Questions solved in parallel with a thread pool (default: 1)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Turn budget for looping solvers such as 'react' (default: solver setting)")
    parser.add_argument("--run-name", default=None,
                        help="Name for per-run outputs such as logs/traces/<run-name>/ (default: output file stem)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    questions_path = Path(args.questions) if args.questions else BASE_DIR / args.split / "questions.jsonl"
    answers_path = Path(args.answers) if args.answers else BASE_DIR / args.split / "answers.jsonl"
    output_path = Path(args.output)
    log_path = Path(args.log) if args.log else None

    if not questions_path.is_file():
        _log(f"Error: Questions file not found at {questions_path}")
        return 1

    if args.workers < 1:
        _log("Error: --workers must be >= 1")
        return 1
    if args.max_turns is not None and args.max_turns < 1:
        _log("Error: --max-turns must be >= 1")
        return 1

    shops = {s.strip() for s in args.shops.split(",") if s.strip()} if args.shops else None
    all_questions = read_jsonl(questions_path)
    questions = select_questions(all_questions, shops=shops, limit=args.limit)
    filtered = len(questions) != len(all_questions)

    print(f"[*] Split: {args.split}")
    print(f"[*] Mode: {args.mode}")
    print(f"[*] Questions: {questions_path} ({len(questions)}/{len(all_questions)} selected)")
    print(f"[*] Output: {output_path}{' (resume/append)' if args.resume else ''}")
    if shops:
        unknown_shops = shops - {q.get('business_id') for q in all_questions}
        if unknown_shops:
            _log(f"[!] Unknown shop id(s): {', '.join(sorted(unknown_shops))}")

    run_name = args.run_name or output_path.stem
    ctx = SolverContext(split=args.split, questions_path=questions_path,
                        answers_path=answers_path if answers_path.is_file() else None,
                        max_turns=args.max_turns, run_name=run_name)
    try:
        solver = build_solver(args.mode, ctx)
    except Exception as e:  # noqa: BLE001
        _log(f"Error: could not build solver {args.mode!r}: {e}")
        return 1

    t0 = time.perf_counter()
    try:
        stats = run(solver, questions, output_path, mode=args.mode, resume=args.resume, log_path=log_path,
                    workers=args.workers)
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
    print(f"[+] Wrote {stats['written']} rows ({stats['skipped']} skipped as already done) -> {output_path} "
          f"in {time.perf_counter() - t0:.2f}s; solver errors: {stats['errors']}, "
          f"format downgrades: {stats['format_errors']}")
    report_stats = getattr(solver, "stats_report", None)
    stats_text = report_stats() if callable(report_stats) else None
    if stats_text:
        print(f"\n[*] Solver stats ({args.mode}, run {run_name}):\n{stats_text}")

    if args.evaluate:
        if not answers_path.is_file():
            _log(f"[!] Cannot evaluate: Ground truth answers file not found at {answers_path}")
            return 1
        print(f"\n[*] Evaluating submission against {answers_path}...")
        evaluator = Evaluator(answers=answers_path, questions=questions_path)
        eval_ids = None
        if filtered:
            eval_ids = [q["id"] for q in questions]
            print(f"[*] Note: scoring only the {len(eval_ids)} selected question(s) "
                  f"(--limit/--shops); unselected questions are excluded, not counted as abstain.")
        report = evaluator.evaluate_submission(output_path, ids=eval_ids)
        print(report.summary_table())

        if args.output_report_json:
            out = save_report_json(report, args.output_report_json)
            print(f"[+] Saved evaluation JSON to {out}")
    return 0


# Solvers that live in the agent package. Imported last, because agent.react imports
# Solver / SolverContext / register_solver from this module.
from agent import react as _agent_react  # noqa: E402,F401


if __name__ == "__main__":
    sys.exit(main())
