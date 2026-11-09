"""
ReAct (CodeAct) solver, registered as ``--mode react``.

Per question: a fresh Python sandbox (one worker process) and a message list. Each turn the
LLM replies with text and one ```python block; the block runs in the sandbox and its output
goes back as the next user message. The loop ends when final_answer() is called with a
record that passes formatter.check_format, when max_turns is used up, when the model gives
no code twice in a row, or when the LLM call fails. Every outcome except the last-resort
exception path maps to a §9 record; anything unexpected raises and run.py turns it into an
abstain.

A full trace (messages, code, outputs, usage, provider, timings) is written to
``logs/traces/<run_name>/<question id>.json``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent.llm import LLMClient, LLMConfig, LLMError, Usage
from agent.profile import get_profile
from agent.prompts import (
    NO_CODE_MESSAGE,
    build_system_prompt,
    build_user_prompt,
    format_observation,
    format_rejection,
)
from agent.sandbox import DEFAULT_EXEC_TIMEOUT_S, ExecResult, PythonSandbox
from formatter import check_format, validate_format
from run import Solver, SolverContext, register_solver

DEFAULT_MAX_TURNS = 8
NO_CODE_LIMIT = 2  # consecutive replies without a code block before the question is abstained

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_blocks(text: str) -> List[str]:
    """All non-empty ```python / ```py / ``` blocks, in order. Unterminated fences are ignored
    (a truncated reply is not run). Inline <think> blocks are removed first."""
    visible = _THINK_BLOCK.sub("", (text or "").replace("\r\n", "\n"))
    blocks = [m.group(1).strip("\n") for m in _CODE_FENCE.finditer(visible)]
    return [b for b in blocks if b.strip()]


def strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


def _to_record(final: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (clean §9 record, None) or (None, format reason)."""
    reason = check_format(final)
    if reason is not None:
        return None, reason
    record, _ = validate_format(final)
    return record, None


def _resolve_max_turns(value: Optional[int]) -> int:
    if value is None:
        raw = os.environ.get("AGENT_MAX_TURNS", "").strip()
        value = int(raw) if raw else DEFAULT_MAX_TURNS
    if value < 1:
        raise ValueError(f"max_turns must be >= 1, got {value}")
    return value


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:120] or "default"


@register_solver("react")
class ReactSolver(Solver):
    """LLM writes Python, the sandbox runs it, the observation is fed back, up to max_turns."""

    name = "react"

    def __init__(
        self,
        ctx: SolverContext,
        *,
        llm: Optional[LLMClient] = None,
        config: Optional[LLMConfig] = None,
        exec_timeout_s: Optional[float] = None,
    ) -> None:
        super().__init__(ctx)
        load_dotenv(ctx.base_dir / ".env", override=False)
        self.split_dir = ctx.questions_path.parent
        self.calendar_path = self.split_dir / "calendar.csv"
        self.rulebook_path = self.split_dir / "rulebook.md"
        if not self.rulebook_path.is_file():
            raise FileNotFoundError(f"rulebook not found: {self.rulebook_path}")
        self.rulebook = self.rulebook_path.read_text(encoding="utf-8")
        self.max_turns = _resolve_max_turns(ctx.max_turns)
        self.exec_timeout_s = exec_timeout_s or float(os.environ.get("AGENT_EXEC_TIMEOUT_S") or DEFAULT_EXEC_TIMEOUT_S)
        self.run_name = _safe_name(ctx.run_name)
        self.trace_dir = ctx.base_dir / "logs" / "traces" / self.run_name
        self.profile_cache = ctx.base_dir / "cache" / "profiles"
        if llm is None:
            cfg = config or LLMConfig.from_env(cache_dir=ctx.base_dir / "cache" / "llm")
            llm = LLMClient(cfg)
            llm.require_api_key()
        self.llm = llm
        self._lock = threading.Lock()
        self._profiles: Dict[str, str] = {}
        self._outcomes: Counter = Counter()
        self._turns: List[int] = []

    # -- Solver interface ---------------------------------------------------

    def solve(self, question: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(question["id"])
        biz = str(question["business_id"])
        as_of = str(question["as_of_date"])
        trace: Dict[str, Any] = {
            "id": qid,
            "business_id": biz,
            "as_of_date": as_of,
            "question": question["question"],
            "run_name": self.run_name,
            "max_turns": self.max_turns,
            "outcome": None,
            "record": None,
            "error": None,
            "turns": [],
        }
        messages: List[Dict[str, str]] = []
        record: Optional[Dict[str, Any]] = None
        t0 = time.perf_counter()
        try:
            data_dir = self.split_dir / "data" / biz
            if not data_dir.is_dir():
                raise FileNotFoundError(f"shop data folder not found: {data_dir}")
            readme_path = data_dir / "README.md"
            readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else "(no README for this shop)"
            profile = self._profile(biz, data_dir)
            messages.extend([
                {"role": "system", "content": build_system_prompt(self.max_turns)},
                {"role": "user", "content": build_user_prompt(question, readme, profile, self.rulebook,
                                                              self.max_turns)},
            ])
            with PythonSandbox(data_dir, as_of_date=as_of, calendar_path=self.calendar_path,
                               rulebook_path=self.rulebook_path, timeout_s=self.exec_timeout_s) as sandbox:
                record, trace["outcome"] = self._loop(messages, sandbox, trace)
        except Exception as e:
            trace["outcome"] = "error"
            trace["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            trace["record"] = record or {"type": "abstain"}
            trace["elapsed_s"] = round(time.perf_counter() - t0, 3)
            trace["usage_total"] = _sum_usage(trace["turns"]).to_dict()
            trace["messages"] = messages
            self._write_trace(qid, trace)
            with self._lock:
                outcome = trace["outcome"] or "unknown"
                self._outcomes[outcome] += 1
                self._turns.append(len(trace["turns"]))
        return record or {"type": "abstain"}

    def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()

    def stats_report(self) -> Optional[str]:
        lines = []
        stats = getattr(self.llm, "stats", None)
        if stats is not None:
            lines.append(stats.summary())
        with self._lock:
            outcomes = dict(self._outcomes)
            turns = list(self._turns)
        if outcomes:
            lines.append("Outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
        if turns:
            lines.append(f"Avg LLM turns per question: {sum(turns) / len(turns):.2f} (max {self.max_turns})")
        lines.append(f"Traces: {self.trace_dir}")
        return "\n".join(lines)

    # -- internals ----------------------------------------------------------

    def _profile(self, biz: str, data_dir: Path) -> str:
        with self._lock:
            cached = self._profiles.get(biz)
        if cached is not None:
            return cached
        text = get_profile(data_dir, self.profile_cache)
        with self._lock:
            return self._profiles.setdefault(biz, text)

    def _loop(self, messages: List[Dict[str, str]], sandbox: PythonSandbox,
              trace: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        no_code_streak = 0
        for turn in range(1, self.max_turns + 1):
            step: Dict[str, Any] = {"turn": turn}
            trace["turns"].append(step)
            try:
                res = self.llm.chat(messages)
            except LLMError as e:
                step["llm_error"] = str(e)
                return None, "llm_error"
            visible = strip_think(res.text)
            step.update(
                provider=res.provider, model=res.model, finish_reason=res.finish_reason,
                cached=res.cached, latency_s=round(res.latency_s, 3), attempts=res.attempts,
                usage=res.usage.to_dict(), cost_usd=res.cost_usd, reasoning=res.reasoning_text,
                assistant=visible,
            )
            messages.append({"role": "assistant", "content": visible})

            blocks = extract_code_blocks(res.text)
            if not blocks:
                no_code_streak += 1
                step["code"] = None
                if no_code_streak >= NO_CODE_LIMIT:
                    return None, "no_code"
                step["observation"] = NO_CODE_MESSAGE
                messages.append({"role": "user", "content": NO_CODE_MESSAGE})
                continue

            no_code_streak = 0
            step["code"] = blocks[0]
            extra_blocks = len(blocks) - 1
            ex = sandbox.exec(blocks[0])
            step["exec"] = _exec_dict(ex)
            if ex.final is not None:
                record, reason = _to_record(ex.final)
                if record is not None:
                    step["observation"] = None
                    return record, f"final:{record['type']}"
                step["format_error"] = reason
                obs = format_rejection(reason or "invalid record")
            else:
                obs = format_observation(ex, turn, self.max_turns, extra_blocks)
            step["observation"] = obs
            messages.append({"role": "user", "content": obs})
        return None, "max_turns"

    def _write_trace(self, qid: str, trace: Dict[str, Any]) -> None:
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"{_safe_name(qid)}.json"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(trace, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            print(f"[!] could not write trace for {qid}: {e}", flush=True)


def _exec_dict(ex: ExecResult) -> Dict[str, Any]:
    return {
        "stdout": ex.stdout,
        "stderr": ex.stderr,
        "error": ex.error,
        "final": ex.final,
        "timed_out": ex.timed_out,
        "crashed": ex.crashed,
        "elapsed_s": round(ex.elapsed_s, 3),
    }


def _sum_usage(turns: List[Dict[str, Any]]) -> Usage:
    total = Usage()
    for step in turns:
        u = step.get("usage")
        if isinstance(u, dict):
            total = total + Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                reasoning_tokens=u.get("reasoning_tokens", 0),
                cached_prompt_tokens=u.get("cached_prompt_tokens", 0),
                openrouter_cost_usd=u.get("openrouter_cost_usd", 0.0),
            )
    return total


__all__ = ["ReactSolver", "extract_code_blocks", "strip_think", "DEFAULT_MAX_TURNS"]
