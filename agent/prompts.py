"""
Prompt texts for the ReAct (CodeAct) solver: system protocol, per-question user message,
and the messages the harness sends back to the model (observations and corrections).
"""

from __future__ import annotations



from agent.sandbox import ExecResult, truncate_output

OBSERVATION_LIMIT = 8000

NO_CODE_MESSAGE = (
    "Your reply had no ```python code block, so nothing was run. Reply with a short explanation and "
    "exactly one ```python code block. To finish, call final_answer(...) inside that block."
)

MULTI_BLOCK_NOTE = "(Your reply had {n} code blocks; only the first one was run.)"


def build_system_prompt(max_turns: int) -> str:
    return f"""You answer one business question about one shop, using Python on the shop's raw data files. The question is in Vietnamese. You work in turns.

HOW A TURN WORKS
- Reply with a short explanation, then exactly ONE ```python code block. Only the first code block of a reply is run.
- The code runs in a Python session that keeps variables from earlier turns. Print what you need to see: stdout, stderr and tracebacks come back to you as the next message (long output is cut to about {OBSERVATION_LIMIT} characters, head and tail kept). The value of a trailing bare expression is printed automatically.
- You have at most {max_turns} turns. Use the early turns to inspect files and values (headers, distinct values, date ranges, unit conversions), then compute, then answer.
- When you are sure, call final_answer(...) inside a code block. That ends the task. Example: final_answer("number", 4820000)
- An answer counts only if it is passed to final_answer.

SCORING
- Correct answer +1. Wrong answer -1 (a wrong type or a wrong format counts as wrong). Abstain 0.
- Answer only when you are confident in both the value and the answer type. If the data cannot support a confident answer, call final_answer("abstain").
- final_answer("unanswerable", reason="...") is correct only when the data cannot answer the question at all (rulebook section 7): the shop does not record the information asked for, or the requested period lies completely outside the date range covered by the data. A period that partly overlaps the data is answered from the overlapping part. A period inside the data range with no matching records is answered (as 0 or an empty list), not unanswerable.

PREDEFINED NAMES
- DATA_DIR: this shop's data folder (str). AS_OF_DATE: "today" as "YYYY-MM-DD" (str). AS_OF: the same date as datetime.date.
- CALENDAR_PATH: calendar.csv (solar <-> lunar dates, for Tết and lunar months). RULEBOOK_PATH: the rulebook file (it is also given in full in the user message).
- Already imported: pd (pandas), np (numpy), duckdb, sqlite3, json, re, dt (the datetime module), os, glob.
- final_answer(type, answer=None, reason=None). Types: number, percent, date, boolean, string, list, table, unanswerable, abstain.

ANSWER FORMATS (rulebook section 9; each type has exactly one format)
- number: a JSON number with no unit and no thousands separator. Money is an integer amount in VND, e.g. 4820000. Counts are integers.
- percent: the number of percent with at most 2 decimals: 12.4 means 12.4%. Never 0.124, never "12.4%".
- date: "YYYY-MM-DD", or "YYYY-MM" when the question asks about a month. Never "15/08/2026".
- boolean: true or false.
- string: the name exactly as the data writes it, with full Vietnamese diacritics. No column labels, no quotes around a list.
- list: a JSON array, in the order the question asks for, one item per element, e.g. ["Cà phê sữa", "Bạc xỉu"] or [3, 10]. Never a comma-joined string.
- table: {{"columns": [...], "rows": [[...], ...]}}: the first column is the entity name, the other columns are the measures in the order the question asks; numeric cells are numbers. A pandas DataFrame is also accepted (its index is ignored; call reset_index() first if the names are in the index).
- unanswerable: final_answer("unanswerable", reason="short reason").

CALCULATION
- The rulebook defines revenue, periods, filters, ranking, rounding and unit conversions. Follow it exactly. The wording of the question overrides the rulebook defaults.
- Relative periods (hôm nay, tuần này, tháng này, N ngày gần nhất, ...) are relative to AS_OF_DATE. Records dated after AS_OF_DATE do not count for today-based questions, except order status fields (see the rulebook).
- Round money to the nearest dong, halves up. Use decimal.Decimal with ROUND_HALF_UP: Python's round() uses banker's rounding. Round other averages and percentages to 2 decimals the same way.
- Before answering, check the result against the data (totals, counts, a few rows). Print intermediate tables when a result looks surprising.
"""


def build_user_prompt(question: dict, readme: str, profile: str, rulebook: str, max_turns: int) -> str:
    qtext = question["question"]
    return f"""## Question
- business_id: {question["business_id"]}
- as_of_date (today for this question): {question["as_of_date"]}
- question: {qtext}

## Shop README (written by the shop owner; describes the files, columns and conventions)
{readme.strip()}

## Data profile (generated automatically; cells are raw values, long cells are cut)
{profile.strip()}

## Rulebook (applies to every question; sections 1 to 10)
{rulebook.strip()}

## Now
You have {max_turns} turns. Inspect the files this question needs, compute, and call final_answer(...) as soon as you are sure.
Question again: {qtext}
"""


def format_observation(result: ExecResult, turn: int, max_turns: int, extra_blocks: int = 0) -> str:
    """The message sent back after a turn: stdout, stderr, errors, and the turn budget."""
    parts = [f"Result of turn {turn}/{max_turns}:"]
    if extra_blocks:
        parts.append(MULTI_BLOCK_NOTE.format(n=extra_blocks + 1))
    if result.timed_out or result.crashed:
        parts.append(result.error or "error")
    else:
        if result.stdout:
            parts.append("stdout:\n" + result.stdout.rstrip("\n"))
        if result.stderr:
            parts.append("stderr:\n" + result.stderr.rstrip("\n"))
        if result.error:
            parts.append("error:\n" + result.error.rstrip("\n"))
        if not (result.stdout or result.stderr or result.error):
            parts.append("(no output)")
    remaining = max_turns - turn
    if remaining == 1:
        parts.append("This is your LAST turn: call final_answer(...) in it, or final_answer(\"abstain\").")
    elif 1 < remaining <= 3:
        parts.append(f"{remaining} turns left, including the next one.")
    return truncate_output("\n".join(parts), OBSERVATION_LIMIT)


def format_rejection(reason: str) -> str:
    return (f"final_answer was rejected by the format check: {reason}. Fix the value to match the required "
            "format (see ANSWER FORMATS) and call final_answer again.")


__all__ = ["build_system_prompt", "build_user_prompt", "format_observation", "format_rejection",
           "NO_CODE_MESSAGE", "MULTI_BLOCK_NOTE"]
