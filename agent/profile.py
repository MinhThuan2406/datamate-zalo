"""
Compact, cached data profile of one shop, put into the prompt so the model does not have to
rediscover the file layout with its first few turns.

For every file (README.md excluded, it is shown in full in the prompt separately):
- xlsx: sheet names; for the first sheets, dimensions (when the file records them) and the
  first raw rows, so title rows, blank rows and 2-line headers are visible;
- csv: approximate data-row count, header and 3 sample rows;
- sqlite: tables with column names/types, row count and one sample row.

The whole profile is held under a character budget (about 4k tokens). Files that do not fit
are listed by name only. Results are cached under cache/profiles/, keyed by a fingerprint of
the file names, sizes and mtimes, so a changed or re-downloaded shop is profiled again.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROFILE_VERSION = 1
DEFAULT_BUDGET_CHARS = 12000
XLSX_ROWS = 4                 # raw rows shown per sheet
XLSX_MAX_COLS = 16            # columns shown per row
XLSX_DETAILED_SHEETS = 6      # sheets per workbook that get row samples
CSV_SAMPLE_ROWS = 3
SQLITE_MAX_COLS = 40
CELL_LIMIT = 24
FILE_SHARE_LIMIT = 0.5        # one file may use at most this share of the budget

_KIND_BY_SUFFIX = {".xlsx": "xlsx", ".xlsm": "xlsx", ".csv": "csv",
                   ".sqlite": "sqlite", ".sqlite3": "sqlite", ".db": "sqlite"}


def _kind(path: Path) -> Optional[str]:
    return _KIND_BY_SUFFIX.get(path.suffix.lower())


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def list_data_files(data_dir: Path) -> List[Path]:
    files = []
    for root, dirs, names in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith((".", "~$")) or name.lower() == "readme.md":
                continue
            files.append(Path(root) / name)
    return sorted(files, key=lambda p: str(p.relative_to(data_dir)).lower())


def _render_cell(value: Any, limit: int = CELL_LIMIT) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        text = value.strftime("%Y-%m-%d") if (value.hour, value.minute, value.second) == (0, 0, 0) \
            else value.strftime("%Y-%m-%d %H:%M")
    elif isinstance(value, dt.date):
        text = value.isoformat()
    elif isinstance(value, float):
        text = str(int(value)) if value.is_integer() else repr(round(value, 4))
    else:
        text = str(value)
    text = " ".join(text.split())  # collapse newlines and runs of spaces
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_row(values: Iterable[Any]) -> str:
    cells = [_render_cell(v) for v in values]
    while cells and cells[-1] == "":
        cells.pop()
    if not cells:
        return "(empty row)"
    return " | ".join(cells)


def _xlsx_lines(path: Path) -> List[str]:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001 - a broken file must not break the profile
        return [f"  (could not open workbook: {type(e).__name__})"]
    lines: List[str] = []
    try:
        sheets = list(wb.worksheets)
        names = [ws.title for ws in sheets]
        shown = ", ".join(repr(n) for n in names[:40]) + (" ..." if len(names) > 40 else "")
        lines.append(f"  sheets ({len(names)}): {shown}")
        for ws in sheets[:XLSX_DETAILED_SHEETS]:
            if ws.max_row:
                dims = f"max_row={ws.max_row}, max_col={ws.max_column}"
            else:
                dims = "dimensions not recorded"
            lines.append(f"  [sheet {ws.title!r}] {dims}")
            for idx, row in enumerate(
                    ws.iter_rows(min_row=1, max_row=XLSX_ROWS, max_col=XLSX_MAX_COLS, values_only=True), start=1):
                lines.append(f"    r{idx}: {_render_row(row)}")
        if len(sheets) > XLSX_DETAILED_SHEETS:
            lines.append(f"  (sheets after the first {XLSX_DETAILED_SHEETS} are not sampled; "
                         "they may share the same layout)")
    finally:
        wb.close()
    return lines


def _count_newlines(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def _csv_lines(path: Path) -> List[str]:
    rows: List[List[str]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            for i, row in enumerate(csv.reader(f)):
                rows.append(row)
                if i >= CSV_SAMPLE_ROWS:
                    break
        approx_rows = max(_count_newlines(path) - 1, 0)  # header assumed on line 1
    except (OSError, csv.Error) as e:
        return [f"  (could not read csv: {type(e).__name__})"]
    lines = [f"  ~{approx_rows} data rows (by line count)"]
    for i, row in enumerate(rows):
        label = "header" if i == 0 else f"row{i}"
        lines.append(f"    {label}: {_render_row(row)}")
    return lines


def _sqlite_lines(path: Path) -> List[str]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        return [f"  (could not open sqlite file: {e})"]
    lines: List[str] = []
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        lines.append(f"  tables ({len(tables)}): {', '.join(tables)}")
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            cols = con.execute(f"PRAGMA table_info({quoted})").fetchall()  # cid, name, type, ...
            count = con.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            col_text = ", ".join(f"{c[1]} {c[2]}".strip() for c in cols[:SQLITE_MAX_COLS])
            if len(cols) > SQLITE_MAX_COLS:
                col_text += f", ... (+{len(cols) - SQLITE_MAX_COLS} columns)"
            lines.append(f"  table {table} ({count} rows): {col_text}")
            sample = con.execute(f"SELECT * FROM {quoted} LIMIT 1").fetchone()
            if sample is not None:
                lines.append(f"    sample: {_render_row(sample)}")
    except sqlite3.Error as e:
        lines.append(f"  (sqlite error: {e})")
    finally:
        con.close()
    return lines


_RENDERERS = {"xlsx": _xlsx_lines, "csv": _csv_lines, "sqlite": _sqlite_lines}


def build_profile(data_dir: Path, budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    data_dir = Path(data_dir)
    files = list_data_files(data_dir)
    header = [f"Files of this shop ({len(files)}; README.md is given separately):"]
    for path in files:
        rel = path.relative_to(data_dir).as_posix()
        kind = _kind(path)
        note = kind or "not profiled"
        header.append(f"- {rel} ({note}, {_human_size(path.stat().st_size)})")
    used = sum(len(line) + 1 for line in header)

    blocks: List[str] = []
    omitted: List[str] = []
    max_block = int(budget_chars * FILE_SHARE_LIMIT)
    for path in files:
        kind = _kind(path)
        if kind is None:
            continue
        rel = path.relative_to(data_dir).as_posix()
        body = _RENDERERS[kind](path)
        text = f"## {rel}\n" + "\n".join(body)
        if len(text) > max_block:
            text = text[: max_block - 40].rstrip() + "\n  ...[file detail truncated]"
        if used + len(text) + 2 > budget_chars:
            omitted.append(rel)
            continue
        blocks.append(text)
        used += len(text) + 2
    if omitted:
        blocks.append("## Not shown in detail (budget reached; inspect them in code)\n" + "\n".join(
            f"- {name}" for name in omitted))
    return "\n".join(header) + "\n\n" + "\n\n".join(blocks)


def _fingerprint(data_dir: Path, budget_chars: int) -> str:
    h = hashlib.sha256(f"v{PROFILE_VERSION}|{budget_chars}|".encode("utf-8"))
    for path in list_data_files(data_dir):
        st = path.stat()
        h.update(f"{path.relative_to(data_dir).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
    return h.hexdigest()


def get_profile(data_dir: Path, cache_dir: Optional[Path] = None,
                budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """Builds the profile or reads it from cache. Cache writes are atomic; failures are not fatal."""
    data_dir = Path(data_dir)
    if cache_dir is None:
        return build_profile(data_dir, budget_chars)
    fp = _fingerprint(data_dir, budget_chars)
    label = f"{data_dir.parent.parent.name}__{data_dir.name}"
    path = Path(cache_dir) / f"{label}__{fp[:16]}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        pass
    text = build_profile(data_dir, budget_chars)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
    return text


__all__ = ["build_profile", "get_profile", "list_data_files"]
