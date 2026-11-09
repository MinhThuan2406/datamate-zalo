"""Tests for agent.profile: one xlsx shop, one csv shop and the sqlite shop from train, plus synthetic files."""

import datetime as dt
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.profile import DEFAULT_BUDGET_CHARS, _render_row, build_profile, get_profile  # noqa: E402

TRAIN_DATA = ROOT / "train" / "data"
XLSX_SHOP = TRAIN_DATA / "biz_106"   # xlsx only
CSV_SHOP = TRAIN_DATA / "biz_101"    # one csv file
SQLITE_SHOP = TRAIN_DATA / "biz_123" # one pos.sqlite file


@unittest.skipUnless(XLSX_SHOP.is_dir() and CSV_SHOP.is_dir() and SQLITE_SHOP.is_dir(),
                     "train data not present")
class TestRealShops(unittest.TestCase):
    def test_csv_shop(self):
        text = build_profile(CSV_SHOP)
        self.assertIn("## ban_hang.csv", text)
        self.assertIn("Nội dung", text)          # header row shown raw
        self.assertIn("data rows", text)
        self.assertNotIn("- README.md", text)    # README is given separately in the prompt
        self.assertLess(len(text), DEFAULT_BUDGET_CHARS)

    def test_xlsx_shop(self):
        text = build_profile(XLSX_SHOP)
        self.assertIn("## DanhSachHoaDon.xlsx", text)
        self.assertIn("sheets (1): 'Sheet1'", text)
        self.assertIn("DANH SÁCH HÓA ĐƠN", text)  # title row above the header is visible
        self.assertIn("Mã hóa đơn", text)         # header row
        self.assertIn("ORD000001", text)          # first data row
        self.assertLessEqual(len(text), DEFAULT_BUDGET_CHARS + 500)

    def test_sqlite_shop(self):
        text = build_profile(SQLITE_SHOP)
        self.assertIn("tables (17)", text)
        self.assertIn("table don_hang", text)
        self.assertIn("TongThanhToan", text)
        self.assertIn("table chi_tiet_don (", text)
        self.assertIn("rows):", text)
        self.assertLessEqual(len(text), DEFAULT_BUDGET_CHARS)

    def test_profile_is_cached_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = get_profile(CSV_SHOP, cache)
            second = get_profile(CSV_SHOP, cache)
            self.assertEqual(first, second)
            self.assertEqual(len(list(cache.glob("*.txt"))), 1)


class TestSyntheticShop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "shop"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _xlsx(self, name, sheets):
        from openpyxl import Workbook

        wb = Workbook()
        wb.remove(wb.active)
        for title, rows in sheets:
            ws = wb.create_sheet(title)
            for row in rows:
                ws.append(row)
        wb.save(self.root / name)

    def test_xlsx_title_rows_dates_and_sheet_limit(self):
        rows = [["BÁO CÁO NGÀY"], [], ["Ngày", "Hàng", "SL"], [dt.datetime(2025, 5, 16), "Cà phê", 2.5]]
        sheets = [(f"T{m:02d}", rows) for m in range(1, 9)]  # 8 sheets: only the first 6 get samples
        self._xlsx("bao_cao.xlsx", sheets)
        text = build_profile(self.root)
        self.assertIn("sheets (8)", text)
        self.assertIn("[sheet 'T01']", text)
        self.assertIn("r1: BÁO CÁO NGÀY", text)
        self.assertIn("r2: (empty row)", text)
        self.assertIn("r4: 2025-05-16 | Cà phê | 2.5", text)
        self.assertIn("sheets after the first 6 are not sampled", text)
        self.assertNotIn("[sheet 'T07']", text)

    def test_csv_counts_and_header(self):
        lines = ["Mã,Tên,Giá"] + [f"M{i},Hàng {i},{i * 1000}" for i in range(1, 51)]
        (self.root / "hang.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        text = build_profile(self.root)
        self.assertIn("~50 data rows", text)
        self.assertIn("header: Mã | Tên | Giá", text)
        self.assertIn("row3: M3 | Hàng 3 | 3000", text)

    def test_broken_file_is_reported_not_fatal(self):
        (self.root / "hong.xlsx").write_bytes(b"not a zip file")
        (self.root / "ghi_chu.docx").write_bytes(b"x")
        text = build_profile(self.root)
        self.assertIn("could not open workbook", text)
        self.assertIn("ghi_chu.docx (not profiled", text)

    def test_budget_moves_big_files_to_a_name_list(self):
        for i in range(40):
            (self.root / f"file_{i:02d}.csv").write_text(
                "a,b,c\n" + "\n".join(f"{x},{'y' * 60},{x}" for x in range(20)) + "\n", encoding="utf-8")
        text = build_profile(self.root, budget_chars=3000)
        self.assertLessEqual(len(text), 3000 + 600)
        self.assertIn("Not shown in detail", text)
        self.assertIn("file_39.csv", text)  # every file is still named somewhere

    def test_cache_key_changes_when_a_file_changes(self):
        (self.root / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as cache_tmp:
            cache = Path(cache_tmp)
            before = get_profile(self.root, cache)
            time.sleep(0.01)
            (self.root / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
            after = get_profile(self.root, cache)
            self.assertNotEqual(before, after)
            self.assertEqual(len(list(cache.glob("*.txt"))), 2)

    def test_render_row_trims_and_cuts(self):
        self.assertEqual(_render_row(["a", None, "", None]), "a")
        self.assertEqual(_render_row([]), "(empty row)")
        self.assertEqual(_render_row([1.0, 2.5]), "1 | 2.5")
        long = _render_row(["x" * 100])
        self.assertLessEqual(len(long), 24)
        self.assertTrue(long.endswith("…"))
        self.assertEqual(_render_row([dt.datetime(2026, 1, 2, 9, 30)]), "2026-01-02 09:30")


if __name__ == "__main__":
    unittest.main()
