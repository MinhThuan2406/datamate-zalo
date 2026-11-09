"""Tests for the rule-based intent classifier (intent.py)."""

import collections
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intent import INTENT_NAMES, classify_intent, fold_text, tokenize  # noqa: E402


class TestTokenize(unittest.TestCase):

    def test_fold_and_tokenize(self):
        self.assertEqual(fold_text("Tồn kho ĐẾN hết"), "ton kho den het")
        self.assertEqual(tokenize("tồn kho còn bao nhiêu?"), ["ton", "kho", "con", "bao", "nhieu"])

    def test_teencode_and_telex(self):
        self.assertEqual(tokenize("dthu hnay bnhiu"), ["doanh", "thu", "hom", "nay", "bao", "nhieu"])
        self.assertEqual(tokenize("toongr tieenf"), ["tong", "tien"])
        self.assertEqual(tokenize("bán dduwowcj"), ["ban", "duoc"])
        self.assertEqual(tokenize("thangs 3 laf"), ["thang", "3", "la"])
        self.assertEqual(tokenize("sp dc"), ["san", "pham", "duoc"])
        self.assertEqual(tokenize("bn order"), ["bao", "nhieu", "don"])


class TestClassifyIntent(unittest.TestCase):

    def assertIntent(self, question, expected):
        self.assertEqual(classify_intent(question), expected, question)

    def test_required_cases(self):
        self.assertIntent("tong dthu thang 8 nam 2025 la bao nhieu", "doanh_thu")
        self.assertIntent("tồn kho còn bao nhiêu", "kho_ton")
        self.assertIntent("tỉ lệ huỷ đơn tháng 6", "ti_le_huy")
        self.assertIntent("top 3 khách nợ nhiều nhất", "cong_no")

    def test_tong_never_matches_ton(self):
        for q in ["tổng doanh thu tháng 8", "Tổng dthu tuần trước", "toongr dthu hôm qua là bn",
                  "tong cong ban duoc bao nhieu tien"]:
            self.assertEqual(classify_intent(q), "doanh_thu", q)

    def test_accented_and_unaccented_agree(self):
        pairs = [
            ("Tồn kho Bia Tiger cuối ngày hôm nay là bao nhiêu lon?",
             "ton kho bia tiger cuoi ngay hom nay la bao nhieu lon"),
            ("Tỉ lệ huỷ đơn tháng 6/2025 là bao nhiêu %?", "ti le huy don thang 6/2025 la bao nhieu %"),
            ("Khách còn nợ cửa hàng tổng cộng bao nhiêu?", "khach con no cua hang tong cong bao nhieu"),
        ]
        for a, b in pairs:
            self.assertEqual(classify_intent(a), classify_intent(b), (a, b))

    def test_examples_per_intent(self):
        cases = {
            "cong_no": ["Tháng 6/2026 cửa hàng thu được bao nhiêu tiền khách trả nợ?",
                        "cho bảng 3 khách còn nowj nhiều nhất"],
            "ti_le_huy": ["thang 6/2025 bi huy may don", "Tháng 3/2026 tỉ lệ đơn bị huỷ là bao nhiêu phần trăm?"],
            "han_dung_lo": ["Lô Oresol gần hết hạn nhất có hạn dùng ngày nào?"],
            "giao_hang": ["Tháng 4/2026 xe đã giao tổng cộng mấy đợt hàng?"],
            "kho_ton": ["Cuối ngày hôm nay Bánh tiramisu còn bao nhiêu cái vậy?",
                        "Có bao nhiêu mặt hàng vẫn đang bán mà trong 7 ngày gần nhất không bán ra được cái nào?"],
            "nhap_hang": ["Tháng 9/2025 tổng giá trị các phiếu nhập hàng là bao nhiêu?"],
            "lai_gia_von_chi_phi": ["Tháng 4/2025 lãi gộp của cửa hàng là bao nhiêu?",
                                    "hàng nào có lãi gộp cao nhất tháng 1/2026"],
            "so_sanh_ky": ["dthu tháng 2/2026 tăng hay giảm bao nhiêu phần trăm so với tháng trước?",
                           "tu thang 9/2025 den thang 11/2025, doanh thu thang sau co luon cao hon thang trc khong"],
            "top_k_xep_hang": ["tuan roi top 5 hang ban chay nhat la gi",
                               "tháng 5/2025 ngày nào cửa hàng bán được nhìu tiền nhất"],
            "trung_binh_don": ["Giá trị trung bình một đơn tuần trước là bao nhiêu?"],
            "so_don": ["Hôm qua cửa hàng bán được mấy đơn?",
                       "trong thang 10 nam 2025, so khach mua it nhat 4 don la bao nhieu"],
            "doanh_thu": ["Cho tôi biết doanh thu tháng 5/2025.", "30 ngày gần nhất dthu được bao nhiêu?",
                          "Tháng 8/2025 bán ra bao nhiêu đĩa Lòng mề gà xào sa tế ạ?"],
            "khac": ["Nhân viên Phùng Thu Trang làm việc có tận tâm với khách không?", "", None],
        }
        for expected, questions in cases.items():
            for q in questions:
                self.assertIntent(q, expected)

    def test_train_distribution_is_balanced(self):
        q_path = ROOT / "train" / "questions.jsonl"
        if not q_path.is_file():
            self.skipTest("train/questions.jsonl not available")
        with open(q_path, encoding="utf-8") as f:
            qs = [json.loads(line)["question"] for line in f if line.strip()]
        dist = collections.Counter(classify_intent(q) for q in qs)
        self.assertTrue(set(dist) <= set(INTENT_NAMES))
        self.assertLess(max(dist.values()) / len(qs), 0.45)
        self.assertLess(dist["khac"] / len(qs), 0.05)
        tong = [q for q in qs if fold_text(q).startswith("tong ")]
        self.assertTrue(tong)
        self.assertFalse([q for q in tong if classify_intent(q) == "kho_ton"])


if __name__ == "__main__":
    unittest.main()
