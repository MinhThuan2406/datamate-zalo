"""
Rule-based intent classifier for DataMate questions.

Used by the evaluator to break scores down by business intent. It is a
heuristic bucketing tool, not a solver: it only needs to be stable and
roughly right so that per-intent accuracy numbers are meaningful.

Pipeline
--------
1. ``fold_text``: lowercase, NFD + drop combining marks (Vietnamese diacritics),
   map đ -> d. Accented and unaccented questions end up identical
   ("tồn kho" and "ton kho" -> "ton kho").
2. ``tokenize``: split on non-alphanumerics, then per token
     a. undo common Telex typing slips ("toongr" -> "tong", "dduwowcj" -> "duoc",
        "thangs" -> "thang", "laf" -> "la");
     b. expand teencode / abbreviations ("dthu" -> "doanh thu",
        "bnhiu" -> "bao nhieu", "hnay" -> "hom nay", "sp" -> "san pham",
        "dc" -> "duoc", "dh" -> "don hang", "vs" -> "voi", ...).
3. Keywords are whole words or word sequences matched against the token
   stream (never raw substrings), so "ton" no longer matches "tong" and
   "no" no longer matches "nong"/"noi".

Intents and precedence
----------------------
Rules are tried in the order of ``INTENTS``; the first match wins. The order
puts *domain-specific* metrics first (they need their own rulebook section
regardless of how the question is phrased), then *operations* that sit on top
of generic sales numbers (period comparison, ranking), then the generic sales
metrics, most specific first:

  1. cong_no              debt, repayments, aging, installments (§4)
  2. ti_le_huy            cancellation rate and cancelled-order counts (§1)
  3. han_dung_lo          expiry dates / lots (§6 pharmacy)
  4. giao_hang            delivery trips (§6 VLXD)
  5. kho_ton              stock on hand, stock value, out-of-stock, slow movers (§5)
  6. nhap_hang            purchase receipts, suppliers, avg purchase price,
                          purchase-minus-sales (§5)
  7. lai_gia_von_chi_phi  gross / net profit, COGS, expenses (§5)
  8. so_sanh_ky           period comparisons: tăng/giảm, chênh lệch, so với,
                          % change, "tăng liên tục", price-increase effect (§1)
  9. top_k_xep_hang       top-k, "... nhất", liệt kê, xếp hạng, peak hour (§3)
 10. trung_binh_don       average order value / average per ticket (§3)
 11. so_don               counts of orders / customers / tickets / visits /
                          distinct products (§1)
 12. doanh_thu            revenue, amounts sold, quantity sold, share of
                          revenue, selling price (§1) -- the generic sales bucket
 13. khac                 anything else

Examples: "top 3 khách nợ nhiều nhất" -> cong_no (debt beats ranking);
"hàng nào có lãi gộp cao nhất" -> lai_gia_von_chi_phi;
"doanh thu tháng 6 tăng bao nhiêu % so với tháng trước" -> so_sanh_ky;
"ngày nào bán được nhiều tiền nhất" -> top_k_xep_hang.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "INTENTS",
    "classify_intent",
    "fold_text",
    "tokenize",
]

# -----------------------------------------------------------------------------
# Text folding & tokenization
# -----------------------------------------------------------------------------


def fold_text(s: str) -> str:
    """Lowercase and strip Vietnamese diacritics (đ -> d)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d")


# Shape of one unaccented Vietnamese syllable: optional onset, 1-3 vowels,
# optional final. Vietnamese syllables never end in s/f/r/x/j, so a trailing
# one of those is an unconverted Telex tone key ("thangs" -> "thang").
_SYLLABLE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?"
    r"[aeiouy]{1,3}(?:ch|ng|nh|[cmnpt])?$"
)
_TELEX_VOWELS = (("dd", "d"), ("aa", "a"), ("ee", "e"), ("oo", "o"),
                 ("aw", "a"), ("ow", "o"), ("uw", "u"))


def _detelex(tok: str) -> str:
    """Undo Telex typing slips; keep the token if the result isn't one Vietnamese syllable."""
    if len(tok) < 3 or not tok.isalpha():
        return tok
    out = tok
    for a, b in _TELEX_VOWELS:
        out = out.replace(a, b)
    if out[-1] in "sfrxj" and _SYLLABLE.match(out[:-1]):
        out = out[:-1]
    # Accept only rewrites that produce a plausible syllable, so brand names
    # like "tiger", "seller", "cheese", "for" stay intact.
    return out if out != tok and _SYLLABLE.match(out) else tok


# Teencode / abbreviations / a few English words seen in questions.
_TEENCODE: Dict[str, str] = {
    "dthu": "doanh thu",
    "bnhiu": "bao nhieu",
    "bnhieu": "bao nhieu",
    "bn": "bao nhieu",
    "nhiu": "nhieu",
    "hnay": "hom nay",
    "hqua": "hom qua",
    "sp": "san pham",
    "dc": "duoc",
    "dk": "duoc",
    "dh": "don hang",
    "trc": "truoc",
    "vs": "voi",
    "th": "thang",
    "ko": "khong",
    "kh": "khach hang",
    "ncc": "nha cung cap",
    "hsd": "han dung",
    "sl": "so luong",
    "z": "vay",
    "total": "tong",
    "revenue": "doanh thu",
    "bill": "don",
    "bills": "don",
    "order": "don",
    "orders": "don",
}


def tokenize(question: str) -> List[str]:
    """Fold, split on non-alphanumerics, undo Telex slips, expand teencode."""
    raw = re.findall(r"[a-z0-9]+", fold_text(question))
    toks: List[str] = []
    for t in raw:
        if t not in _TEENCODE:
            t = _detelex(t)
        toks.extend(_TEENCODE.get(t, t).split())
    return toks


# -----------------------------------------------------------------------------
# Matching helpers
# -----------------------------------------------------------------------------


class _Q:
    """A tokenized question with whole-word / word-sequence matching."""

    __slots__ = ("raw", "tokens", "_padded", "_set")

    def __init__(self, question: str):
        self.raw = question if isinstance(question, str) else ""
        self.tokens = tokenize(self.raw)
        self._padded = " " + " ".join(self.tokens) + " "
        self._set = set(self.tokens)

    def has(self, *phrases: str) -> bool:
        """True if any phrase (space-separated words) occurs as a word sequence."""
        for p in phrases:
            if " " in p:
                if f" {p} " in self._padded:
                    return True
            elif p in self._set:
                return True
        return False

    def has_all(self, *words: str) -> bool:
        return all(self.has(w) for w in words)

    @property
    def percent(self) -> bool:
        return "%" in self.raw or self.has("phan tram", "ti le", "ty le", "ti trong", "ty trong")


# -----------------------------------------------------------------------------
# Intent rules (ordered by precedence -- first match wins)
# -----------------------------------------------------------------------------


def _cong_no(q: _Q) -> bool:
    return q.has(
        "no", "cong no", "mua chiu", "ghi no", "tra gop", "tien gop", "ky gop",
        "gop ky", "so gop", "dang gop",
    )


def _ti_le_huy(q: _Q) -> bool:
    if not q.has("huy"):
        return False
    return q.has("don", "don hang", "ti le", "ty le", "bi huy", "da huy",
                 "trang thai huy", "huy don")


def _han_dung_lo(q: _Q) -> bool:
    if q.has("han dung", "het han", "qua han", "can date", "date"):
        return True
    # "lô" folds to "lo", which collides with "lọ" (a unit). Require a lot context.
    return q.has("lo nao", "lo gan", "lo con ton", "may lo hang")


def _giao_hang(q: _Q) -> bool:
    return q.has("dot giao", "dot hang", "giao hang", "cho giao", "chua giao",
                 "da giao", "lan giao", "ngay giao", "xe giao")


def _kho_ton(q: _Q) -> bool:
    if q.has("ton kho", "con ton", "ton cuoi", "ton dau", "luong ton", "so ton",
             "hang ton", "gia tri ton", "trong kho", "kho con", "het hang",
             "con hang", "hang e", "bi e", "e hang"):
        return True
    # "còn lại mấy lon" / "còn lại bao nhiêu" (remaining stock), but not "còn lại bán được".
    if q.has("con lai may", "con lai bao nhieu", "con bao nhieu", "con may"):
        return True
    # slow movers: "đang bán mà ... không bán ra/được"
    if q.has("khong ban duoc", "khong ban ra", "chua ban duoc", "chua ban ra"):
        return True
    return False


def _nhap_hang(q: _Q) -> bool:
    return q.has("nhap", "phieu nhap", "nha cung cap", "nhap hang", "nhap vao")


def _lai_gia_von_chi_phi(q: _Q) -> bool:
    return q.has("lai gop", "lai rong", "loi gop", "loi rong", "loi nhuan",
                 "gia von", "tien von", "chi phi", "lai bao nhieu", "loi nhat",
                 "lai nhat", "tien lai", "tien loi", "hang loi", "lai cao")


def _so_sanh_ky(q: _Q) -> bool:
    if q.has(
        "tang hay giam", "giam hay tang", "tang giam", "thay doi", "chenh lech",
        "chenh", "so voi", "so sanh", "hon hay kem", "kem hay hon", "tang truong",
        "tang lien tuc", "giam lien tuc", "lien tuc", "tang bao nhieu",
        "giam bao nhieu", "tang may", "giam may", "nhieu hon", "it hon", "cao hon",
        "thap hon", "kem hon", "tang gia", "len gia", "ky truoc", "cung ky",
    ):
        return True
    # "so 14 ngày kể từ ... với 14 ngày trước"
    return bool(re.search(r" so \d+ ngay ", q._padded))


def _top_k_xep_hang(q: _Q) -> bool:
    if q.has("top", "liet ke", "xep hang", "gio cao diem", "cao diem", "best seller",
             "best", "ban chay"):
        return True
    toks = q.tokens
    for i, t in enumerate(toks):
        if t != "nhat" or i == 0:
            continue
        prev = toks[i - 1]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        # not "chủ nhật" (Sunday), "cập nhật" (update), "N ngày gần nhất" (last N days)
        if prev in ("chu", "cap", "gan", "moi"):
            continue
        # "ít nhất / nhiều nhất N đơn" = at least / at most N (a filter, not a ranking)
        if prev in ("it", "nhieu", "toi") and nxt.isdigit():
            continue
        return True
    return False


def _trung_binh_don(q: _Q) -> bool:
    if not q.has("trung binh", "binh quan"):
        return False
    return q.has("don", "moi don", "mot don", "phieu", "moi phieu", "mot phieu",
                 "luot", "hoa don")


def _so_don(q: _Q) -> bool:
    return q.has(
        "bao nhieu don", "may don", "so don", "so luong don", "bao nhieu hoa don",
        "may hoa don", "bao nhieu khach", "may khach", "so khach", "bao nhieu luot",
        "may luot", "so luot", "bao nhieu phieu", "may phieu", "so phieu",
        "bao nhieu xe", "may xe", "so xe", "bao nhieu chiec xe", "may chiec xe",
        "khach moi", "thanh vien moi", "don co", "don hang co", "luot xe",
        "nhieu don", "so mat hang", "may mat hang", "bao nhieu mat hang",
        "so san pham", "may san pham", "bao nhieu san pham", "dem so",
    )


def _doanh_thu(q: _Q) -> bool:
    return q.has(
        "doanh thu", "tien ban", "tien hang", "ban duoc", "ban dc", "thu ve",
        "thu duoc", "ban ra", "ban", "mua", "chiem", "dong gop", "gia ban",
        "don gia", "tien mat", "chuyen khoan", "cod", "voucher", "giam gia",
        "phi ship", "diem", "khach phai tra", "thu may cu", "tien",
    )


INTENTS: Tuple[Tuple[str, Callable[[_Q], bool]], ...] = (
    ("cong_no", _cong_no),
    ("ti_le_huy", _ti_le_huy),
    ("han_dung_lo", _han_dung_lo),
    ("giao_hang", _giao_hang),
    ("kho_ton", _kho_ton),
    ("nhap_hang", _nhap_hang),
    ("lai_gia_von_chi_phi", _lai_gia_von_chi_phi),
    ("so_sanh_ky", _so_sanh_ky),
    ("top_k_xep_hang", _top_k_xep_hang),
    ("trung_binh_don", _trung_binh_don),
    ("so_don", _so_don),
    ("doanh_thu", _doanh_thu),
)

INTENT_NAMES: Tuple[str, ...] = tuple(name for name, _ in INTENTS) + ("khac",)


def classify_intent(question: str) -> str:
    """Return the intent label for a question (see module docstring)."""
    if not isinstance(question, str) or not question.strip():
        return "khac"
    q = _Q(question)
    for name, rule in INTENTS:
        if rule(q):
            return name
    return "khac"
