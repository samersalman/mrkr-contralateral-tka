"""Unit tests for src/laterality.py — synthetic inputs only, no CSVs read.

Covers the Stage-2 required test list (13 items) plus a few documented edge
cases (bilateral-vs-single precedence, multi-token uninterpretable, NaN as
missing). Every assertion uses explicit synthetic values; nothing here opens a
data file.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.laterality import (
    add_days,
    contralateral_side,
    days_between,
    horizon_date,
    landmark_date,
    last_observation,
    normalize_cpt,
    parse_modifier,
    within,
)


# 1. 27447 normalization -----------------------------------------------------
def test_normalize_cpt_index_and_padding():
    assert normalize_cpt("27447") == "27447"
    assert normalize_cpt(" 27447 ") == "27447"           # whitespace stripped
    assert normalize_cpt("99214") == "99214"             # other valid 5-digit CPT
    assert normalize_cpt(27447) == "27447"               # integer input
    assert normalize_cpt("447") == "00447"               # left zero-pad to 5


@pytest.mark.parametrize("junk", [None, "", "   ", "abc", "274470", "27.44", "2744X7"])
def test_normalize_cpt_junk_returns_none(junk):
    assert normalize_cpt(junk) is None


def test_normalize_cpt_hcpcs_letter_suffix():
    # 4 digits + trailing letter is a valid 5-char CPT-like token (secondary path).
    assert normalize_cpt("0074t") == "0074T"


# 2. RT extraction (+ case-insensitivity) ------------------------------------
def test_parse_modifier_single_rt():
    assert parse_modifier("RT") == ("R", "single_rt")
    assert parse_modifier("rt") == ("R", "single_rt")    # case-insensitive
    assert parse_modifier("  RT  ") == ("R", "single_rt")  # padding tolerated


# 3. LT extraction -----------------------------------------------------------
def test_parse_modifier_single_lt():
    assert parse_modifier("LT") == ("L", "single_lt")
    assert parse_modifier("lt") == ("L", "single_lt")


# 4. Bilateral 50 ------------------------------------------------------------
def test_parse_modifier_bilateral_50():
    assert parse_modifier("50") == ("B", "bilateral_50")


# 5. Multi-modifier, single side ---------------------------------------------
def test_parse_modifier_multi_single_side():
    assert parse_modifier("RT XP") == ("R", "multi_single_side")
    assert parse_modifier("74 LT") == ("L", "multi_single_side")
    assert parse_modifier("LT XU") == ("L", "multi_single_side")
    assert parse_modifier("59 RT") == ("R", "multi_single_side")
    assert parse_modifier("LT   XP") == ("L", "multi_single_side")  # extra spaces


# 6. Conflicting (both RT and LT) --------------------------------------------
def test_parse_modifier_conflicting():
    assert parse_modifier("RT LT") == ("U", "conflicting")
    assert parse_modifier("LT XP RT") == ("U", "conflicting")
    assert parse_modifier("rt lt") == ("U", "conflicting")


# 7. Missing -----------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "", "   ", float("nan")])
def test_parse_modifier_missing(raw):
    assert parse_modifier(raw) == ("U", "missing")


# 8. Uninterpretable ---------------------------------------------------------
def test_parse_modifier_uninterpretable():
    assert parse_modifier("22") == ("U", "uninterpretable")
    assert parse_modifier("XP") == ("U", "uninterpretable")
    assert parse_modifier("73 22") == ("U", "uninterpretable")  # multi-token, no side


# Documented edge case: modifier 50 alongside a single side (no conflict) ->
# bilateral_50 takes precedence over multi_single_side.
def test_parse_modifier_bilateral_with_side_precedence():
    assert parse_modifier("50 RT") == ("B", "bilateral_50")
    # 50 present but RT and LT both present -> conflicting still wins.
    assert parse_modifier("50 RT LT") == ("U", "conflicting")


# 9. Contralateral assignment ------------------------------------------------
def test_contralateral_side():
    assert contralateral_side("R") == "L"
    assert contralateral_side("L") == "R"
    assert contralateral_side("B") is None
    assert contralateral_side("U") is None
    assert contralateral_side(None) is None
    assert contralateral_side("r") == "L"   # robustness to case


# 10. Date ordering / days_between sign and magnitude ------------------------
def test_days_between_sign_and_magnitude():
    d0 = date(2018, 1, 1)
    d1 = date(2018, 1, 11)
    assert days_between(d0, d1) == 10        # forward -> positive
    assert days_between(d1, d0) == -10       # backward -> negative
    assert days_between(d0, d0) == 0         # same day -> zero
    assert days_between(None, d1) is None    # None handled gracefully
    assert days_between(d0, None) is None
    # add_days / within round-trip
    assert add_days(d0, 10) == d1
    assert within(days_between(d0, d1), 1, 365) is True
    assert within(days_between(d0, d1), 11, 365) is False
    assert within(None, 1, 365) is False


# 11. Five-year horizon ------------------------------------------------------
def test_horizon_date_five_year():
    idx = date(2018, 1, 1)
    h = horizon_date(idx, 5)
    assert h == date(2023, 1, 1)             # round(5 * 365.25) = 1826 days
    assert days_between(idx, h) == 1826
    assert horizon_date(None, 5) is None


# 12. Day-90 landmark --------------------------------------------------------
def test_landmark_date_day90():
    assert landmark_date(date(2018, 1, 1)) == date(2018, 4, 1)
    assert days_between(date(2018, 1, 1), landmark_date(date(2018, 1, 1))) == 90
    assert landmark_date(None) is None


# 13. Last observation -------------------------------------------------------
def test_last_observation():
    dates = [date(2019, 5, 1), None, date(2020, 1, 15), date(2018, 12, 31)]
    assert last_observation(dates) == date(2020, 1, 15)
    assert last_observation([None, None]) is None      # all None -> None
    assert last_observation([]) is None                # empty -> None
    assert last_observation([date(2021, 6, 6)]) == date(2021, 6, 6)
