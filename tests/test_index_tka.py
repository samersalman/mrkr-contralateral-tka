"""Unit tests for the PURE helpers in src/index_tka.py — synthetic inputs only.

Exercises resolve_index_side (the canonical coded-vs-recovered / contra decision)
and _normalize_signal_sides across the Decision-A coded rule, the blank-earliest
recovery rule (concordant single side, conflict, no-signal), and the
unsided-no-blank exclusion. Nothing here reads a data file.
"""
from __future__ import annotations

import pytest

from src.index_tka import (
    REASON_CONFLICT,
    REASON_NO_SIGNAL,
    REASON_UNSIDED_NO_BLANK,
    _normalize_signal_sides,
    resolve_index_side,
)


# --------------------------------------------------------------------------- #
# _normalize_signal_sides                                                      #
# --------------------------------------------------------------------------- #
def test_normalize_signal_sides_forms():
    assert _normalize_signal_sides(None) == set()
    assert _normalize_signal_sides("R") == {"R"}
    assert _normalize_signal_sides("l") == {"L"}          # case-insensitive
    assert _normalize_signal_sides(" r ") == {"R"}        # padding tolerated
    assert _normalize_signal_sides("B") == set()          # non-R/L dropped
    assert _normalize_signal_sides(["R", "L"]) == {"R", "L"}
    assert _normalize_signal_sides({"R"}) == {"R"}
    assert _normalize_signal_sides([None, "R", "X"]) == {"R"}


# --------------------------------------------------------------------------- #
# Coded (Decision A) — single side among possibly-blank same-day companions    #
# --------------------------------------------------------------------------- #
def test_coded_single_rt():
    r = resolve_index_side(["RT"])
    assert r["side_source"] == "coded"
    assert r["index_side"] == "R"
    assert r["contra_side"] == "L"
    assert r["n_concordant_signals"] == 0
    assert r["exclude_reason"] is None


def test_coded_single_lt_with_blank_companions():
    # earliest date carries one LT plus blank/NULL companion lines -> coded L.
    r = resolve_index_side(["LT", None, None])
    assert r["side_source"] == "coded"
    assert r["index_side"] == "L"
    assert r["contra_side"] == "R"
    assert r["n_concordant_signals"] == 0


def test_coded_multi_single_side_token():
    # 'RT XP'-style multi-token single side is still a coded single side.
    r = resolve_index_side(["RT XP", None])
    assert r["side_source"] == "coded"
    assert r["index_side"] == "R"
    assert r["contra_side"] == "L"


def test_coded_takes_precedence_over_recovery_signals():
    # A coded earliest side is authoritative; recovery signals are ignored.
    r = resolve_index_side(
        ["LT"],
        {"same_day_image_laterality": "R", "icd_m17_laterality": "R",
         "studydesc_text": "R"},
    )
    assert r["side_source"] == "coded"
    assert r["index_side"] == "L"
    assert r["n_concordant_signals"] == 0


# --------------------------------------------------------------------------- #
# Recovery — blank earliest, concordant single side, no conflict               #
# --------------------------------------------------------------------------- #
def test_recovered_single_signal():
    r = resolve_index_side(
        [None],
        {"same_day_image_laterality": "R", "icd_m17_laterality": None,
         "studydesc_text": None},
    )
    assert r["side_source"] == "recovered"
    assert r["index_side"] == "R"
    assert r["contra_side"] == "L"
    assert r["n_concordant_signals"] == 1
    assert r["exclude_reason"] is None


def test_recovered_two_signals_agree():
    r = resolve_index_side(
        [None, None],
        {"same_day_image_laterality": "L", "icd_m17_laterality": "L",
         "studydesc_text": None},
    )
    assert r["side_source"] == "recovered"
    assert r["index_side"] == "L"
    assert r["contra_side"] == "R"
    assert r["n_concordant_signals"] == 2


def test_recovered_three_signals_agree():
    r = resolve_index_side(
        [None],
        {"same_day_image_laterality": "R", "icd_m17_laterality": "R",
         "studydesc_text": "R"},
    )
    assert r["side_source"] == "recovered"
    assert r["index_side"] == "R"
    assert r["n_concordant_signals"] == 3


# --------------------------------------------------------------------------- #
# No valid index — conflict / no signal / unsided-no-blank                     #
# --------------------------------------------------------------------------- #
def test_none_conflicting_signals():
    r = resolve_index_side(
        [None],
        {"same_day_image_laterality": "R", "icd_m17_laterality": "L",
         "studydesc_text": None},
    )
    assert r["side_source"] == "none"
    assert r["index_side"] is None
    assert r["contra_side"] is None
    assert r["n_concordant_signals"] == 0
    assert r["exclude_reason"] == REASON_CONFLICT


def test_none_conflict_within_single_signal():
    # A single image signal carrying BOTH sides same-day is itself a conflict.
    r = resolve_index_side([None], {"same_day_image_laterality": ["R", "L"]})
    assert r["side_source"] == "none"
    assert r["exclude_reason"] == REASON_CONFLICT


def test_none_no_signal():
    r = resolve_index_side([None, None], {"same_day_image_laterality": None})
    assert r["side_source"] == "none"
    assert r["index_side"] is None
    assert r["exclude_reason"] == REASON_NO_SIGNAL


def test_none_no_signal_when_signals_absent():
    # Blank earliest with no recovery_signals mapping at all -> no signal.
    r = resolve_index_side([None])
    assert r["side_source"] == "none"
    assert r["exclude_reason"] == REASON_NO_SIGNAL


def test_none_unsided_earliest_no_blank():
    # Earliest date unsided (bilateral '50') but NO blank companion -> not
    # eligible for recovery; excluded with the unsided-no-blank reason.
    r = resolve_index_side(["50"])
    assert r["side_source"] == "none"
    assert r["index_side"] is None
    assert r["exclude_reason"] == REASON_UNSIDED_NO_BLANK


def test_uninterpretable_earliest_no_blank_is_unsided():
    r = resolve_index_side(["22"])  # non-laterality token, not missing
    assert r["side_source"] == "none"
    assert r["exclude_reason"] == REASON_UNSIDED_NO_BLANK


def test_bilateral_earliest_with_blank_attempts_recovery():
    # '50' (bilateral, unsided) alongside a blank line IS blank-eligible; with a
    # concordant signal it recovers.
    r = resolve_index_side(
        ["50", None],
        {"same_day_image_laterality": "L"},
    )
    assert r["side_source"] == "recovered"
    assert r["index_side"] == "L"
    assert r["n_concordant_signals"] == 1


# --------------------------------------------------------------------------- #
# Empty / degenerate inputs                                                    #
# --------------------------------------------------------------------------- #
def test_empty_modifiers_is_unsided_no_blank():
    # No same-day lines at all -> unsided, no blank -> excluded (not recovery).
    r = resolve_index_side([])
    assert r["side_source"] == "none"
    assert r["exclude_reason"] == REASON_UNSIDED_NO_BLANK


@pytest.mark.parametrize("side,contra", [("R", "L"), ("L", "R")])
def test_contra_is_opposite_of_index(side, contra):
    r = resolve_index_side([f"{side}T"])
    assert r["index_side"] == side
    assert r["contra_side"] == contra
