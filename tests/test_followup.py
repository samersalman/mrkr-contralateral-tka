"""Unit tests for the PURE helpers in src/followup.py — synthetic inputs only.

Exercises resolve_followup (landmark-anchored event/censor resolution) and
reverse_km (censoring-KM median follow-up), plus the followup_scaffold consistency
these two guarantee. Nothing here reads a data file.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from src.followup import (
    DEFAULT_HORIZON_DAYS,
    REASON_ADMIN_HORIZON,
    REASON_EVENT,
    REASON_LAST_OBSERVED,
    followup_scaffold,
    resolve_followup,
    reverse_km,
)

LM = date(2020, 1, 1)                       # a landmark (time origin) for the tests
HORIZON = DEFAULT_HORIZON_DAYS              # 1826 days = 5 years


def _d(days: int) -> date:
    """Landmark + days (negative allowed)."""
    return LM + timedelta(days=days)


# --------------------------------------------------------------------------- #
# resolve_followup — EVENT branch (event within [landmark, landmark+horizon])  #
# --------------------------------------------------------------------------- #
def test_event_inside_horizon():
    # event 30 d after landmark, records extend well beyond -> counted event.
    ind, t, reason = resolve_followup(LM, _d(2000), _d(30))
    assert (ind, t, reason) == (1, 30, REASON_EVENT)


def test_event_at_landmark_is_zero_time_event():
    ind, t, reason = resolve_followup(LM, _d(500), LM)
    assert (ind, t, reason) == (1, 0, REASON_EVENT)


def test_event_at_horizon_boundary_is_event():
    # event exactly on the horizon (day 1826) is still within horizon.
    ind, t, reason = resolve_followup(LM, _d(2000), _d(HORIZON))
    assert (ind, t, reason) == (1, HORIZON, REASON_EVENT)


def test_event_outside_horizon_is_admin_censored():
    # event 2000 d out (> 1826) does NOT count; censored at the horizon.
    ind, t, reason = resolve_followup(LM, _d(2000), _d(2000))
    assert (ind, t, reason) == (0, HORIZON, REASON_ADMIN_HORIZON)


def test_event_before_landmark_is_not_counted():
    # a pre-landmark "event" must not yield negative time; falls to censoring.
    ind, t, reason = resolve_followup(LM, _d(500), _d(-5))
    assert (ind, t, reason) == (0, 500, REASON_LAST_OBSERVED)


# --------------------------------------------------------------------------- #
# resolve_followup — CENSOR branch (no event) with last_obs before/after       #
# --------------------------------------------------------------------------- #
def test_censor_last_obs_after_horizon_admin():
    # last record beyond the horizon -> administratively censored at the horizon.
    ind, t, reason = resolve_followup(LM, _d(2000), None)
    assert (ind, t, reason) == (0, HORIZON, REASON_ADMIN_HORIZON)


def test_censor_last_obs_before_horizon_last_observed():
    # records run out at day 500 (< horizon) -> censored at last observation.
    ind, t, reason = resolve_followup(LM, _d(500), None)
    assert (ind, t, reason) == (0, 500, REASON_LAST_OBSERVED)


def test_censor_last_obs_before_landmark_is_zero():
    # last record before the landmark -> zero follow-up (clamped, non-negative).
    ind, t, reason = resolve_followup(LM, _d(-10), None)
    assert (ind, t, reason) == (0, 0, REASON_LAST_OBSERVED)


def test_censor_last_obs_at_horizon_binds_admin():
    # last_obs exactly on the horizon -> admin_horizon (horizon <= last_observed).
    ind, t, reason = resolve_followup(LM, _d(HORIZON), None)
    assert (ind, t, reason) == (0, HORIZON, REASON_ADMIN_HORIZON)


def test_none_last_observed_is_zero_followup():
    ind, t, reason = resolve_followup(LM, None, None)
    assert (ind, t, reason) == (0, 0, REASON_LAST_OBSERVED)


# --------------------------------------------------------------------------- #
# resolve_followup with NO event reproduces the scaffold (in-module invariant)  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("last_off", [-10, 0, 90, 500, HORIZON, 2000])
def test_resolve_no_event_matches_scaffold(last_off):
    index_date = _d(-90)                     # so that landmark == LM (index + 90)
    last_obs = _d(last_off)
    sc = followup_scaffold(index_date, last_obs)
    assert sc["landmark_date"] == LM
    ind, t, _reason = resolve_followup(LM, last_obs, None)
    assert ind == 0
    assert t == sc["followup_days_from_landmark_if_no_event"]
    assert t >= 0


# --------------------------------------------------------------------------- #
# followup_scaffold — field-level definitions                                  #
# --------------------------------------------------------------------------- #
def test_scaffold_observed_through_90_strict_gt():
    idx = date(2020, 1, 1)
    lm = followup_scaffold(idx, idx)["landmark_date"]        # index + 90
    # last_observed exactly ON the landmark is NOT observed_through_90 (strict >).
    assert followup_scaffold(idx, lm)["observed_through_90"] is False
    assert followup_scaffold(idx, lm + timedelta(days=1))["observed_through_90"] is True


def test_scaffold_complete_5y_uses_ge():
    idx = date(2020, 1, 1)
    lm = followup_scaffold(idx, idx)["landmark_date"]
    horizon = lm + timedelta(days=HORIZON)
    # complete_5y uses >= (last_observed on the horizon counts as complete).
    assert followup_scaffold(idx, horizon)["complete_5y"] is True
    assert followup_scaffold(idx, horizon - timedelta(days=1))["complete_5y"] is False


def test_scaffold_censor_date_is_min_horizon_lastobs():
    idx = date(2020, 1, 1)
    sc_short = followup_scaffold(idx, followup_scaffold(idx, idx)["landmark_date"] + timedelta(days=200))
    assert sc_short["followup_days_from_landmark_if_no_event"] == 200
    assert sc_short["complete_5y"] is False


# --------------------------------------------------------------------------- #
# reverse_km — flipped-indicator censoring KM                                  #
# --------------------------------------------------------------------------- #
def test_reverse_km_all_censored_median():
    # All observations censored -> flipped to events -> a standard KM on
    # [10,20,30,40,50] whose survival crosses 0.5 at t=30.
    median, kmf = reverse_km([10, 20, 30, 40, 50], [0, 0, 0, 0, 0])
    assert median == 30.0
    # flipping turned all 5 censored obs into "events".
    assert int(kmf.event_observed.sum()) == 5


def test_reverse_km_all_events_is_inf():
    # All observations are events -> flipped to all-censored -> curve never
    # reaches 0.5 -> median follow-up is +inf (lifelines convention).
    median, kmf = reverse_km([10, 20, 30, 40, 50], [1, 1, 1, 1, 1])
    assert math.isinf(median)
    assert int(kmf.event_observed.sum()) == 0


def test_reverse_km_flips_indicators():
    # Mixed: 2 events + 3 censored -> flipped fit has exactly 3 "events".
    _median, kmf = reverse_km([5, 10, 15, 20, 25], [1, 0, 1, 0, 0])
    assert int(kmf.event_observed.sum()) == 3
    assert np.isfinite(kmf.median_survival_time_) or math.isinf(kmf.median_survival_time_)
