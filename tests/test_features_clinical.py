"""Unit tests for the PURE helpers in src/features_clinical.py — synthetic inputs only.

Exercises the decisions that would silently corrupt the model if wrong: the
STRICTLY-pre-index record window (leakage), the deterministic contralateral-KLG selection
rule, the TRAIN-ONLY imputation fit (non-negotiable #2), the config-driven pain value
domain (a non-string domain value would silently zero the pain features), and the
EVALUATION-ONLY status of the carried protocol section 21 / 24 columns.

No patient data is read. The pain-domain query runs against a synthetic Parquet written to
a tmp_path; the config-contract tests read config/feasibility.yaml only.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, load_config
from src.features_clinical import (
    KLG_LEFT_COL,
    KLG_RIGHT_COL,
    apply_imputer,
    check_eval_only,
    fit_imputer,
    load_pain,
    missingness_table,
    select_klg_contra,
    select_pre_index,
)

IDX = pd.Timestamp("2020-06-01")            # one synthetic index date for all patients


def _cohort(**extra) -> pd.DataFrame:
    base = dict(empi_anon=["p1"], index_date=[IDX])
    base.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(base)


def _records(offsets: list[int], **cols) -> pd.DataFrame:
    """One record per day-offset from IDX (negative = before index)."""
    df = pd.DataFrame(dict(empi_anon=["p1"] * len(offsets),
                           date_anon=[IDX + pd.Timedelta(days=d) for d in offsets]))
    for k, v in cols.items():
        df[k] = v
    return df


# --------------------------------------------------------------------------- #
# select_pre_index — the leakage boundary                                       #
# --------------------------------------------------------------------------- #
def test_record_on_index_date_is_excluded():
    kept = select_pre_index(_records([0]), _cohort())
    assert len(kept) == 0, "a record dated ON index_date must not contribute"


def test_record_one_day_before_index_is_included():
    kept = select_pre_index(_records([-1]), _cohort())
    assert len(kept) == 1
    assert kept["date_anon"].iloc[0] == IDX - pd.Timedelta(days=1)


def test_record_after_index_date_is_excluded():
    assert len(select_pre_index(_records([1, 30, 400]), _cohort())) == 0


def test_pain_lookback_boundary_is_inclusive_at_the_far_edge():
    # lookback 365: day -365 is IN (>= index - 365), day -366 is OUT.
    kept = select_pre_index(_records([-364, -365, -366]), _cohort(), lookback_days=365)
    offs = sorted((kept["date_anon"] - IDX).dt.days.tolist())
    assert offs == [-365, -364], "the 365-day lookback window is [index-365, index)"


def test_record_outside_pain_lookback_is_excluded():
    kept = select_pre_index(_records([-2, -500]), _cohort(), lookback_days=365)
    assert (kept["date_anon"] - IDX).dt.days.tolist() == [-2]


def test_no_lookback_keeps_all_history():
    kept = select_pre_index(_records([-2, -5000]), _cohort(), lookback_days=None)
    assert len(kept) == 2


def test_null_dates_are_dropped():
    rec = pd.concat([_records([-10]),
                     pd.DataFrame(dict(empi_anon=["p1"], date_anon=[pd.NaT]))],
                    ignore_index=True)
    assert len(select_pre_index(rec, _cohort())) == 1


def test_inner_join_drops_records_for_non_cohort_patients():
    rec = pd.DataFrame(dict(empi_anon=["p1", "pX"],
                            date_anon=[IDX - pd.Timedelta(days=1)] * 2))
    kept = select_pre_index(rec, _cohort())
    assert kept["empi_anon"].tolist() == ["p1"]


def test_cohort_columns_are_carried_through_for_side_specific_logic():
    kept = select_pre_index(_records([-1]), _cohort(contra_side="L"))
    assert kept["contra_side"].iloc[0] == "L"


# --------------------------------------------------------------------------- #
# select_klg_contra — side selection, nearest frontal, deterministic tie rule    #
# --------------------------------------------------------------------------- #
def _frontal(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["empi_anon", "contra_side", "days_to_index",
                                       KLG_LEFT_COL, KLG_RIGHT_COL])


def test_contra_left_reads_the_left_klg_column():
    f = _frontal([dict(empi_anon="p1", contra_side="L", days_to_index=10,
                       L_KLG_inference=3.0, R_KLG_inference=0.0)])
    out = select_klg_contra(f, pd.DataFrame(dict(empi_anon=["p1"])))
    assert out["klg_contra"].iloc[0] == 3.0


def test_contra_right_reads_the_right_klg_column():
    f = _frontal([dict(empi_anon="p1", contra_side="R", days_to_index=10,
                       L_KLG_inference=3.0, R_KLG_inference=1.0)])
    out = select_klg_contra(f, pd.DataFrame(dict(empi_anon=["p1"])))
    assert out["klg_contra"].iloc[0] == 1.0


def test_multi_frontal_picks_the_one_closest_to_index():
    f = _frontal([
        dict(empi_anon="p1", contra_side="L", days_to_index=200, L_KLG_inference=0.0,
             R_KLG_inference=0.0),
        dict(empi_anon="p1", contra_side="L", days_to_index=12, L_KLG_inference=4.0,
             R_KLG_inference=0.0),
        dict(empi_anon="p1", contra_side="L", days_to_index=60, L_KLG_inference=2.0,
             R_KLG_inference=0.0)])
    out = select_klg_contra(f, pd.DataFrame(dict(empi_anon=["p1"])))
    assert out["klg_contra"].iloc[0] == 4.0, "smallest days_to_index must win"
    assert out["klg_n_frontal"].iloc[0] == 1


def test_tie_on_days_to_index_is_averaged_and_order_independent():
    rows = [dict(empi_anon="p1", contra_side="L", days_to_index=5, L_KLG_inference=2.0,
                 R_KLG_inference=9.0),
            dict(empi_anon="p1", contra_side="L", days_to_index=5, L_KLG_inference=4.0,
                 R_KLG_inference=9.0)]
    out = select_klg_contra(_frontal(rows), pd.DataFrame(dict(empi_anon=["p1"])))
    rev = select_klg_contra(_frontal(rows[::-1]), pd.DataFrame(dict(empi_anon=["p1"])))
    assert out["klg_contra"].iloc[0] == 3.0 == rev["klg_contra"].iloc[0]
    assert out["klg_n_frontal"].iloc[0] == 2


def test_tie_average_skips_nan_but_keeps_the_row_count():
    rows = [dict(empi_anon="p1", contra_side="R", days_to_index=5, L_KLG_inference=0.0,
                 R_KLG_inference=np.nan),
            dict(empi_anon="p1", contra_side="R", days_to_index=5, L_KLG_inference=0.0,
                 R_KLG_inference=3.0)]
    out = select_klg_contra(_frontal(rows), pd.DataFrame(dict(empi_anon=["p1"])))
    assert out["klg_contra"].iloc[0] == 3.0
    assert out["klg_n_frontal"].iloc[0] == 2


def test_all_nan_frontals_give_nan_klg():
    rows = [dict(empi_anon="p1", contra_side="R", days_to_index=5, L_KLG_inference=2.0,
                 R_KLG_inference=np.nan)]
    out = select_klg_contra(_frontal(rows), pd.DataFrame(dict(empi_anon=["p1"])))
    assert pd.isna(out["klg_contra"].iloc[0])


def test_patient_without_any_frontal_gets_nan_and_zero_count():
    f = _frontal([dict(empi_anon="p1", contra_side="L", days_to_index=5,
                       L_KLG_inference=2.0, R_KLG_inference=2.0)])
    out = select_klg_contra(f, pd.DataFrame(dict(empi_anon=["p1", "p2"])))
    row = out.set_index("empi_anon").loc["p2"]
    assert pd.isna(row["klg_contra"]) and row["klg_n_frontal"] == 0


def test_output_is_one_row_per_cohort_patient():
    f = _frontal([dict(empi_anon="p1", contra_side="L", days_to_index=5,
                       L_KLG_inference=2.0, R_KLG_inference=2.0),
                  dict(empi_anon="p1", contra_side="L", days_to_index=5,
                       L_KLG_inference=2.0, R_KLG_inference=2.0)])
    out = select_klg_contra(f, pd.DataFrame(dict(empi_anon=["p1", "p2", "p3"])))
    assert len(out) == 3 and out["empi_anon"].is_unique


# --------------------------------------------------------------------------- #
# fit_imputer / apply_imputer — TRAIN ONLY (non-negotiable #2)                   #
# --------------------------------------------------------------------------- #
def _split_frame() -> pd.DataFrame:
    # train median of x is 2.0; val/test carry extreme values that must NOT move it.
    return pd.DataFrame({
        "empi_anon": [f"p{i}" for i in range(9)],
        "split": ["train"] * 5 + ["val"] * 2 + ["test"] * 2,
        "x": [1.0, 2.0, 2.0, 3.0, np.nan, 100.0, 200.0, 300.0, np.nan],
        "g": ["a", "a", "a", "b", "b", "z", "z", "z", None],
    })


def test_fit_uses_train_rows_only_for_the_median():
    df = _split_frame()
    params = fit_imputer(df, ["x"], missing_indicator_cols=["x"])
    assert params["columns"]["x"]["fill_value"] == 2.0, "val/test values leaked into the fit"
    assert df["x"].median() != 2.0, "the test would be vacuous if the overall median matched"
    assert params["n_fit_rows"] == 5
    assert params["fit_split"] == "train"


def test_extreme_value_only_in_val_test_does_not_move_the_fit():
    df = _split_frame()
    base = fit_imputer(df, ["x"])["columns"]["x"]["fill_value"]
    df.loc[df["split"] != "train", "x"] = 999.0
    assert fit_imputer(df, ["x"])["columns"]["x"]["fill_value"] == base


def test_fit_counts_missing_in_fit_split_and_overall():
    spec = fit_imputer(_split_frame(), ["x"])["columns"]["x"]
    assert spec["n_missing_fit_split"] == 1 and spec["n_missing_all"] == 2


def test_categorical_column_uses_most_frequent():
    spec = fit_imputer(_split_frame(), ["g"])["columns"]["g"]
    assert spec["strategy"] == "most_frequent"
    assert spec["fill_value"] == "a", "'z' occurs only in val/test and must not be chosen"


def test_apply_fills_val_and_test_with_the_train_statistic():
    df = _split_frame()
    params = fit_imputer(df, ["x"], missing_indicator_cols=["x"])
    out = apply_imputer(df, params)
    assert out["x_imp"].notna().all()
    assert out.loc[out["split"] == "test", "x_imp"].tolist() == [300.0, 2.0]
    assert out["x"].isna().sum() == 2, "the raw column must be preserved for auditing"


def test_apply_emits_the_missing_indicator_only_where_declared():
    df = _split_frame()
    out = apply_imputer(df, fit_imputer(df, ["x", "g"], missing_indicator_cols=["x"]))
    assert out["x_missing"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert "g_missing" not in out.columns


def test_apply_is_idempotent_and_does_not_refit():
    df = _split_frame()
    params = fit_imputer(df, ["x"])
    once = apply_imputer(df, params)
    twice = apply_imputer(once, params)
    assert twice["x_imp"].equals(once["x_imp"])


def test_fit_rejects_a_column_entirely_missing_in_train():
    df = _split_frame()
    df.loc[df["split"] == "train", "x"] = np.nan
    with pytest.raises(AssertionError):
        fit_imputer(df, ["x"])


def test_fit_rejects_an_absent_split_column():
    with pytest.raises(AssertionError):
        fit_imputer(_split_frame().drop(columns=["split"]), ["x"])


def test_fit_rejects_an_empty_fit_split():
    with pytest.raises(AssertionError):
        fit_imputer(_split_frame(), ["x"], fit_split="nosuchsplit")


# --------------------------------------------------------------------------- #
# missingness_table — aggregate only, never an identifier                       #
# --------------------------------------------------------------------------- #
def test_missingness_table_reports_overall_and_each_split():
    df = _split_frame()
    tab = missingness_table(df, ["x"], {"x": "predictor"}, ["x"])
    assert sorted(tab["scope"]) == ["overall", "test", "train", "val"]
    assert int(tab.loc[tab.scope == "overall", "n_missing"].iloc[0]) == 2
    assert int(tab.loc[tab.scope == "train", "n_missing"].iloc[0]) == 1
    assert "empi_anon" not in tab["column"].tolist()


# --------------------------------------------------------------------------- #
# check_eval_only — the carried section 21 / 24 columns are NEVER predictors    #
# --------------------------------------------------------------------------- #
def test_carried_column_is_returned_when_it_is_purely_evaluation_only():
    assert check_eval_only(["weight_bearing_frontal"], ["age_at_index"],
                           ["age_at_index_imp"]) == ["weight_bearing_frontal"]


def test_carried_column_that_is_also_a_primary_predictor_is_rejected():
    with pytest.raises(AssertionError):
        check_eval_only(["age_at_index"], ["age_at_index"], ["age_at_index_imp"])


def test_carried_column_present_in_model_columns_is_rejected():
    with pytest.raises(AssertionError):
        check_eval_only(["weight_bearing_frontal"], ["age_at_index"],
                        ["age_at_index_imp", "weight_bearing_frontal"])


def test_carried_column_whose_imputed_twin_is_a_model_column_is_rejected():
    with pytest.raises(AssertionError):
        check_eval_only(["weight_bearing_frontal"], ["age_at_index"],
                        ["age_at_index_imp", "weight_bearing_frontal_imp"])


def test_carried_column_may_not_use_the_generated_model_column_suffixes():
    for bad in ("something_imp", "something_missing"):
        with pytest.raises(AssertionError):
            check_eval_only([bad], ["age_at_index"], ["age_at_index_imp"])


def test_carry_columns_are_deduplicated_and_order_preserved():
    assert check_eval_only(["b", "a", "b"], ["age_at_index"]) == ["b", "a"]


def test_no_carry_columns_is_allowed():
    assert check_eval_only([], ["age_at_index"], ["age_at_index_imp"]) == []


# --------------------------------------------------------------------------- #
# load_pain — the pain value domain comes from CONFIG, and it must be STRINGS   #
# --------------------------------------------------------------------------- #
PAIN_INDEX = pd.Timestamp("2020-06-01")


def _pain_parquet(tmp_path) -> str:
    """knee_pain / pain_score are VARCHAR in the real source; mirror that exactly.

    Deliberately uses a NON-default domain ('Y', column 'side', bilateral 'BOTH') so a
    hard-coded value inside the module cannot make these tests pass.
    """
    rows = [
        # p1 (contra L): a flagged bilateral row with score 7, plus an UNflagged score 9
        ("p1", -10, "Y", "7", "BOTH"),
        ("p1", -20, "N", "9", "R"),
        # p2 (contra R): flagged, side matches contra exactly
        ("p2", -5, "Y", "5", "R"),
        # p3 (contra R): flagged, side is the OTHER knee
        ("p3", -5, "Y", "4", "L"),
        # out-of-window and post-index rows must never contribute
        ("p1", -400, "Y", "10", "BOTH"),
        ("p1", 3, "Y", "10", "BOTH"),
    ]
    df = pd.DataFrame(rows, columns=["empi_anon", "off", "knee_pain", "pain_score", "side"])
    df["date_anon"] = PAIN_INDEX + pd.to_timedelta(df.pop("off"), unit="D")
    p = tmp_path / "pain.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def _run_load_pain(tmp_path, **kw) -> pd.DataFrame:
    cohort = pd.DataFrame(dict(empi_anon=["p1", "p2", "p3"],
                               index_date=[PAIN_INDEX] * 3,
                               contra_side=["L", "R", "R"]))
    con = duckdb.connect()
    try:
        con.register("cohort", cohort)
        args = dict(lookback_days=365, knee_col="knee_pain", score_col="pain_score",
                    flag_true="Y", laterality_col="side", bilateral_code="BOTH")
        args.update(kw)
        return load_pain(con, _pain_parquet(tmp_path), **args).set_index("empi_anon")
    finally:
        con.close()


def test_pain_flag_domain_is_taken_from_the_caller_not_hard_coded(tmp_path):
    out = _run_load_pain(tmp_path)
    assert out["knee_pain_any"].to_dict() == {"p1": 1, "p2": 1, "p3": 1}


def test_the_old_hard_coded_flag_value_now_matches_nothing(tmp_path):
    """Proves the domain is genuinely plumbed through: '1' is not this frame's TRUE value."""
    out = _run_load_pain(tmp_path, flag_true="1")
    assert set(out["knee_pain_any"]) == {0}
    assert out["pain_score_max"].isna().all()


def test_pain_score_max_only_counts_rows_where_the_flag_is_true(tmp_path):
    out = _run_load_pain(tmp_path)
    assert out.loc["p1", "pain_score_max"] == 7, "the unflagged score 9 must be ignored"


def test_bilateral_code_from_the_caller_decides_the_contralateral_flag(tmp_path):
    matching = _run_load_pain(tmp_path)
    assert matching.loc["p1", "knee_pain_contra"] == 1, "a bilateral row counts as contra"
    assert matching.loc["p2", "knee_pain_contra"] == 1, "side == contra_side counts"
    assert matching.loc["p3", "knee_pain_contra"] == 0, "the other knee must not count"
    other = _run_load_pain(tmp_path, bilateral_code="X")
    assert other.loc["p1", "knee_pain_contra"] == 0, "bilateral_code was hard-coded"


def test_laterality_column_name_from_the_caller_is_used(tmp_path):
    with pytest.raises(duckdb.BinderException):
        _run_load_pain(tmp_path, laterality_col="not_a_column")


def test_only_rows_inside_the_lookback_and_strictly_pre_index_contribute(tmp_path):
    out = _run_load_pain(tmp_path)
    assert out.loc["p1", "n_pain_preindex_rows"] == 2, \
        "the -400d and the post-index rows must be excluded"


def test_a_non_string_flag_value_is_rejected_loudly(tmp_path):
    """The int/str trap: knee_pain is VARCHAR, so an int would zero the feature silently."""
    with pytest.raises(AssertionError, match="must be a STRING"):
        _run_load_pain(tmp_path, flag_true=1)
    with pytest.raises(AssertionError, match="must be a STRING"):
        _run_load_pain(tmp_path, bilateral_code=1)


# --------------------------------------------------------------------------- #
# Config contract — the shipped YAML must satisfy what the module assumes        #
# --------------------------------------------------------------------------- #
def _fcfg() -> dict:
    return load_config(DEFAULT_CONFIG)["features_clinical"]


@pytest.mark.parametrize("key", ["pain_knee_flag_col", "pain_score_col", "pain_knee_flag_true",
                                 "pain_laterality_col", "pain_bilateral_code"])
def test_pain_domain_config_values_are_strings(key):
    value = _fcfg()[key]
    assert isinstance(value, str), \
        f"features_clinical.{key} must be quoted in the YAML (the source column is VARCHAR)"


def test_pain_flag_true_is_the_verified_domain_value():
    assert _fcfg()["pain_knee_flag_true"] == "1", "knee_pain is the STRING '0'/'1'"


def test_configured_carry_columns_are_not_predictors():
    fcfg = _fcfg()
    assert check_eval_only(list(fcfg["carry_columns"]),
                           list(fcfg["primary_predictors"])) == list(fcfg["carry_columns"])


def test_the_weight_bearing_column_needed_by_protocol_21_and_24_is_carried():
    assert "weight_bearing_frontal" in _fcfg()["carry_columns"], \
        "the weight-bearing subgroup (section 21) and WB-only sensitivity (section 24) need it"


# --------------------------------------------------------------------------- #
# Protocol Table 7 model ladder: M0 = age, sex, comorbidities, pain, image      #
# interval; M1 = M0 plus inferred KLG. Table 6: KLG is a SECONDARY comparator.  #
# --------------------------------------------------------------------------- #
def test_m0_has_no_klg_column_because_table_6_calls_it_a_secondary_comparator():
    """The B1 regression guard: KLG inside M0 biases the primary estimand toward zero."""
    primary = list(_fcfg()["primary_predictors"])
    assert not [c for c in primary if c.startswith("klg")], (
        "protocol Table 6 lists dataset-inferred KLG as a SECONDARY COMPARATOR ONLY and "
        "Table 7 places it in M1; an M0 containing a radiograph-derived severity grade is "
        "not 'routine clinical variables' and makes the M4-vs-M0 comparison of protocol "
        f"Table 8 measure imaging against imaging. Got: {primary}")


def test_m0_contains_every_predictor_domain_protocol_table_7_names():
    """Table 7: M0 = 'Age, sex, comorbidities, pain, image-to-index interval'."""
    primary = set(_fcfg()["primary_predictors"])
    assert {"age_at_index", "sex_female"} <= primary
    assert {"knee_pain_any", "pain_score_max"} & primary, "the pain domain must be present"
    assert set(_fcfg()["comorbidity_flags"]) <= primary, "every comorbidity flag is in M0"
    assert "days_to_index" in primary, \
        "the image-to-index interval is a Table 7 M0 predictor and was silently absent"
    assert len(primary) == 11


def test_m1_is_exactly_m0_plus_inferred_klg():
    fcfg = _fcfg()
    primary, m1 = list(fcfg["primary_predictors"]), list(fcfg["m1_predictors"])
    assert set(primary) < set(m1), "M1 must be a strict superset of M0"
    assert [c for c in m1 if c not in primary] == ["klg_contra"], \
        "protocol Table 7 defines M1 as 'M0 plus inferred KLG' — nothing else"
    assert fcfg["m1_requires_klg_eligible_subset"] is True, \
        "protocol Secondary objective 2 restricts M1 to the eligible bilateral-frontal subset"


def test_klg_keeps_its_missing_indicator_so_m1_and_the_audits_still_work():
    mic = list(_fcfg()["imputation"]["missing_indicator_cols"])
    assert "klg_contra" in mic, \
        "klg_contra_missing defines M1 eligibility and the complete-case sensitivity filter"


def test_predictor_sets_name_the_protocol_table_7_ladder():
    mc = load_config(DEFAULT_CONFIG)["model_clinical"]
    assert mc["predictor_sets"] == {"m0": "primary_predictors", "m1_klg": "m1_predictors"}
    for key in ("m1_report_md", "m1_metrics_csv", "m1_model_json"):
        assert isinstance(mc[key], str) and mc[key], f"model_clinical.{key} must be set"
