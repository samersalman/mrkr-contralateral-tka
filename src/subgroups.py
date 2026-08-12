"""subgroups.py — LOCKED subgroup event tallies over the PRIMARY final cohort.

MRKR Contralateral TKA — LOCKED EXTRACTION. For the primary final landmark
cohort (recovery_any / 730-day = 3,709 patients / 533 primary events, VERIFIED)
this module reports, for every protocol subgroup, the patient count, the primary
(recovery_any / 5-year) event count, the event percentage, and a stability flag
('>=100' | '50-99' | '<50', protocol section 21 sparse/moderate bins) used to
decide which strata are reportable vs suppressed.

Subgroup families are DECLARED ONCE, in ``config/feasibility.yaml`` under
``subgroups.families``, and this module owns the parser for that declaration
(:func:`load_families`, :func:`family_mask`). ``src/eval_models.py`` imports both, so the
cohort description, the equity audit and the v6 imaging robustness table cannot drift
apart. Before 2026-08-11 the list was hard-coded here at ``build_rows`` and again at
``eval_models.subgroup_levels``; that duplication is what this refactor removes.

Families this module tallies (``scopes`` containing ``cohort``):
  * sex                     : Female, Male                         (partition)
  * age at index            : <65, >=65   (cutoff = subgroups.age_cutoff)  (partition)
  * race                    : Black / White / Asian (3 selected groups only;
                              NOT a partition — other race values are excluded)
  * obesity                 : yes, no   (lifetime MAX(obesity) per patient)  (partition)
  * imaging weight-bearing  : WB, non-WB  (final_cohort.weight_bearing_frontal)  (partition)
  * imaging views           : frontal-only (view_set=='frontal') vs multi-view
                              (any non-frontal-only study)                  (partition)
  * laterality source       : coded / recovered (final_cohort.side_source)  (partition)
  * acquisition era         : THREE declared schemes over the calendar year of
                              final_cohort.study_date — five-year calendar spans,
                              development-tertile edges, development-median edge.
                              **Every era row carries the family's ``note``, which
                              records that StudyDate_anon has a per-patient random
                              shift and that protocol section 17's written
                              confirmation of cross-patient date comparability has
                              never been obtained (deviation D17).**  (partitions)

The image-quality families (``source: imaging``) are declared in the same config block but
are NOT tallied here: they need ``derived-data/cohort/preprocess_labels.csv``, which is a
per-image CSV, and this module is a locked extraction over the typed Parquet inputs.
``src/eval_models.py`` performs that join.

Guardrails honoured (identical to the sibling extraction modules):
  * Read-only on the typed Parquet inputs; no DICOMs, no models, no metrics.
  * ICD comorbidity flags are PER-DIAGNOSIS-LINE: obesity presence for a patient
    is MAX(obesity) GROUP BY empi_anon over the ICD table (never row-level).
    The obesity window is LIFETIME (all diagnosis lines, no date restriction).
    The large ICD table (~1.8 GB) is streamed via DuckDB, never full pandas.
  * Deterministic. Every parameter comes from config/feasibility.yaml.
  * NO patient identifiers (empi_anon) are ever written. outputs/*.csv is
    AGGREGATE COUNTS only; patient-level data stays in memory.
  * Logging APPENDS to outputs/logs/run.log with the 'subgroups' prefix.

Regression anchors (assert; these are the re-gate recovery_any / 730 event counts
and PARTITION the 533 primary events):
  Female events=342, Male=191 (sum 533); age<65=228, age>=65=305 (sum 533);
  Black=156, White=328, Asian=13; Female+Male patients == 3,709; age<65+age>=65
  patients == 3,709.
  Added 2026-08-11 with the new families, NOT in place of anything above:
  laterality source coded 1,881 patients / 394 events, recovered 1,828 / 139; and
  every level of all three acquisition-era schemes (see ANCHORS below). Obesity /
  WB / view-set splits stay anchor-free and are asserted only to partition.
  EVERY family declared ``partition: true`` is asserted to sum to 3,709 patients and
  533 events, and that check is now driven by the config flag rather than by a
  hard-coded list of three families.

Run from the project root::

    python3 -m src.subgroups --config config/feasibility.yaml

Writes ONLY: outputs/subgroup_counts.csv, and appends outputs/logs/run.log.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import load_config

MODULE = "subgroups"


# =========================================================================== #
# THE SUBGROUP FAMILY DECLARATION - parsed here, consumed here AND in         #
# src/eval_models.py. config/feasibility.yaml -> subgroups.families is the     #
# only place a family, a level, a label or a rule is written down.            #
# =========================================================================== #
#: Every operator the rule vocabulary admits. Closed on purpose: a rule that is
#: not one of these is a schema error, not a silently ignored family.
RULE_OPS = ("eq", "ne", "lt", "le", "gt", "ge", "is_true", "is_false", "between")

#: Which consumer may read which family. ``equity`` is FROZEN at the six original
#: families because it writes the published ``{val,test}_subgroups.csv``.
FAMILY_SCOPES = ("cohort", "equity", "robustness")

#: ``source: imaging`` families need the per-image preprocess_labels join, which only
#: eval_models performs, so they may not appear in a scope this module serves.
FAMILY_SOURCES = ("clinical", "imaging")

#: The six families that were hard-coded before 2026-08-11. Asserted, so a config edit
#: cannot quietly add a seventh row to a published equity table.
FROZEN_EQUITY_FAMILIES = ("sex", "age_at_index", "race", "obesity",
                          "imaging_weight_bearing", "imaging_views")


@dataclass(frozen=True)
class Level:
    """One stratum: two renderings of its name plus the rule that selects it."""

    key: str
    cohort_label: str
    report_label: str
    rule: dict


@dataclass(frozen=True)
class Family:
    """One subgroup family, exactly as declared in config."""

    key: str
    cohort_label: str
    report_label: str
    scopes: tuple[str, ...]
    source: str
    partition: bool
    note: str
    levels: tuple[Level, ...]


def _label(text: str, cfg) -> str:
    """Substitute the one templated token labels may carry: ``{cutoff}``."""
    return str(text).format(cutoff=int(cfg["subgroups"]["age_cutoff"]))


def _resolve_value(rule: dict, cfg):
    """A rule's comparison value, either literal or looked up under ``subgroups:``."""
    ref = rule.get("value_from")
    if ref is None:
        assert "value" in rule or rule["op"] in ("is_true", "is_false"), (
            f"rule {rule!r} has neither 'value' nor 'value_from' and its op needs one")
        return rule.get("value")
    node = cfg["subgroups"]
    for part in str(ref).split("."):
        assert part in node, f"rule value_from={ref!r} does not resolve under subgroups:"
        node = node[part]
    return node


def load_families(cfg, scope: str) -> list[Family]:
    """Every declared family visible to ``scope``, in declaration order.

    Declaration order IS published row order, in both consumers, so this function never
    sorts. The schema is validated here rather than at every call site: an unknown op, an
    unknown scope, a reviewer-facing label carrying an underscore, a duplicate key or an
    ``imaging`` family offered to a non-imaging consumer are all schema errors that stop
    the run instead of producing a table with a missing stratum in it.
    """
    assert scope in FAMILY_SCOPES, f"unknown family scope {scope!r}; known {FAMILY_SCOPES}"
    declared = cfg["subgroups"]["families"]
    assert declared, "config subgroups.families is empty; both consumers read only it"
    out: list[Family] = []
    seen_keys: set[str] = set()
    for raw in declared:
        key = str(raw["key"])
        assert key not in seen_keys, f"duplicate family key {key!r} in subgroups.families"
        seen_keys.add(key)
        scopes = tuple(str(s) for s in raw["scopes"])
        bad = sorted(set(scopes) - set(FAMILY_SCOPES))
        assert not bad, f"family {key!r} declares unknown scope(s) {bad}"
        source = str(raw.get("source", "clinical"))
        assert source in FAMILY_SOURCES, f"family {key!r} has unknown source {source!r}"
        assert not (source == "imaging" and ({"cohort", "equity"} & set(scopes))), (
            f"family {key!r} is source:imaging, so it needs the per-image "
            f"preprocess_labels join and may not be offered to scope cohort or equity")
        report_label = _label(raw["report_label"], cfg)
        assert "_" not in report_label, (
            f"family {key!r} report_label {report_label!r} carries an underscore; "
            f"reviewer-facing names in the published subgroup tables never do")
        levels: list[Level] = []
        for lv in raw["levels"]:
            rule = dict(lv["rule"])
            assert rule.get("op") in RULE_OPS, (
                f"family {key!r} level {lv['key']!r} has op {rule.get('op')!r}, which is "
                f"not one of {RULE_OPS}")
            lv_report = _label(lv["report_label"], cfg)
            assert "_" not in lv_report, (
                f"family {key!r} level report_label {lv_report!r} carries an underscore")
            levels.append(Level(key=str(lv["key"]),
                                cohort_label=_label(lv.get("cohort_label", lv["key"]), cfg),
                                report_label=lv_report, rule=rule))
        assert len(levels) >= 2, f"family {key!r} declares fewer than two levels"
        if scope in scopes:
            out.append(Family(key=key,
                              cohort_label=_label(raw.get("cohort_label", key), cfg),
                              report_label=report_label, scopes=scopes, source=source,
                              partition=bool(raw.get("partition", False)),
                              note=str(raw.get("note", "")).strip(),
                              levels=tuple(levels)))
    if scope == "equity":
        got = tuple(f.key for f in out)
        assert got == FROZEN_EQUITY_FAMILIES, (
            f"scope 'equity' writes the published {{val,test}}_subgroups.csv, whose row set "
            f"is frozen at {FROZEN_EQUITY_FAMILIES}. config now yields {got}. Adding a family "
            f"here rewrites a published table; declare it scope 'robustness' instead.")
    assert out, f"no declared family carries scope {scope!r}"
    return out


def family_mask(rule: dict, frame: pd.DataFrame, cfg) -> np.ndarray:
    """Evaluate one level's rule against ``frame`` and return a plain boolean array.

    NaN never satisfies any rule. That is deliberate and it is what the ``partition``
    assertions are there to catch: a family whose levels do not sum to N is a family with
    missing values in its column, and this module refuses it rather than dropping the
    patients quietly into neither stratum.
    """
    col = str(rule["column"])
    assert col in frame.columns, (
        f"subgroup rule needs column {col!r}, which is not on the frame "
        f"(have {sorted(frame.columns)[:12]}...)")
    s = frame[col]
    op = str(rule["op"])
    if op in ("is_true", "is_false"):
        def _truthy(v) -> bool:
            if v is None or (isinstance(v, float) and v != v):
                return False                      # NaN / None satisfies neither level
            return bool(v is True or v == 1)
        present = s.notna().to_numpy(dtype=bool)
        truth = s.astype("object").map(_truthy).to_numpy(dtype=bool)
        return truth if op == "is_true" else (~truth & present)
    val = _resolve_value(rule, cfg)
    if op == "between":
        lo, hi = (float(val[0]), float(val[1]))
        x = pd.to_numeric(s, errors="coerce")
        return ((x >= lo) & (x <= hi)).fillna(False).to_numpy(dtype=bool)
    if op == "eq":
        return (s == val).fillna(False).to_numpy(dtype=bool)
    if op == "ne":
        return ((s != val) & s.notna()).to_numpy(dtype=bool)
    x = pd.to_numeric(s, errors="coerce")
    v = float(val)
    cmp = {"lt": x < v, "le": x <= v, "gt": x > v, "ge": x >= v}[op]
    return cmp.fillna(False).to_numpy(dtype=bool)


# --------------------------------------------------------------------------- #
# Logging: append to run.log (module | ISO-timestamp | level | message).      #
# --------------------------------------------------------------------------- #
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a")  # APPEND, never truncate
        fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"))
        lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        lg.addHandler(sh)
    return lg


# --------------------------------------------------------------------------- #
# Patient-level frame: one row per final-cohort patient with every subgroup    #
# attribute. Demographics joined; obesity = per-patient LIFETIME MAX over the   #
# streamed ICD table (per-diagnosis-line flag aggregated MAX GROUP BY patient). #
# --------------------------------------------------------------------------- #
def build_patient_frame(con, cfg, log) -> pd.DataFrame:
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    fc_path = (coh / "final_cohort.parquet").as_posix()
    demo_path = cfg.parquet_path("demographics").as_posix()
    icd_path = cfg.parquet_path("icd").as_posix()

    sex_col = cfg["subgroups"]["sex_col"]
    race_col = cfg["subgroups"]["race_col"]
    obesity_flag = cfg["subgroups"]["obesity_flag"]

    # obesity = MAX(obesity) GROUP BY empi_anon, restricted (semi-join) to the
    # final-cohort patients — identical per-patient MAX as computing globally,
    # but streams only the needed ICD rows. LEFT JOIN => patients with no ICD
    # line -> COALESCE 0 (no obesity diagnosis on record).
    pt = con.execute(f"""
        WITH ob AS (
            SELECT empi_anon, MAX({obesity_flag}) AS obese
            FROM read_parquet('{icd_path}')
            WHERE empi_anon IN (SELECT empi_anon FROM read_parquet('{fc_path}'))
            GROUP BY empi_anon
        )
        SELECT
            f.empi_anon,
            d.{sex_col}                       AS sex,
            d.{race_col}                      AS race,
            f.age_at_index                    AS age_at_index,
            f.weight_bearing_frontal          AS weight_bearing_frontal,
            f.view_set                        AS view_set,
            f.side_source                     AS side_source,
            CAST(YEAR(f.study_date) AS INTEGER) AS acquisition_year,
            CAST(f.primary_event AS INTEGER)  AS primary_event,
            COALESCE(ob.obese, 0)             AS {obesity_flag}
        FROM read_parquet('{fc_path}') f
        LEFT JOIN read_parquet('{demo_path}') d USING (empi_anon)
        LEFT JOIN ob USING (empi_anon)
    """).df()

    assert pt["empi_anon"].is_unique, "final cohort not one-row-per-patient"
    n_obese = int((pt[obesity_flag] == 1).sum())
    # acquisition_year is the calendar year of final_cohort.study_date, which derives from
    # image.StudyDate_anon. That column carries a PER-PATIENT RANDOM SHIFT (see
    # src/inventory.py and outputs/data_quality_report.md), so the year is comparable
    # WITHIN a patient and NOT, without the protocol section-17 written confirmation that
    # D17 records as never obtained, ACROSS patients. The caveat rides on every era row
    # through the family's note column; it is restated here so nobody adds an era
    # consumer without meeting it.
    log.info("loaded final cohort n=%d events=%d | joined sex/race; obesity(lifetime MAX)=%d/%d "
             "patients; side_source and acquisition_year (%d-%d, shifted dates) carried through",
             len(pt), int(pt["primary_event"].sum()), n_obese, len(pt),
             int(pt["acquisition_year"].min()), int(pt["acquisition_year"].max()))
    return pt


# --------------------------------------------------------------------------- #
# Subgroup tally + stability flag.                                             #
# --------------------------------------------------------------------------- #
def make_stability(cfg):
    """Return f(n_events) -> '<{s}' | '{s}-{m-1}' | '>={m}' from event_flag_bins."""
    bins = cfg["subgroups"]["event_flag_bins"]
    sparse = int(bins["sparse"])       # 50
    moderate = int(bins["moderate"])   # 100

    def stability(n_events: int) -> str:
        if n_events >= moderate:
            return f">={moderate}"
        if n_events >= sparse:
            return f"{sparse}-{moderate - 1}"
        return f"<{sparse}"

    return stability


SUBGROUP_COUNT_COLUMNS = ["subgroup_family", "subgroup", "n_patients", "n_events",
                          "event_pct", "stability_flag"]


def build_rows(pt: pd.DataFrame, cfg, stability) -> pd.DataFrame:
    """One row per declared level, in declaration order.

    Nothing about which families exist, what they are called or how a level is selected
    lives in this function any more; it all comes from ``subgroups.families``. The only
    thing that stays here is the tally itself. Notable rules, all now in config:

    * race is three SELECTED groups and is declared ``partition: false``;
    * age uses strict ``<`` / ``>=`` around ``subgroups.age_cutoff``, so a NaN age would
      fall to neither bucket — which :func:`validate` is what catches;
    * multi-view is the strict complement of the single-token ``frontal`` view set, so the
      split is guaranteed to partition; it equals "has lateral or sunrise" for all but 2
      ``frontal+other`` studies (both non-events), which are correctly multi-view.
    """
    rows: list[dict] = []
    for fam in load_families(cfg, "cohort"):
        for lv in fam.levels:
            m = family_mask(lv.rule, pt, cfg)
            n = int(m.sum())
            e = int(pt.loc[m, "primary_event"].sum())
            rows.append({
                "subgroup_family": fam.cohort_label,
                "subgroup": lv.cohort_label,
                "n_patients": n,
                "n_events": e,
                "event_pct": round(100.0 * e / n, 2) if n else 0.0,
                "stability_flag": stability(e),
            })
    return pd.DataFrame(rows, columns=SUBGROUP_COUNT_COLUMNS)


# --------------------------------------------------------------------------- #
# Validation — regression anchors + fresh-split partition checks.              #
# --------------------------------------------------------------------------- #
#: Locked event/patient counts for the primary final landmark cohort (recovery_any /
#: 730-day, 3,709 patients / 533 primary events). These are REGRESSION ANCHORS, not
#: parameters: if the extraction changes, one of these fails and the run stops.
#:
#: The first seven were here before 2026-08-11 and are unchanged. The rest were added
#: with the laterality-source and acquisition-era families so those strata are pinned to
#: the same standard as the original six, rather than being merely partition-checked.
#: An anchor is (subgroup_family, subgroup, column) -> value, using the COHORT labels.
ANCHORS: dict[tuple[str, str, str], int] = {
    # --- original, unchanged --------------------------------------------------- #
    ("sex", "Female", "n_events"): 342,
    ("sex", "Male", "n_events"): 191,
    ("age_at_index", "<65", "n_events"): 228,
    ("age_at_index", ">=65", "n_events"): 305,
    ("race", "Black", "n_events"): 156,
    ("race", "White", "n_events"): 328,
    ("race", "Asian", "n_events"): 13,
    # --- laterality source (D2 recovered index side = 49.3% of the cohort) ----- #
    ("laterality_source", "coded", "n_patients"): 1881,
    ("laterality_source", "coded", "n_events"): 394,
    ("laterality_source", "recovered", "n_patients"): 1828,
    ("laterality_source", "recovered", "n_events"): 139,
    # --- acquisition era, all three declared schemes --------------------------- #
    ("acquisition_era_calendar", "2007-2011", "n_patients"): 214,
    ("acquisition_era_calendar", "2007-2011", "n_events"): 1,
    ("acquisition_era_calendar", "2012-2016", "n_patients"): 1086,
    ("acquisition_era_calendar", "2012-2016", "n_events"): 80,
    ("acquisition_era_calendar", "2017-2021", "n_patients"): 2409,
    ("acquisition_era_calendar", "2017-2021", "n_events"): 452,
    ("acquisition_era_devtertile", "<=2015", "n_patients"): 935,
    ("acquisition_era_devtertile", "<=2015", "n_events"): 44,
    ("acquisition_era_devtertile", "2016-2018", "n_patients"): 1284,
    ("acquisition_era_devtertile", "2016-2018", "n_events"): 205,
    ("acquisition_era_devtertile", ">=2019", "n_patients"): 1490,
    ("acquisition_era_devtertile", ">=2019", "n_events"): 284,
    ("acquisition_era_devmedian", "<=2017", "n_patients"): 1714,
    ("acquisition_era_devmedian", "<=2017", "n_events"): 155,
    ("acquisition_era_devmedian", ">=2018", "n_patients"): 1995,
    ("acquisition_era_devmedian", ">=2018", "n_events"): 378,
}


def validate(df: pd.DataFrame, n_total: int, n_events_total: int, log, cfg=None) -> None:
    """Anchors, then a config-driven partition check on every ``partition: true`` family.

    The partition list used to be three hard-coded family names. It is now read off the
    ``partition`` flag in ``subgroups.families``, so a new family is partition-checked the
    moment it is declared and cannot be added without either partitioning or saying in
    config that it does not.
    """
    def get(family, subgroup, col):
        sel = df[(df.subgroup_family == family) & (df.subgroup == subgroup)]
        assert len(sel) == 1, f"missing/dup subgroup {family}/{subgroup}"
        return int(sel.iloc[0][col])

    # --- ANCHORS (re-gate recovery_any/730 counts) ------------------------- #
    for (fam, sg, col), exp in ANCHORS.items():
        got = get(fam, sg, col)
        assert got == exp, f"ANCHOR FAIL: {fam}/{sg} {col}={got} != {exp}"

    # sex & age event partitions of the 533
    assert get("sex", "Female", "n_events") + get("sex", "Male", "n_events") == n_events_total
    assert get("age_at_index", "<65", "n_events") + get("age_at_index", ">=65", "n_events") == n_events_total
    # sex & age patient partitions of N (explicit anchor)
    assert get("sex", "Female", "n_patients") + get("sex", "Male", "n_patients") == n_total
    assert get("age_at_index", "<65", "n_patients") + get("age_at_index", ">=65", "n_patients") == n_total

    # --- every declared partition family sums to N and to the event total --- #
    checked: list[str] = []
    if cfg is not None:
        for fam in load_families(cfg, "cohort"):
            if not fam.partition:
                continue
            np_ = sum(get(fam.cohort_label, lv.cohort_label, "n_patients") for lv in fam.levels)
            ne_ = sum(get(fam.cohort_label, lv.cohort_label, "n_events") for lv in fam.levels)
            assert np_ == n_total, f"{fam.key} patient split {np_} != {n_total}"
            assert ne_ == n_events_total, f"{fam.key} event split {ne_} != {n_events_total}"
            checked.append(fam.key)

    # --- no identifier leakage in the persisted frame ---------------------- #
    assert not any("empi" in c.lower() for c in df.columns), "identifier column in output"

    log.info("VALIDATION OK — %d anchor(s) match; %d partition famil(ies) sum to N=%d, "
             "events=%d: %s", len(ANCHORS), len(checked), n_total, n_events_total,
             ", ".join(checked) if checked else "none checked (no cfg passed)")


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOCKED subgroup event tallies (read-only).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    outputs_dir = cfg.out("outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    age_cut = int(cfg["subgroups"]["age_cutoff"])
    bins = cfg["subgroups"]["event_flag_bins"]
    log.info("START LOCKED subgroups (primary=recovery_any/730 final cohort; age_cutoff=%d; "
             "stability bins sparse=%d moderate=%d; obesity=lifetime MAX per patient)",
             age_cut, int(bins["sparse"]), int(bins["moderate"]))

    con = duckdb.connect()
    try:
        pt = build_patient_frame(con, cfg, log)
        n_total = int(len(pt))
        n_events_total = int(pt["primary_event"].sum())

        stability = make_stability(cfg)
        df = build_rows(pt, cfg, stability)
        validate(df, n_total, n_events_total, log, cfg=cfg)

        out_path = outputs_dir / "subgroup_counts.csv"
        df.to_csv(out_path, index=False)

        # log each family + flag the suppressed (<sparse events) strata.
        for r in df.itertuples(index=False):
            log.info("%-22s %-14s n=%-5d events=%-4d (%.2f%%) [%s]",
                     r.subgroup_family, r.subgroup, r.n_patients, r.n_events,
                     r.event_pct, r.stability_flag)
        sparse = int(bins["sparse"])
        suppressed = df[df["n_events"] < sparse]
        if len(suppressed):
            log.info("SUPPRESS (<%d events, protocol sec 21): %s", sparse,
                     ", ".join(f"{r.subgroup_family}/{r.subgroup}(events={r.n_events})"
                               for r in suppressed.itertuples(index=False)))
        else:
            log.info("SUPPRESS (<%d events): none", sparse)

        log.info("DONE — wrote subgroup_counts.csv (%d rows) -> %s", len(df), out_path)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
