"""subgroups.py — LOCKED subgroup event tallies over the PRIMARY final cohort.

MRKR Contralateral TKA — LOCKED EXTRACTION. For the primary final landmark
cohort (recovery_any / 730-day = 3,709 patients / 533 primary events, VERIFIED)
this module reports, for every protocol subgroup, the patient count, the primary
(recovery_any / 5-year) event count, the event percentage, and a stability flag
('>=100' | '50-99' | '<50', protocol section 21 sparse/moderate bins) used to
decide which strata are reportable vs suppressed.

Subgroup families (config-driven; see config/feasibility.yaml -> subgroups):
  * sex                     : Female, Male                         (partition)
  * age at index            : <65, >=65   (cutoff = subgroups.age_cutoff)  (partition)
  * race                    : Black / White / Asian (3 selected groups only;
                              NOT a partition — other race values are excluded)
  * obesity                 : yes, no   (lifetime MAX(obesity) per patient)  (partition)
  * imaging weight-bearing  : WB, non-WB  (final_cohort.weight_bearing_frontal)  (partition)
  * imaging views           : frontal-only (view_set=='frontal') vs multi-view
                              (any non-frontal-only study)                  (partition)

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
  patients == 3,709. Obesity / WB / view-set splits are computed fresh (no anchor)
  and each is asserted to sum to 3,709 patients and 533 events.

Run from the project root::

    python3 -m src.subgroups --config config/feasibility.yaml

Writes ONLY: outputs/subgroup_counts.csv, and appends outputs/logs/run.log.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.config import load_config

MODULE = "subgroups"


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
            CAST(f.primary_event AS INTEGER)  AS primary_event,
            COALESCE(ob.obese, 0)             AS obese
        FROM read_parquet('{fc_path}') f
        LEFT JOIN read_parquet('{demo_path}') d USING (empi_anon)
        LEFT JOIN ob USING (empi_anon)
    """).df()

    assert pt["empi_anon"].is_unique, "final cohort not one-row-per-patient"
    n_obese = int((pt["obese"] == 1).sum())
    log.info("loaded final cohort n=%d events=%d | joined sex/race; obesity(lifetime MAX)=%d/%d patients",
             len(pt), int(pt["primary_event"].sum()), n_obese, len(pt))
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


def build_rows(pt: pd.DataFrame, cfg, stability) -> pd.DataFrame:
    age_cut = int(cfg["subgroups"]["age_cutoff"])
    rg = cfg["subgroups"]["race_groups"]

    rows: list[dict] = []

    def add(family: str, subgroup: str, mask: pd.Series) -> None:
        sub = pt[mask]
        n = int(len(sub))
        e = int(sub["primary_event"].sum())
        pct = round(100.0 * e / n, 2) if n else 0.0
        rows.append({
            "subgroup_family": family,
            "subgroup": subgroup,
            "n_patients": n,
            "n_events": e,
            "event_pct": pct,
            "stability_flag": stability(e),
        })

    # sex (partition)
    add("sex", "Female", pt["sex"] == "Female")
    add("sex", "Male", pt["sex"] == "Male")

    # age at index (partition; NaN ages, if any, fall to neither bucket by design
    # of the strict < / >= comparison — asserted to still sum to N below)
    add("age_at_index", "<65", pt["age_at_index"] < age_cut)
    add("age_at_index", ">=65", pt["age_at_index"] >= age_cut)

    # race (3 SELECTED groups only — NOT a partition of the cohort)
    add("race", "Black", pt["race"] == rg["black"])
    add("race", "White", pt["race"] == rg["white"])
    add("race", "Asian", pt["race"] == rg["asian"])

    # obesity — lifetime MAX(obesity) per patient (partition)
    add("obesity", "yes", pt["obese"] == 1)
    add("obesity", "no", pt["obese"] == 0)

    # imaging weight-bearing — final_cohort.weight_bearing_frontal (partition)
    add("imaging_weight_bearing", "WB", pt["weight_bearing_frontal"] == True)      # noqa: E712
    add("imaging_weight_bearing", "non-WB", pt["weight_bearing_frontal"] == False)  # noqa: E712

    # imaging views — frontal-only vs multi-view. frontal-only is exactly the
    # single-token 'frontal' view_set; multi-view is its strict complement (any
    # study carrying a non-frontal view). Complement guarantees the partition
    # sums to N/events; it equals "has lateral or sunrise" for all but 2
    # 'frontal+other' studies (both non-events), which are correctly multi-view.
    add("imaging_views", "frontal-only", pt["view_set"] == "frontal")
    add("imaging_views", "multi-view", pt["view_set"] != "frontal")

    return pd.DataFrame(rows, columns=[
        "subgroup_family", "subgroup", "n_patients", "n_events",
        "event_pct", "stability_flag"])


# --------------------------------------------------------------------------- #
# Validation — regression anchors + fresh-split partition checks.              #
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, n_total: int, n_events_total: int, log) -> None:
    def get(family, subgroup, col):
        sel = df[(df.subgroup_family == family) & (df.subgroup == subgroup)]
        assert len(sel) == 1, f"missing/dup subgroup {family}/{subgroup}"
        return int(sel.iloc[0][col])

    # --- ANCHORS (re-gate recovery_any/730 event counts) ------------------- #
    anchors = {
        ("sex", "Female", "n_events"): 342,
        ("sex", "Male", "n_events"): 191,
        ("age_at_index", "<65", "n_events"): 228,
        ("age_at_index", ">=65", "n_events"): 305,
        ("race", "Black", "n_events"): 156,
        ("race", "White", "n_events"): 328,
        ("race", "Asian", "n_events"): 13,
    }
    for (fam, sg, col), exp in anchors.items():
        got = get(fam, sg, col)
        assert got == exp, f"ANCHOR FAIL: {fam}/{sg} {col}={got} != {exp}"

    # sex & age event partitions of the 533
    assert get("sex", "Female", "n_events") + get("sex", "Male", "n_events") == n_events_total
    assert get("age_at_index", "<65", "n_events") + get("age_at_index", ">=65", "n_events") == n_events_total
    # sex & age patient partitions of N (explicit anchor)
    assert get("sex", "Female", "n_patients") + get("sex", "Male", "n_patients") == n_total
    assert get("age_at_index", "<65", "n_patients") + get("age_at_index", ">=65", "n_patients") == n_total

    # --- FRESH SPLITS (no anchor) — assert each is a clean N / events partition #
    for fam, a, b in [("obesity", "yes", "no"),
                      ("imaging_weight_bearing", "WB", "non-WB"),
                      ("imaging_views", "frontal-only", "multi-view")]:
        np_ = get(fam, a, "n_patients") + get(fam, b, "n_patients")
        ne_ = get(fam, a, "n_events") + get(fam, b, "n_events")
        assert np_ == n_total, f"{fam} patient split {np_} != {n_total}"
        assert ne_ == n_events_total, f"{fam} event split {ne_} != {n_events_total}"

    # --- no identifier leakage in the persisted frame ---------------------- #
    assert not any("empi" in c.lower() for c in df.columns), "identifier column in output"

    log.info("VALIDATION OK — anchors match; sex/age/obesity/WB/views partitions sum to N=%d, events=%d",
             n_total, n_events_total)


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
        validate(df, n_total, n_events_total, log)

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
