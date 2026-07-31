"""cohort_flow.py — Stage-4 SEQUENTIAL, NON-OVERLAPPING 11-step exclusion table.

MRKR Contralateral TKA — Phase-1 metadata-only feasibility (LOCKED extraction).
Builds the prompt-1 Stage-4 cohort-flow table for the PRIMARY landmark cohort
(recovery_any / 730-day pre-index imaging window) alongside a parallel STRICT
sensitivity column (coded-laterality index only / 365-day window).

Each step count is the number of patients surviving ALL prior exclusions
(cumulative set intersection), applied in this EXACT order:

  1  Total patients in demographics                                    (83,011)
  2  Patients with any knee radiograph                                 (83,011)
  3  Patients with CPT 27447                                           ( 8,525)
  4  Aged >=40 at index
  5  With exactly one interpretable index side
        (primary: coded OR recovered single side; strict: coded only)
  6  Without prior contralateral arthroplasty
  7  Without protocol infection/osteomyelitis exclusion (high-spec)
  8  With an eligible pre-index contralateral study
        (primary window 730d; strict 365d)
  9  Without contralateral CPT 27447 through day 90
 10  Observed through day 90
 11  Final primary landmark cohort

Provenance of each gate (REUSE, do not redefine):
  * steps 3-7  -> derived-data/cohort/index_candidates.parquet gate booleans
                  (age via literal age_at_index >= age_min so the side='none'
                   population — whose age_ok is NULL — is dropped at the SIDE
                   gate, step 5, NOT the age gate; this honours the prompt's
                   "apply age>=40 BEFORE the side gate" ordering).
  * step 8     -> candidate_studies.parquet in_window_{730|365} flags.
  * step 9     -> outcomes.parquet  NOT has_contra_27447_day_0_90.
  * step 10    -> followup.parquet  observed_through_90.
  * steps 1-2  -> source-parquet/{demographics,image}.parquet (patient universe).

Guardrails honoured: read-only on all Parquet inputs; deterministic; no DICOMs,
no models, no metrics; NO patient identifiers (empi_anon) are ever written — the
persisted CSV is AGGREGATE COUNTS only (survivor SETS live in memory purely for
reconciliation). Logging APPENDS to outputs/logs/run.log with the cohort_flow
prefix. No git, no network.

Run from the project root::

    python3 -m src.cohort_flow --config config/feasibility.yaml

Writes ONLY: outputs/cohort_flow.csv, and appends outputs/logs/run.log.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config

MODULE = "cohort_flow"

# --------------------------------------------------------------------------- #
# Regression anchors (assert). These reconcile to final_cohort.parquet and the #
# verified re-gate (src/regate.py -> outputs/tables/regate_grid.csv) and to the #
# index-level index_flow.csv.                                                   #
# --------------------------------------------------------------------------- #
ANCHOR_STEP3 = 8525          # all CPT-27447 patients (both columns)
ANCHOR_PRIMARY_STEP7 = 6381  # recovery_any index-level count (== index_final.parquet)
ANCHOR_PRIMARY_STEP11 = 3709  # LOCKED primary cohort
ANCHOR_STRICT_STEP11 = 1664  # strict/365 sensitivity cohort

# Step descriptions (shared column; parentheticals name the primary/strict variant).
STEP_DESCRIPTIONS = [
    "Total patients in demographics",
    "Patients with any knee radiograph",
    "Patients with CPT 27447",
    "Aged >=40 at index",
    "With exactly one interpretable index side (primary: coded or recovered single side; strict: coded only)",
    "Without prior contralateral arthroplasty",
    "Without protocol infection/osteomyelitis exclusion (high-specificity)",
    "With an eligible pre-index contralateral study (primary window 730d; strict 365d)",
    "Without contralateral CPT 27447 through day 90",
    "Observed through day 90",
    "Final primary landmark cohort",
]


def setup_logging(log_path: Path) -> logging.Logger:
    """Append-only logger: 'cohort_flow | ISO-timestamp | level | message'."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(MODULE)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a")  # APPEND, never truncate
        fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        logger.addHandler(sh)
    return logger


def _keep_true(df: pd.DataFrame, col: str) -> set:
    """empi_anon of rows where a BOOLEAN column is True (NULL -> excluded)."""
    return set(df.loc[df[col] == True, "empi_anon"])  # noqa: E712 (NULL-safe on purpose)


def cumulative_flow(step_keepsets: list) -> list[int]:
    """Sequential cumulative intersection. Element 0 seeds the surviving set;
    a keep-set of None (step 11) is an identity pass-through (no new filter)."""
    surv: set | None = None
    counts: list[int] = []
    for ks in step_keepsets:
        if surv is None:
            surv = set(ks)
        elif ks is None:
            pass  # identity (final landmark cohort = survivors after step 10)
        else:
            surv = surv & ks
        counts.append(len(surv))
    return counts, surv  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage-4 sequential 11-step cohort-flow table (read-only).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    src = cfg.path(cfg["paths"]["source_parquet_dir"])
    out_csv = cfg.out("outputs_dir") / "cohort_flow.csv"

    age_min = float(cfg["index"]["age_min"])
    # Windows: primary from the LOCKED primary_definition; strict = 365-day sensitivity.
    primary_window = int(cfg["primary_definition"]["pre_index_window_days"][1])   # 730
    strict_window = 365
    prim_win_col = f"in_window_{primary_window}"
    strict_win_col = f"in_window_{strict_window}"

    log.info("START cohort-flow: primary=recovery_any/%dd strict=coded/%dd age_min=%.0f",
             primary_window, strict_window, age_min)

    # ---- inputs (read-only; pandas/pyarrow — resilient to slow I/O) --------- #
    # (Deliberately NOT duckdb: under heavy concurrent load its parquet reader
    #  aborts on EINTR; pyarrow tolerates the same slow syscalls gracefully.)
    ic = pd.read_parquet(coh / "index_candidates.parquet")
    cand = pd.read_parquet(coh / "candidate_studies.parquet",
                           columns=["empi_anon", prim_win_col, strict_win_col])
    outc = pd.read_parquet(coh / "outcomes.parquet",
                           columns=["empi_anon", "has_contra_27447_day_0_90"])
    fu = pd.read_parquet(coh / "followup.parquet",
                         columns=["empi_anon", "observed_through_90"])
    fc_ids = set(pd.read_parquet(coh / "final_cohort.parquet",
                                 columns=["empi_anon"])["empi_anon"])

    # Patient universe for steps 1-2 (distinct empi). image.parquet is the large
    # (~51 MB) input; read only its empi column (columnar) to keep the scan light.
    demo_ids = set(pd.read_parquet(src / "demographics.parquet",
                                   columns=["empi_anon"])["empi_anon"])
    img_ids = set(pd.read_parquet(src / "image.parquet",
                                  columns=["empi_anon"])["empi_anon"])
    log.info("universe: demographics=%d image(distinct empi)=%d | index_candidates=%d "
             "candidate_studies rows=%d outcomes=%d followup=%d final_cohort=%d",
             len(demo_ids), len(img_ids), len(ic), len(cand), len(outc), len(fu), len(fc_ids))

    # ---- gate keep-sets ----------------------------------------------------- #
    cpt_ids = set(ic["empi_anon"])                                      # step 3
    age_ids = set(ic.loc[ic["age_at_index"] >= age_min, "empi_anon"])   # step 4 (literal)
    side_primary = set(ic.loc[ic["side_source"].isin(["coded", "recovered"]), "empi_anon"])
    side_strict = set(ic.loc[ic["side_source"] == "coded", "empi_anon"])
    prior_ids = _keep_true(ic, "no_prior_contra_arthroplasty")         # step 6
    inf_ids = _keep_true(ic, "no_infection_highspec")                  # step 7
    elig_primary = _keep_true(cand, prim_win_col)                      # step 8 (730d)
    elig_strict = _keep_true(cand, strict_win_col)                     # step 8 (365d)
    keep_s9 = set(outc.loc[~outc["has_contra_27447_day_0_90"].astype(bool), "empi_anon"])
    keep_s10 = _keep_true(fu, "observed_through_90")                   # step 10

    # ---- assemble the two ordered cumulative flows -------------------------- #
    def steps_for(side_ids: set, elig_ids: set) -> list:
        return [
            demo_ids,       # 1 total demographics
            img_ids,        # 2 any knee radiograph
            cpt_ids,        # 3 CPT 27447
            age_ids,        # 4 aged >=40
            side_ids,       # 5 interpretable index side
            prior_ids,      # 6 no prior contralateral arthroplasty
            inf_ids,        # 7 no infection/osteomyelitis (high-spec)
            elig_ids,       # 8 eligible pre-index contralateral study
            keep_s9,        # 9 no contralateral CPT 27447 day 0-90
            keep_s10,       # 10 observed through day 90
            None,           # 11 final landmark cohort (identity of step 10)
        ]

    n_primary, surv_primary = cumulative_flow(steps_for(side_primary, elig_primary))
    n_strict, _ = cumulative_flow(steps_for(side_strict, elig_strict))

    # excluded[i] = prior_n - this_n (step 1 has no prior -> 0).
    def excluded(seq: list[int]) -> list[int]:
        return [0] + [seq[i - 1] - seq[i] for i in range(1, len(seq))]

    exc_primary = excluded(n_primary)
    exc_strict = excluded(n_strict)

    # ---- VALIDATION --------------------------------------------------------- #
    def _mono(seq: list[int], name: str) -> None:
        assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)), \
            f"{name} not monotone non-increasing: {seq}"

    _mono(n_primary, "n_primary")
    _mono(n_strict, "n_strict")
    assert n_primary[2] == ANCHOR_STEP3, f"primary step3 {n_primary[2]} != {ANCHOR_STEP3}"
    assert n_strict[2] == ANCHOR_STEP3, f"strict step3 {n_strict[2]} != {ANCHOR_STEP3}"
    assert n_primary[6] == ANCHOR_PRIMARY_STEP7, \
        f"primary step7 {n_primary[6]} != {ANCHOR_PRIMARY_STEP7} (index-level cross-check)"
    assert n_primary[10] == ANCHOR_PRIMARY_STEP11, \
        f"primary step11 {n_primary[10]} != {ANCHOR_PRIMARY_STEP11}"
    assert n_strict[10] == ANCHOR_STRICT_STEP11, \
        f"strict step11 {n_strict[10]} != {ANCHOR_STRICT_STEP11}"
    # step-11 survivor SET must equal the LOCKED final_cohort.parquet exactly.
    assert surv_primary == fc_ids, \
        f"primary step-11 set != final_cohort.parquet (symdiff={len(surv_primary ^ fc_ids)})"

    # index-level cross-check against index_flow.csv (recovery_any n_no_infection == 6,381).
    idx_flow_path = cfg.out("tables_dir") / "index_flow.csv"
    if idx_flow_path.exists():
        idxflow = pd.read_csv(idx_flow_path)
        ra = idxflow.loc[idxflow["strategy"] == "recovery_any"]
        if not ra.empty:
            v = int(ra.iloc[0]["n_no_infection"])
            assert v == ANCHOR_PRIMARY_STEP7, \
                f"index_flow.csv recovery_any n_no_infection {v} != step7 {ANCHOR_PRIMARY_STEP7}"
            log.info("cross-check OK: index_flow.csv recovery_any n_no_infection=%d == step7", v)

    # ---- WRITE cohort_flow.csv (AGGREGATE only; no empi) -------------------- #
    table = pd.DataFrame({
        "step": list(range(1, len(STEP_DESCRIPTIONS) + 1)),
        "description": STEP_DESCRIPTIONS,
        "n_primary": n_primary,
        "n_excluded_primary": exc_primary,
        "n_strict": n_strict,
        "n_excluded_strict": exc_strict,
    })
    assert not any("empi" in c.lower() for c in table.columns), "identifier column in output"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    # ---- log the flow + the biggest single analytic exclusion --------------- #
    for _, r in table.iterrows():
        log.info("STEP %2d | prim n=%d (-%d) | strict n=%d (-%d) | %s",
                 int(r["step"]), int(r["n_primary"]), int(r["n_excluded_primary"]),
                 int(r["n_strict"]), int(r["n_excluded_strict"]), r["description"])
    # Biggest single exclusion among the analytic gates (steps 4-11; step 3 is the
    # source-population definition, not a cohort exclusion).
    analytic = table[table["step"] >= 4]
    big = analytic.loc[analytic["n_excluded_primary"].idxmax()]
    log.info("biggest single analytic exclusion (primary, steps 4-11): step %d '%s' "
             "excluded %d (should be step 8 imaging window)",
             int(big["step"]), big["description"], int(big["n_excluded_primary"]))
    log.info("ANCHORS OK: step3=%d/%d step7_primary=%d step11 primary=%d strict=%d; "
             "step-11 set == final_cohort.parquet. wrote %s",
             n_primary[2], n_strict[2], n_primary[6], n_primary[10], n_strict[10], out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
