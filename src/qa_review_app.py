"""qa_review_app.py — the local scoring interface for the protocol section 23 / 7 audits.

=============================================================================
README FOR A REVIEWER (you do not need to know any Python to use this)
=============================================================================

WHAT THIS IS
    Three review lists have been prepared for you. This program shows you one
    item at a time in your web browser and saves your answers as you go. It runs
    entirely on this computer. Nothing is uploaded anywhere, and it works with the
    Wi-Fi switched off.

HOW TO START IT
    Open the Terminal app, then copy and paste ONE of these lines and press Return.
    Replace `jdoe` with your own short name (letters and digits, no spaces) -- your
    answers are saved in a file named after you, so the two reviewers never overwrite
    each other.

        cd "/Users/samersalman/Desktop/Radiographic Prediction of Contralateral Knee Arthroplasty"
        python3 -m src.qa_review_app --mode image       --reviewer jdoe
        python3 -m src.qa_review_app --mode laterality  --reviewer jdoe
        python3 -m src.qa_review_app --mode outcome     --reviewer jdoe

    A browser tab opens by itself. If it does not, open a browser and go to the
    address the Terminal prints (it looks like http://127.0.0.1:8765).

    Do all three lists. `image` is the long one (400 radiographs); `laterality`
    (200 patients) and `outcome` (200 records) are shorter.

STOPPING AND COMING BACK
    Every answer is written to disk the moment you give it. Close the tab and press
    Ctrl+C in the Terminal whenever you like. Start it again with the same command
    and the same name and it reopens on the first item you have not answered yet.
    Nothing is lost, and you can go back and change an earlier answer at any time.

DO NOT LOOK AT THE OTHER REVIEWER'S ANSWERS
    The two reviews must be independent. Score the whole list before either of you
    compares notes; the disagreements are the measurement.

THE KEYS (image mode)
    o          this item is OK          (the statement on screen is true)
    e          this item is an ERROR    (the statement on screen is false)
    u          cannot be assessed       (see NOT ASSESSABLE below)
               ... each of these moves you to the next item automatically
    1 - 6      jump to item 1-6 to change an answer
    a          mark every unanswered item on this image OK
    Return     save this image and go to the next one
    <- / ->    previous / next image (your answers are kept)
    /          type a note about this image; press Esc to leave the note box
    ?          show the key list on screen

    You can also just click the buttons. The keys are there because there are 400
    images and clicking 2,400 buttons is not a reasonable thing to ask of anyone.

THE SIX ITEMS (image mode)
    Each item is a STATEMENT about the picture on screen. Answer OK if the statement
    is true, ERROR if it is false. The wording comes from protocol section 23 and
    from `crop_qa.score_items` in config/feasibility.yaml.

    1. laterality       This is the CONTRALATERAL knee -- the side OPPOSITE the knee
                        that was already replaced. Judge this on the LEFT panel: it
                        shows the whole original film with the half the pipeline kept
                        outlined in green and the discarded INDEX half marked in red.
    2. view             The view label in the header (frontal / lateral / sunrise) is
                        the view actually shown.
    3. native_knee      The knee is NATIVE. There is no knee replacement, no
                        prosthesis and no arthroplasty hardware in this knee.
    4. crop_adequacy    The crop is adequate: the whole tibiofemoral joint is inside
                        the frame, the joint line is not cut off, and the image is
                        usable.
    5. burned_in_text   NO burned-in laterality marker, letter, arrow or other text
                        survives inside the crop.
    6. non_knee_content NO gross non-knee content: no second knee, no hip, no ankle,
                        nothing that is not this one knee.

    NOT ASSESSABLE (the `u` key) is a real answer, not a way of skipping. Use it when
    you looked and the picture cannot answer the question. This happens on purpose for
    item 1 on the 82 test-split images: their original films are no longer available,
    so the left panel reads "NO FULL FILM" and the correct half-select is undecidable
    from a finished crop. Every pre-index film here shows TWO NATIVE knees, the crop is
    mirrored to a left knee, and the border mask removes the L/R marker -- so a crop is
    anatomically identical whether the right half or the wrong half was taken. Guessing
    would be worse than useless. Answer `u` on item 1 for those, and answer items 2-6
    normally, because those ARE decidable from the crop.

    Leaving an item blank means "not reviewed yet". It is not the same as `u`.

=============================================================================
NOTES FOR WHOEVER MAINTAINS THIS
=============================================================================

WHY A LOCAL SERVER AND NOT A SELF-CONTAINED HTML FILE
    A single .html file cannot write to `derived-data/cohort/scores/`. A browser can
    only hand a file to the download folder, so resumability would have to live in
    localStorage -- one cleared cache, one private window, one wrong "Save As" and
    2,400 judgements are gone, and the output would land somewhere the pipeline does
    not read. `http.server` is in the standard library, so this still installs nothing
    (see requirements.txt: nothing here is outside the stdlib except pandas, which the
    project already pins). It binds 127.0.0.1 only, so it is unreachable from the
    network and works with no network at all. Every answer is flushed to the CSV
    before the HTTP response returns, and an append-only JSONL beside it records every
    submission including corrections, so the review has an audit trail.

DATA HYGIENE
    Everything this module writes lands under `derived-data/`, which is git-ignored.
    The score files carry `empi_anon` because the workbooks do and a reviewer needs to
    be able to look a case up. Nothing goes to `outputs/`.

FLOW
    src.crop_qa --rebuild-image-audit     ->  image_audit_workbook.csv + qa_panels/
    src.qa_review_app --mode image ...    ->  scores/image_audit_scores_<reviewer>.csv
    src.qa_review_app --mode image --merge ->  writes <item>_r<k> back into the workbook
    src.crop_qa --score <workbook>        ->  agreement, Cohen kappa, the 2% gate

Run:  python3 -m src.qa_review_app --mode image --reviewer jdoe
      python3 -m src.qa_review_app --mode image --merge --reviewers jdoe,asmith
      python3 -m src.qa_review_app --mode image --status
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.crop_qa import (
    AUDIT_PATIENT_CSV, IMAGE_AUDIT_WORKBOOK, OUTCOME_AUDIT_WORKBOOK, QA_PANEL_DIR,
    SCORE_ERROR, SCORE_NA, SCORE_OK, setup_logging,
)

APP_VERSION = "1.0"
SCORES_DIR = "scores"
DEFAULT_PORT = 8765

# The three verdicts a reviewer can give an image item, and the key that gives it.
VERDICTS = [(SCORE_OK, "o", "OK"), (SCORE_ERROR, "e", "ERROR"), (SCORE_NA, "u", "NOT ASSESSABLE")]

# One keystroke per answer value, across all three modes. `a`, `/`, `?`, Return and the
# arrow keys are reserved by the page, so nothing here may use them. CANNOT_TELL and
# UNCLEAR share `c` because no mode offers both.
VALUE_KEYS = {SCORE_OK: "o", SCORE_ERROR: "e", SCORE_NA: "u",
              "L": "l", "R": "r", "Y": "y", "N": "n",
              "CANNOT_TELL": "c", "UNCLEAR": "c", "NONE": "x"}

# Item wording. The KEYS come from config crop_qa.score_items; the prose comes from
# protocol section 23 plus the specific failure modes src.preprocess_images can produce.
# Each is a STATEMENT: OK means true, ERROR means false. Do not reword these without
# re-reading section 23 — the polarity of items 5 and 6 is the easy thing to get wrong
# ("burned_in_text = OK" means NO text survived).
ITEM_TEXT = {
    "laterality": ("This is the CONTRALATERAL knee",
                   "The knee shown is the side OPPOSITE the one already replaced. Judge this on "
                   "the LEFT panel: the green outline is the half the pipeline kept, the red "
                   "INDEX label is the half it discarded. If the left panel says NO FULL FILM, "
                   "this item is NOT ASSESSABLE (press u) — do not guess from the crop."),
    "view": ("The view label is correct",
             "The view named in the header above (frontal, lateral or sunrise) is the view "
             "actually shown in the crop."),
    "native_knee": ("The knee is NATIVE",
                    "No knee replacement, prosthesis or arthroplasty hardware in THIS knee. "
                    "Every image in this cohort is pre-index, so a prosthesis here means the "
                    "wrong knee or the wrong study was selected."),
    "crop_adequacy": ("The crop is adequate",
                      "The whole tibiofemoral joint is inside the frame, the joint line is not "
                      "clipped, and the image is diagnostically usable at this size."),
    "burned_in_text": ("NO burned-in text or marker survives",
                       "No laterality marker (L / R), letter, arrow, ruler or other burned-in "
                       "annotation is visible anywhere inside the crop."),
    "non_knee_content": ("NO gross non-knee content",
                         "No second knee, no hip, no ankle, no other body part and no non-image "
                         "content. The crop contains this one knee and nothing else."),
}

# ---- the three review lists -------------------------------------------------
# workbook          : source of rows, under derived-data/cohort/
# fields            : what the reviewer answers, as (name, [allowed values], label)
# merge_targets     : column in the workbook each field folds back into
MODES: dict[str, dict] = {
    "image": {
        "title": "Protocol section 23 (i) — image quality and laterality review",
        "workbook": IMAGE_AUDIT_WORKBOOK,
        "row_key": "key",
        "has_panel": True,
        "fields": None,        # filled from config crop_qa.score_items at load time
        "meta": ["qa_index", "split", "view", "laterality", "contra_side", "index_side",
                 "horizontal_flip", "half_selected", "crop_method", "crop_confidence",
                 "masked_pct", "panel_has_full_film"],
    },
    "laterality": {
        "title": "Protocol section 7 — index laterality audit",
        "workbook": AUDIT_PATIENT_CSV,
        "row_key": "crop_key",
        "has_panel": True,
        "fields": [
            ("reviewer_index_side", ["L", "R", "CANNOT_TELL"],
             "Which side is the INDEX (already replaced) knee?"),
            ("reviewer_agrees_Y_N", ["Y", "N", "CANNOT_TELL"],
             "Does that agree with the recorded index_side shown above?"),
        ],
        "meta": ["split", "side_source", "n_concordant_signals", "index_side", "contra_side",
                 "tier_name", "view_set", "views_in_shards", "n_images", "crop_view"],
    },
    "outcome": {
        "title": "Protocol section 23 (ii) — outcome-record audit",
        "workbook": OUTCOME_AUDIT_WORKBOOK,
        "row_key": "empi_anon",
        "has_panel": False,
        "fields": [
            ("reviewer_event_confirmed_Y_N", ["Y", "N", "UNCLEAR"],
             "Does the CPT chronology show a CONTRALATERAL knee arthroplasty?"),
            ("reviewer_event_side", ["L", "R", "NONE", "UNCLEAR"],
             "Which side did that arthroplasty involve?"),
            ("reviewer_agrees_with_primary_event_Y_N", ["Y", "N", "UNCLEAR"],
             "Do you agree with the derived primary_event flag shown above?"),
        ],
        "free_fields": [("reviewer_event_date", "Event date you would assign (YYYY-MM-DD, "
                                                "blank if none)")],
        "meta": ["split", "side_source", "index_date", "index_side", "contra_side",
                 "primary_event", "event_date", "days_index_to_event", "event_indicator",
                 "time_from_landmark", "landmark_date", "last_observed", "censor_reason",
                 "has_contra_27447_day_0_90", "upper_bound_event", "composite_uni_event",
                 "augmented_event", "n_knee_arthroplasty_cpt_rows", "cpt_chronology"],
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_reviewer(name: str) -> str:
    """Reviewer id reduced to what is safe in a filename. Empty is refused upstream."""
    keep = [c for c in str(name).strip().lower() if c.isalnum() or c in "-_"]
    return "".join(keep)[:40]


# =============================================================================
# The review list: workbook rows + whatever this reviewer has already answered
# =============================================================================
class ReviewList:
    """One mode's rows, one reviewer's answers, and the CSV they are written to."""

    def __init__(self, cfg, mode: str, reviewer: str):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
        self.cfg, self.mode, self.reviewer = cfg, mode, reviewer
        self.spec = dict(MODES[mode])
        self.coh = cfg.path(cfg["paths"]["cohort_dir"])
        if self.spec["fields"] is None:            # image mode reads its items from config
            self.spec["fields"] = [(it, [v for v, _, _ in VERDICTS], ITEM_TEXT[it][0])
                                   for it in list(cfg["crop_qa"]["score_items"])]
        self.field_names = [f[0] for f in self.spec["fields"]]
        self.free_names = [f[0] for f in self.spec.get("free_fields", [])]

        wb_path = self.coh / self.spec["workbook"]
        if not wb_path.exists():
            raise FileNotFoundError(
                f"{wb_path} does not exist. Build it first with `python3 -m src.crop_qa "
                f"--rebuild-image-audit` (image) or `python3 -m src.crop_qa` (all audits).")
        self.workbook_path = wb_path
        self.wb = pd.read_csv(wb_path, dtype=str).fillna("")
        self.panel_dir = self.coh / QA_PANEL_DIR

        self.scores_dir = self.coh / SCORES_DIR
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.scores_dir / f"{mode}_audit_scores_{reviewer}.csv"
        self.log_path = self.scores_dir / f"{mode}_audit_scores_{reviewer}.jsonl"
        self.answers: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    # ---- persistence --------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        return (["reviewer", "mode", "row_index", "row_key", "empi_anon", "split"]
                + [f"score_{f}" for f in self.field_names]
                + [f"score_{f}" for f in self.free_names]
                + ["note", "scored_utc", "seconds_on_panel", "app_version"])

    def _load(self) -> None:
        if not self.csv_path.exists():
            return
        prev = pd.read_csv(self.csv_path, dtype=str).fillna("")
        for r in prev.to_dict("records"):
            self.answers[str(r.get("row_key", ""))] = r

    def _flush(self) -> None:
        """Rewrite the whole CSV atomically. 400 rows — cheap, and it makes edits simple."""
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
            w.writeheader()
            for i, row in enumerate(self.rows_meta()):
                a = self.answers.get(row["row_key"])
                if a:
                    w.writerow({**a, "row_index": i})
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.csv_path)

    def rows_meta(self) -> list[dict]:
        rk = self.spec["row_key"]
        out = []
        for i, r in enumerate(self.wb.to_dict("records")):
            key = str(r.get(rk, ""))
            meta = {c: str(r.get(c, "")) for c in self.spec["meta"] if c in r}
            out.append({"i": i, "row_key": key, "empi_anon": str(r.get("empi_anon", "")),
                        "split": str(r.get("split", "")), "meta": meta,
                        "panel": (f"{key}.png" if self.spec["has_panel"] else ""),
                        "panel_exists": bool(self.spec["has_panel"]
                                             and (self.panel_dir / f"{key}.png").exists())})
        return out

    def save(self, row_key: str, scores: dict, note: str, seconds: float) -> dict:
        with self._lock:
            rows = {r["row_key"]: r for r in self.rows_meta()}
            if row_key not in rows:
                raise KeyError(f"{row_key} is not in {self.spec['workbook']}")
            row = rows[row_key]
            rec = {"reviewer": self.reviewer, "mode": self.mode, "row_index": row["i"],
                   "row_key": row_key, "empi_anon": row["empi_anon"], "split": row["split"],
                   "note": str(note or "").replace("\r", " ").replace("\n", " "),
                   "scored_utc": _utc(), "seconds_on_panel": round(float(seconds or 0.0), 1),
                   "app_version": APP_VERSION}
            allowed = {f[0]: set(f[1]) for f in self.spec["fields"]}
            for f in self.field_names:
                v = str(scores.get(f, "") or "")
                if v and v not in allowed[f]:
                    raise ValueError(f"{f}={v!r} is not one of {sorted(allowed[f])}")
                rec[f"score_{f}"] = v
            for f in self.free_names:
                rec[f"score_{f}"] = str(scores.get(f, "") or "").strip()
            self.answers[row_key] = rec
            self._flush()
            # Append-only trail: keeps every submission, including corrections to a row
            # that was already answered. The CSV holds the current answer; this holds how
            # it got there.
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            return rec

    def progress(self) -> dict:
        done = sum(1 for k, a in self.answers.items()
                   if all(str(a.get(f"score_{f}", "")) for f in self.field_names))
        return {"n_rows": len(self.wb), "n_done": done, "n_left": len(self.wb) - done}


# =============================================================================
# Folding the per-reviewer files back into the workbooks
# =============================================================================
def merge_scores(cfg, mode: str, reviewers: list[str] | None, log) -> int:
    """Write each reviewer's answers into the workbook columns the scorer reads.

    Image mode targets `<item>_r<k>` / `notes_r<k>`, which `crop_qa.score_image_audit`
    already understands. The other two workbooks were written with a SINGLE set of
    `reviewer_*` columns, so reviewer 1 fills those (protocol sections 7 and 23 (ii) ask
    for a review, not for a second independent one) and any further reviewer is appended
    as `<field>_r<k>` rather than silently overwriting.
    """
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    spec = MODES[mode]
    scores_dir = coh / SCORES_DIR
    found = sorted(scores_dir.glob(f"{mode}_audit_scores_*.csv"))
    by_rev = {p.stem.replace(f"{mode}_audit_scores_", ""): p for p in found}
    if reviewers:
        missing = [r for r in reviewers if r not in by_rev]
        if missing:
            log.error("no score file for reviewer(s) %s in %s", missing, scores_dir)
            return 2
        order = list(reviewers)
    else:
        order = sorted(by_rev)
    if not order:
        log.error("no score files found in %s — nobody has reviewed anything yet", scores_dir)
        return 2

    wb_path = coh / spec["workbook"]
    wb = pd.read_csv(wb_path, dtype=str).fillna("")
    rk = spec["row_key"]
    fields = ([(it, None, None) for it in list(cfg["crop_qa"]["score_items"])]
              if spec["fields"] is None else spec["fields"])
    names = [f[0] for f in fields] + [f[0] for f in spec.get("free_fields", [])]

    for k, rev in enumerate(order, start=1):
        sc = pd.read_csv(by_rev[rev], dtype=str).fillna("")
        sc = sc.drop_duplicates("row_key", keep="last").set_index("row_key")
        for name in names:
            src = sc[f"score_{name}"] if f"score_{name}" in sc.columns else None
            if src is None:
                continue
            if mode == "image":
                target = f"{name}_r{k}"
            else:
                target = name if k == 1 else f"{name}_r{k}"
            wb[target] = wb[rk].astype(str).map(src).fillna("")
        notes_col = "notes_r%d" % k if mode == "image" else (
            "reviewer_notes" if k == 1 else f"reviewer_notes_r{k}")
        wb[notes_col] = wb[rk].astype(str).map(sc["note"]).fillna("") if "note" in sc.columns else ""
        n = int((wb[rk].astype(str).isin(sc.index)).sum())
        log.info("reviewer %s -> r%d: %d of %d rows merged into %s", rev, k, n, len(wb),
                 wb_path.name)

    wb.to_csv(wb_path, index=False)
    log.info("merged %d reviewer file(s) into %s", len(order), wb_path)
    if mode == "image":
        log.info("now run: python3 -m src.crop_qa --score %s", wb_path)
    return 0


# =============================================================================
# The page. One file, no external anything — a strict offline requirement.
# =============================================================================
def render_page(state: dict) -> str:
    return _PAGE.replace("__STATE__", json.dumps(state))


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MRKR QA review</title>
<style>
:root{--bg:#14161a;--fg:#e8e8ea;--dim:#9aa0a8;--line:#2b2f36;--ok:#2f9e5e;--err:#c0392b;
      --na:#8a6d1f;--sel:#3b82f6;--card:#1b1e24}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{display:flex;gap:14px;align-items:baseline;padding:8px 14px;border-bottom:1px solid var(--line);
       position:sticky;top:0;background:var(--bg);z-index:5;flex-wrap:wrap}
header b{font-size:15px}
.dim{color:var(--dim)}
.bar{height:5px;background:var(--line);border-radius:3px;flex:1;min-width:120px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--sel)}
main{display:flex;gap:14px;padding:14px;align-items:flex-start;flex-wrap:wrap}
#panelwrap{flex:1 1 520px;min-width:320px}
#panel{width:100%;background:#000;border:1px solid var(--line);border-radius:6px;display:block}
#nopanel{padding:28px;border:1px dashed var(--err);border-radius:6px;color:var(--err);text-align:center}
#side{flex:1 1 420px;min-width:340px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin-bottom:10px}
.item{display:flex;gap:10px;align-items:flex-start;padding:7px 8px;border-radius:5px;border:1px solid transparent}
.item.focus{border-color:var(--sel);background:#1f2530}
.item .n{width:18px;color:var(--dim);font-variant-numeric:tabular-nums}
.item .t{flex:1}
.item .t small{color:var(--dim);display:block;margin-top:2px;line-height:1.35}
.btns{display:flex;gap:4px;flex-shrink:0}
button{font:12px inherit;background:#262b33;color:var(--fg);border:1px solid var(--line);
       border-radius:4px;padding:4px 8px;cursor:pointer}
button:hover{border-color:var(--sel)}
button.on-OK{background:var(--ok);border-color:var(--ok);color:#fff}
button.on-ERROR{background:var(--err);border-color:var(--err);color:#fff}
button.on-NOT_ASSESSABLE{background:var(--na);border-color:var(--na);color:#fff}
button.on-L,button.on-R,button.on-Y,button.on-N,button.on-CANNOT_TELL,button.on-UNCLEAR,
button.on-NONE{background:var(--sel);border-color:var(--sel);color:#fff}
input,textarea{width:100%;background:#0f1115;color:var(--fg);border:1px solid var(--line);
               border-radius:4px;padding:6px;font:13px inherit}
kbd{background:#262b33;border:1px solid var(--line);border-radius:3px;padding:0 5px;font-size:11px}
table{border-collapse:collapse;font-size:12px;width:100%}
td{padding:1px 8px 1px 0;vertical-align:top}
td:first-child{color:var(--dim);white-space:nowrap}
#chrono{font-family:ui-monospace,Menlo,monospace;font-size:11px;white-space:pre-wrap;max-height:220px;overflow:auto}
#help{position:fixed;inset:0;background:rgba(0,0,0,.86);display:none;padding:40px;overflow:auto;z-index:9}
#help.on{display:block}
#toast{position:fixed;right:14px;bottom:14px;background:var(--ok);color:#fff;padding:8px 12px;
       border-radius:5px;opacity:0;transition:opacity .2s;z-index:8}
#toast.on{opacity:1}
#toast.warn{background:var(--err)}
.warn{color:var(--err);font-weight:600}
</style></head><body>
<header>
  <b id="title"></b>
  <span class="dim">reviewer <b id="rev"></b></span>
  <span class="dim" id="pos"></span>
  <span class="bar"><i id="prog"></i></span>
  <span class="dim" id="done"></span>
  <button onclick="help()">? keys</button>
</header>
<main>
  <div id="panelwrap"><img id="panel" alt="review panel"><div id="nopanel" style="display:none"></div>
    <div class="card" id="metacard"><table id="meta"></table></div>
    <div class="card" id="chronocard" style="display:none"><div class="dim">CPT chronology</div>
      <div id="chrono"></div></div>
  </div>
  <div id="side">
    <div class="card" id="items"></div>
    <div class="card">
      <div class="dim">note (<kbd>/</kbd> to type, <kbd>Esc</kbd> to leave)</div>
      <textarea id="note" rows="2" placeholder="optional"></textarea>
    </div>
    <div class="card">
      <button onclick="go(-1)">&larr; previous</button>
      <button onclick="allok()">a &nbsp;all OK</button>
      <button onclick="commit()"><b>Return</b> &nbsp;save &amp; next</button>
      <button onclick="go(1)">next &rarr;</button>
      <button onclick="jump()">go to #</button>
      <button onclick="nextUnscored()">next unanswered</button>
    </div>
  </div>
</main>
<div id="help" onclick="help()"></div>
<div id="toast"></div>
<script>
const S = __STATE__;
// NOTE: inline on* handlers cannot assign to a `let` binding (they resolve names against
// the element and document first, and `let` never lands on window). Every mutation an
// inline handler performs therefore goes through a top-level `function`, which does.
let i = S.start, dirty = {}, fidx = 0, t0 = Date.now();
const F = S.fields.map(f => f[0]);
const el = id => document.getElementById(id);

function cur(){ return S.rows[i]; }
function ans(){ const k = cur().row_key; if(!dirty[k]) dirty[k] = Object.assign({}, S.saved[k]||{}); return dirty[k]; }
function esc(s){ return String(s==null?'':s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function draw(){
  const r = cur(), a = ans();
  el('title').textContent = S.title;
  el('rev').textContent = S.reviewer;
  el('pos').textContent = 'item ' + (i+1) + ' of ' + S.rows.length;
  const done = S.rows.filter(x => { const v = dirty[x.row_key]||S.saved[x.row_key];
      return v && F.every(f => v[f]); }).length;
  el('prog').style.width = (100*done/S.rows.length) + '%';
  el('done').textContent = done + ' answered, ' + (S.rows.length-done) + ' left';

  if(S.has_panel){
    if(r.panel_exists){ el('panel').style.display='block'; el('nopanel').style.display='none';
      el('panel').src = '/panel/' + encodeURIComponent(r.panel); }
    else { el('panel').style.display='none'; el('nopanel').style.display='block';
      el('nopanel').textContent = 'No panel image on disk for ' + r.row_key; }
  } else { el('panel').style.display='none'; }

  let m = '';
  for(const k in r.meta){ if(k==='cpt_chronology') continue; m += '<tr><td>'+esc(k)+'</td><td>'+esc(r.meta[k])+'</td></tr>'; }
  m += '<tr><td>empi_anon</td><td>'+esc(r.empi_anon)+'</td></tr>';
  el('meta').innerHTML = m;
  if(r.meta.cpt_chronology){ el('chronocard').style.display='block';
    el('chrono').textContent = String(r.meta.cpt_chronology).split(' ; ').join('\n'); }
  else el('chronocard').style.display='none';

  const noFilm = S.mode==='image' && String(r.meta.panel_has_full_film).toLowerCase()!=='true';
  let h = '';
  S.fields.forEach((f, n) => {
    const [name, vals, label] = f, v = a[name] || '';
    const help = S.help[name] || '';
    const flag = (noFilm && name==='laterality')
      ? '<span class="warn">No original film for this image — this item is NOT ASSESSABLE (press u).</span> ' : '';
    h += '<div class="item '+(n===fidx?'focus':'')+'" onclick="setFocus('+n+')">'
       + '<div class="n">'+(n+1)+'</div><div class="t"><b>'+esc(label)+'</b>'
       + (help? '<small>'+flag+esc(help)+'</small>' : (flag?'<small>'+flag+'</small>':''))
       + '</div><div class="btns">'
       + vals.map(x => '<button class="'+(v===x?'on-'+x:'')+'" onclick="event.stopPropagation();set('+n+',\''+x+'\')">'
           + esc(S.vkey[x] ? S.vkey[x]+' '+x : x) + '</button>').join('')
       + '</div></div>';
  });
  (S.free_fields||[]).forEach(f => {
    h += '<div class="item"><div class="n"></div><div class="t"><b>'+esc(f[1])+'</b>'
       + '<input value="'+esc(a[f[0]]||'')+'" oninput="ans()[\''+f[0]+'\']=this.value"></div></div>';
  });
  el('items').innerHTML = h;
  el('note').value = a.__note || (S.saved[r.row_key]||{}).__note || '';
  t0 = Date.now();
}

function setFocus(n){ fidx = n; draw(); }
function set(n, v){ const f = S.fields[n][0], a = ans();
  a[f] = (a[f]===v ? '' : v);
  if(a[f]) fidx = Math.min(n+1, S.fields.length-1);
  draw(); }
function allok(){ const a = ans();
  S.fields.forEach(f => { if(!a[f[0]] && f[1].indexOf('OK')>=0) a[f[0]]='OK'; }); draw(); }
function go(d){ save(false); i = Math.max(0, Math.min(S.rows.length-1, i+d)); fidx = 0; draw(); }
function jump(){ const n = prompt('go to item number (1-'+S.rows.length+')');
  if(n){ save(false); i = Math.max(0, Math.min(S.rows.length-1, parseInt(n,10)-1)); fidx=0; draw(); } }
function nextUnscored(){ save(false);
  for(let k=1;k<=S.rows.length;k++){ const j=(i+k)%S.rows.length, v=dirty[S.rows[j].row_key]||S.saved[S.rows[j].row_key];
    if(!(v && F.every(f=>v[f]))){ i=j; fidx=0; draw(); return; } }
  toast('every item has been answered', true); }
function commit(){ const a = ans(), miss = F.filter(f => !a[f]);
  if(miss.length){ toast('still unanswered: ' + miss.join(', '), true); return; }
  save(true, () => { if(i < S.rows.length-1){ i++; fidx=0; draw(); } else toast('that was the last item'); }); }

function save(force, then){
  const r = cur(), a = dirty[r.row_key];
  if(!a || (!force && !F.some(f=>a[f]) && !el('note').value)) { if(then) then(); return; }
  a.__note = el('note').value;
  const body = {row_key: r.row_key, note: a.__note, seconds: (Date.now()-t0)/1000, scores: {}};
  F.forEach(f => body.scores[f] = a[f]||'');
  (S.free_fields||[]).forEach(f => body.scores[f[0]] = a[f[0]]||'');
  fetch('/api/score', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(body)})
    .then(x => x.json())
    .then(x => { if(x.ok){ S.saved[r.row_key] = Object.assign({}, a); toast('saved'); }
                 else toast(x.error||'save failed', true);
                 if(then) then(); })
    .catch(e => { toast('SAVE FAILED — is the terminal window still running? ' + e, true); });
}

let tt; function toast(m, bad){ const t = el('toast'); t.textContent = m;
  t.className = 'on' + (bad?' warn':''); clearTimeout(tt); tt = setTimeout(()=>t.className='', bad?4000:900); }

function help(){ const h = el('help');
  h.innerHTML = '<h2>Keys</h2><table>' + S.keys.map(k=>'<tr><td><kbd>'+esc(k[0])+'</kbd></td><td>'
    + esc(k[1]) + '</td></tr>').join('') + '</table><p class="dim">click anywhere to close</p>';
  h.classList.toggle('on'); }

document.addEventListener('keydown', e => {
  if(e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT'){
    if(e.key === 'Escape') e.target.blur(); return; }
  const k = e.key;
  if(k === '/'){ e.preventDefault(); el('note').focus(); return; }
  if(k === '?'){ help(); return; }
  if(k === 'Enter'){ e.preventDefault(); commit(); return; }
  if(k === 'ArrowLeft'){ go(-1); return; }
  if(k === 'ArrowRight'){ go(1); return; }
  if(k === 'a'){ allok(); return; }
  if(k >= '1' && k <= String(S.fields.length)){ setFocus(parseInt(k,10)-1); return; }
  const v = S.keyv[k];
  if(v !== undefined){ const vals = S.fields[fidx][1];
    if(vals.indexOf(v) >= 0) set(fidx, v); else toast('"'+k+'" is not an option for this item', true); }
});
window.addEventListener('beforeunload', () => save(false));
draw();
</script></body></html>
"""


# =============================================================================
# Server
# =============================================================================
class Handler(BaseHTTPRequestHandler):
    review: ReviewList                      # set on the server instance
    log_obj = None

    def log_message(self, fmt, *a):         # keep the terminal readable for a reviewer
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                        # noqa: N802
        rv = self.server.review                              # type: ignore[attr-defined]
        if self.path in ("/", "/index.html"):
            self._send(200, render_page(build_state(rv)).encode(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/panel/"):
            from urllib.parse import unquote
            name = unquote(self.path[len("/panel/"):].split("?")[0])
            base = rv.panel_dir.resolve()
            p = (base / name).resolve()
            # Never serve anything outside the panel directory, whatever the URL says.
            if not str(p).startswith(str(base) + os.sep) or not p.is_file():
                self._send(404, b"no such panel", "text/plain")
                return
            self._send(200, p.read_bytes(), "image/png")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):                                       # noqa: N802
        rv = self.server.review                              # type: ignore[attr-defined]
        if self.path != "/api/score":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            rec = rv.save(str(payload.get("row_key", "")), payload.get("scores") or {},
                          payload.get("note", ""), payload.get("seconds", 0.0))
            out = {"ok": True, "row_key": rec["row_key"], "scored_utc": rec["scored_utc"]}
            code = 200
        except Exception as exc:
            out, code = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400
            if self.server.log_obj is not None:               # type: ignore[attr-defined]
                self.server.log_obj.error("save failed: %s", exc)   # type: ignore[attr-defined]
        self._send(code, json.dumps(out).encode(), "application/json")


def build_state(rv: ReviewList) -> dict:
    rows = rv.rows_meta()
    saved = {k: {**{f: a.get(f"score_{f}", "") for f in rv.field_names + rv.free_names},
                 "__note": a.get("note", "")}
             for k, a in rv.answers.items()}
    start = next((r["i"] for r in rows
                  if not all(saved.get(r["row_key"], {}).get(f) for f in rv.field_names)), 0)
    values: list[str] = []
    for f in rv.spec["fields"]:
        for v in f[1]:
            if v not in values:
                values.append(v)
    vkey = {v: VALUE_KEYS[v] for v in values if v in VALUE_KEYS}
    keys = [(k, f"mark the highlighted item {v.replace('_', ' ')}") for v, k in vkey.items()] + [
        ("1-%d" % len(rv.field_names), "jump to that item"),
        ("a", "mark every unanswered item OK") if rv.mode == "image" else
        ("a", "(image mode only)"),
        ("Return", "save this item and go to the next"),
        ("left / right arrow", "previous / next item (answers are kept)"),
        ("/", "type a note"), ("Esc", "leave the note box"), ("?", "show or hide this list")]
    return {
        "mode": rv.mode, "title": rv.spec["title"], "reviewer": rv.reviewer,
        "rows": rows, "saved": saved, "start": start,
        "fields": [[f[0], list(f[1]), f[2]] for f in rv.spec["fields"]],
        "free_fields": [[f[0], f[1]] for f in rv.spec.get("free_fields", [])],
        "help": {k: v[1] for k, v in ITEM_TEXT.items()} if rv.mode == "image" else {},
        "has_panel": bool(rv.spec["has_panel"]),
        "vkey": vkey,
        "keyv": {k: v for v, k in vkey.items()},
        "keys": keys,
    }


def free_port(preferred: int) -> int:
    for p in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port near %d" % preferred)


def serve(rv: ReviewList, port: int, open_browser: bool, log) -> int:
    port = free_port(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.review = rv                                        # type: ignore[attr-defined]
    httpd.log_obj = log                                      # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{port}/"
    p = rv.progress()
    log.info("%s", rv.spec["title"])
    log.info("reviewer '%s' — %d of %d already answered, %d to go", rv.reviewer,
             p["n_done"], p["n_rows"], p["n_left"])
    log.info("answers are written to %s", rv.csv_path)
    log.info("")
    log.info("    OPEN THIS IN YOUR BROWSER:  %s", url)
    log.info("")
    log.info("Press Ctrl+C here when you are done. Nothing is lost — every answer is already "
             "on disk.")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        p = rv.progress()
        log.info("stopped. %d of %d answered; resume with the same command.",
                 p["n_done"], p["n_rows"])
    finally:
        httpd.server_close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Local, offline scoring interface for the protocol section 23 / 7 audits.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--mode", default="image", choices=sorted(MODES),
                    help="which review list to score")
    ap.add_argument("--reviewer", default=None,
                    help="your short name; your answers go to a file named after it so two "
                         "reviewers never overwrite each other")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--status", action="store_true",
                    help="print how far every reviewer has got, and exit")
    ap.add_argument("--merge", action="store_true",
                    help="fold the per-reviewer score files back into the workbook, then exit")
    ap.add_argument("--reviewers", default=None,
                    help="with --merge, comma-separated reviewer order (r1,r2,...). "
                         "Default: alphabetical.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))

    if args.merge:
        revs = [safe_reviewer(r) for r in args.reviewers.split(",")] if args.reviewers else None
        return merge_scores(cfg, args.mode, revs, log)

    if args.status:
        coh = cfg.path(cfg["paths"]["cohort_dir"])
        any_found = False
        for mode in sorted(MODES):
            for p in sorted((coh / SCORES_DIR).glob(f"{mode}_audit_scores_*.csv")):
                rev = p.stem.replace(f"{mode}_audit_scores_", "")
                try:
                    rv = ReviewList(cfg, mode, rev)
                except FileNotFoundError as exc:
                    log.warning("%s", exc)
                    continue
                pr = rv.progress()
                log.info("%-11s %-14s %d/%d answered (%d left)", mode, rev, pr["n_done"],
                         pr["n_rows"], pr["n_left"])
                any_found = True
        if not any_found:
            log.info("no score files yet under %s", coh / SCORES_DIR)
        return 0

    reviewer = safe_reviewer(args.reviewer or "")
    if not reviewer:
        log.error("--reviewer is required. Use your own short name, e.g. "
                  "`python3 -m src.qa_review_app --mode %s --reviewer jdoe`. Your answers are "
                  "saved in a file named after it, which is what keeps the two reviews "
                  "independent.", args.mode)
        return 2
    try:
        rv = ReviewList(cfg, args.mode, reviewer)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    return serve(rv, args.port, not args.no_browser, log)


if __name__ == "__main__":
    raise SystemExit(main())
