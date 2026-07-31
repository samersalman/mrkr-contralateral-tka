"""verify_transfer.py — gate the bulk Globus DICOM transfer before preprocessing.

Track B verification: the Nightingale -> Google Drive transfer is correct only if every
image on the locked manifest arrived, intact, and nothing else did. This walks the
destination root and checks (a) the *.dcm count equals transfer.expected_n_files, (b) every
manifest relative path exists, (c) no file is smaller than transfer.min_file_bytes (catches
truncation and un-materialized Drive placeholders), (d) no unexpected extra DICOMs. It then
opens a seeded random sample with pydicom and confirms the files parse and carry decodable
pixel data — the only check that forces Drive to hand over real bytes.

Exit code is 0 only when every check passes, so the preprocessing step can be gated on it.
The report is AGGREGATE ONLY: counts, sizes, and at most a handful of truncated relative
paths. No patient identifiers, no full path lists.

The destination root is resolved as ``--dest-root`` > ``$MRKR_DRIVE_ROOT`` >
``transfer.dest_root`` from the config, so a second user can point the same command at
their own Drive mount without editing a tracked file (protocol section 28: the analysis
repository is public).

Run:  python3 -m src.verify_transfer --config config/feasibility.yaml
      MRKR_DRIVE_ROOT="$HOME/.../DICOMs-knee-imaging" python3 -m src.verify_transfer
      python3 -m src.verify_transfer --dest-root /some/other/root --sample-dicom 0
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config

MODULE = "verify_transfer"
MAX_PATHS_SHOWN = 5          # data hygiene: never dump the manifest into a tracked report
IGNORED_NAMES = {".DS_Store", "Icon\r", ".localized"}


def display_root(p: Path) -> str:
    """Publication-safe rendering of a filesystem root (protocol section 28).

    ``outputs/transfer_verification.md`` is tracked and publishable, so it must not
    reproduce the operator's home directory or Google account: a path under ``$HOME`` is
    shown as ``~/...`` and a ``GoogleDrive-<account>`` component is masked. The unmasked
    path is still written to ``outputs/logs/run.log``, which is git-ignored, so nothing is
    lost for local debugging.
    """
    s = str(p)
    home = str(Path.home())
    if s == home or s.startswith(home + os.sep):
        s = "~" + s[len(home):]
    return re.sub(r"(GoogleDrive-)[^/]+", r"\1<your-account>", s)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO); lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a"); fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
                                          datefmt="%Y-%m-%dT%H:%M:%S")); lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout); sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s")); lg.addHandler(sh)
    return lg


def read_manifest(paths_file: Path) -> list[str]:
    """De-duplicated, normalized relative paths, in manifest order."""
    seen, out = set(), []
    for raw in paths_file.read_text().splitlines():
        rel = raw.strip().lstrip("/")
        if rel and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def scan_dest(dest_root: Path, suffix: str) -> tuple[dict[str, int], list[str]]:
    """Walk dest_root once. Returns {relative path -> size} for files matching the DICOM
    suffix, plus the relative paths of other (non-DICOM) files found."""
    dicoms: dict[str, int] = {}
    others: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(dest_root):
        for name in filenames:
            if name in IGNORED_NAMES:
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(dest_root).as_posix()
            if name.lower().endswith(suffix.lower()):
                try:
                    dicoms[rel] = full.stat().st_size
                except OSError:
                    dicoms[rel] = -1          # unreadable -> caught by the min-size check
            else:
                others.append(rel)
    return dicoms, others


def sample_dicoms(dest_root: Path, rels: list[str], n: int, seed: int) -> dict:
    """Open n seeded-random transferred files and confirm they parse with real pixel data."""
    import pydicom

    picks = sorted(rels)
    random.Random(seed).shuffle(picks)
    picks = picks[:n]
    res = {"n": len(picks), "ok": 0, "failures": [], "warnings": [], "examples": []}
    for rel in picks:
        try:
            ds = pydicom.dcmread(dest_root / rel)
            if "PixelData" not in ds:
                res["failures"].append((rel, "no PixelData element"))
                continue
            compressed = bool(getattr(ds.file_meta.TransferSyntaxUID, "is_compressed", False))
            try:
                arr = ds.pixel_array
                shape = tuple(arr.shape)
            except Exception as exc:                     # noqa: BLE001 - report, don't crash
                if compressed and len(ds.PixelData) > 0:
                    # a missing JPEG codec plugin is an environment gap, not a bad transfer
                    res["warnings"].append((rel, f"encapsulated pixel data not decodable here: {exc}"))
                    res["ok"] += 1
                    continue
                res["failures"].append((rel, f"pixel decode failed: {exc}"))
                continue
            res["ok"] += 1
            if len(res["examples"]) < 3:
                res["examples"].append(dict(
                    shape=shape,
                    photometric=str(getattr(ds, "PhotometricInterpretation", "?")),
                    rows=int(getattr(ds, "Rows", 0)), cols=int(getattr(ds, "Columns", 0)),
                    syntax=str(ds.file_meta.TransferSyntaxUID.name)))
        except Exception as exc:                         # noqa: BLE001
            res["failures"].append((rel, f"dcmread failed: {exc}"))
    return res


def fmt_bytes(n: int) -> str:
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{n:,} B"


def write_not_started(report: Path, dest_root: Path, expected: int, reason: str) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Transfer Verification — Contralateral TKA DICOMs\n\n"
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python3 -m src.verify_transfer`.\n\n"
        f"**VERDICT: FAIL — 0 of {expected:,} present, transfer not started.**\n\n"
        f"- Destination root: `{display_root(dest_root)}`\n"
        f"- Reason: {reason}\n\n"
        "No further checks were run. Follow `outputs/globus_transfer_runbook.md` to start the "
        "Globus transfer, then re-run `python3 -m src.verify_transfer`.\n")


def build_report(report: Path, dest_root: Path, cfg_expected: int, manifest_n: int,
                 found_n: int, present: list[str], missing: list[str], extra_dcm: list[str],
                 other_files: list[str], too_small: list[tuple[str, int]], sizes: list[int],
                 min_bytes: int, sample: dict | None, checks: list[tuple[str, bool, str]],
                 passed: bool) -> None:
    lines = ["# Transfer Verification — Contralateral TKA DICOMs", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python3 -m src.verify_transfer`.", "",
             f"**VERDICT: {'PASS' if passed else 'FAIL'} — {len(present):,} of {manifest_n:,} "
             f"manifest images present ({100 * len(present) / manifest_n:.2f}%).**", "",
             "## Destination", "",
             f"- Root: `{display_root(dest_root)}`",
             f"- Expected images (config `transfer.expected_n_files`): {cfg_expected:,}",
             f"- Manifest entries (de-duplicated): {manifest_n:,}",
             f"- DICOM files found under the root: {found_n:,}",
             f"- Non-DICOM files found: {len(other_files):,}", ""]

    if sizes:
        lines += ["## Size", "",
                  f"- Total transferred: {fmt_bytes(sum(sizes))}",
                  f"- Median file: {fmt_bytes(int(statistics.median(sizes)))}",
                  f"- Smallest / largest file: {fmt_bytes(min(sizes))} / {fmt_bytes(max(sizes))}",
                  f"- Minimum acceptable size (`transfer.min_file_bytes`): {min_bytes:,} B", ""]

    lines += ["## Checks", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")

    for label, items in (("Missing from the destination", missing),
                         ("Unexpected extra DICOMs", extra_dcm),
                         ("Below the minimum size", [f"{p} ({s:,} B)" for p, s in too_small]),
                         ("Non-DICOM files present (informational)", other_files)):
        if items:
            lines += [f"### {label} — {len(items):,}", ""]
            lines += [f"- `{p}`" for p in items[:MAX_PATHS_SHOWN]]
            if len(items) > MAX_PATHS_SHOWN:
                lines.append(f"- ... and {len(items) - MAX_PATHS_SHOWN:,} more (not listed; "
                             "relative paths carry the de-identified patient id)")
            lines.append("")

    if sample is not None and sample["n"]:
        lines += ["## DICOM read sample", "",
                  f"- Files opened with pydicom: {sample['n']}",
                  f"- Parsed with pixel data: {sample['ok']}",
                  f"- Failures: {len(sample['failures'])}",
                  f"- Warnings (codec unavailable locally): {len(sample['warnings'])}", ""]
        for ex in sample["examples"]:
            lines.append(f"- example: shape {ex['shape']}, {ex['photometric']}, "
                         f"Rows x Columns {ex['rows']} x {ex['cols']}, {ex['syntax']}")
        if sample["failures"]:
            lines += ["", "Sample failures (first few):", ""]
            lines += [f"- `{p}` — {msg}" for p, msg in sample["failures"][:MAX_PATHS_SHOWN]]
        if sample["warnings"]:
            lines += ["", "Sample warnings (first few):", ""]
            lines += [f"- `{p}` — {msg}" for p, msg in sample["warnings"][:MAX_PATHS_SHOWN]]
        lines.append("")

    lines += ["## Next step", "",
              ("All checks passed. The DICOMs are complete and readable; proceed to the "
               "preprocessing step (`notebooks/preprocess_colab.ipynb` in Colab, or "
               "`python3 -m src.preprocess_images` locally)."
               if passed else
               "Verification failed. See `outputs/globus_transfer_runbook.md` section 9 "
               "(Troubleshooting); re-issuing the same `globus transfer ... --sync-level checksum` "
               "command re-sends only the files that are missing or do not match."), ""]

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the Globus DICOM transfer landed correctly.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--dest-root", default=None, help="override the configured destination root")
    ap.add_argument("--sample-dicom", type=int, default=20,
                    help="open N random transferred files with pydicom (0 disables)")
    ap.add_argument("--report", default=None, help="override the output report path")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tr = cfg["transfer"]
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    seed = int(cfg["reproducibility"]["random_seed"])

    # --dest-root > $MRKR_DRIVE_ROOT > transfer.dest_root (the current default).
    dest_root = Path(args.dest_root) if args.dest_root else cfg.transfer_dest_root()
    report = Path(args.report) if args.report else cfg.path(tr["verify_report_md"])
    paths_file = cfg.path(tr["manifest_paths"])
    expected_n = int(tr["expected_n_files"])
    min_bytes = int(tr["min_file_bytes"])
    suffix = str(tr["dicom_suffix"])

    if not paths_file.exists():
        log.error("manifest paths file not found: %s (run src.manifest first)", paths_file)
        return 2
    manifest = read_manifest(paths_file)
    log.info("manifest: %d unique relative paths (config expects %d)", len(manifest), expected_n)

    # ---- degrade gracefully: no destination, or an empty one ----
    if not dest_root.exists():
        write_not_started(report, dest_root, expected_n, "destination root does not exist")
        log.error("FAIL 0 of %d present - transfer not started (destination root does not exist: %s)",
                  expected_n, dest_root)
        return 1

    dicoms, others = scan_dest(dest_root, suffix)
    if not dicoms:
        write_not_started(report, dest_root, expected_n,
                          f"destination root exists but contains no `*{suffix}` files "
                          f"({len(others)} other files)")
        log.error("FAIL 0 of %d present - transfer not started (no %s files under %s)",
                  expected_n, suffix, dest_root)
        return 1

    manifest_set = set(manifest)
    present = [p for p in manifest if p in dicoms]
    missing = [p for p in manifest if p not in dicoms]
    extra_dcm = sorted(p for p in dicoms if p not in manifest_set)
    sizes = [dicoms[p] for p in present]
    too_small = [(p, dicoms[p]) for p in present if dicoms[p] < min_bytes]

    sample = None
    if args.sample_dicom > 0 and present:
        sample = sample_dicoms(dest_root, present, args.sample_dicom, seed)
        log.info("dicom sample: %d opened, %d parsed with pixel data, %d failed, %d warned",
                 sample["n"], sample["ok"], len(sample["failures"]), len(sample["warnings"]))

    checks: list[tuple[str, bool, str]] = [
        ("Manifest matches config", len(manifest) == expected_n,
         f"{len(manifest):,} manifest paths vs {expected_n:,} expected"),
        ("DICOM file count", len(dicoms) == expected_n,
         f"{len(dicoms):,} `*{suffix}` files found vs {expected_n:,} expected"),
        ("Every manifest path present", not missing, f"{len(missing):,} missing"),
        ("No unexpected extra DICOMs", not extra_dcm, f"{len(extra_dcm):,} extra"),
        ("No file below the minimum size", not too_small,
         f"{len(too_small):,} smaller than {min_bytes:,} B"),
    ]
    if sample is not None:
        checks.append(("DICOM read sample", not sample["failures"],
                       f"{sample['ok']}/{sample['n']} parsed with pixel data, "
                       f"{len(sample['failures'])} failed"))

    passed = all(ok for _, ok, _ in checks)
    build_report(report, dest_root, expected_n, len(manifest), len(dicoms), present, missing,
                 extra_dcm, others, too_small, sizes, min_bytes, sample, checks, passed)

    for name, ok, detail in checks:
        log.info("%-32s %-4s %s", name, "PASS" if ok else "FAIL", detail)
    if others:
        log.info("%d non-DICOM file(s) under the destination root (informational only)", len(others))
    pct = 100 * len(present) / len(manifest)
    verdict = (f"{'PASS' if passed else 'FAIL'} {len(present):,} of {len(manifest):,} manifest "
               f"images present ({pct:.2f}%), {fmt_bytes(sum(sizes))} total; report: {report}")
    log.info(verdict) if passed else log.error(verdict)
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} — {len(present):,}/{len(manifest):,} "
          f"images present ({pct:.2f}%)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
