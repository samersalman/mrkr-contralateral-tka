"""make_globus_batch.py — turn the image-transfer manifest into a Globus CLI batch file.

Globus (NOT a browser) is the bulk-transfer mechanism for the Nightingale MRKR DICOMs.
This reads the de-duplicated selected-image paths (outputs/tables/image_transfer_manifest_paths.txt,
one relative dicom_path per line) and writes a Globus `--batch` file with one
"<source_path> <dest_path>" line per image, so `globus transfer` moves EXACTLY the
final-cohort images (not the whole dataset) in a single resumable job.

Quoting contract (verified against globus-cli 3.42.0)
-----------------------------------------------------
`globus transfer --batch` parses every line with `shlex.split(line, comments=True)`
(globus_cli/utils.py:197, called from globus_cli/services/transfer/data.py:56) and then
requires EXACTLY two arguments. Both paths are therefore emitted through `shlex.quote`,
which makes a space-bearing `--dest-root` safe — notably the real Google Drive
destination, whose path contains two space-bearing components ("My Drive" and
"Radiographic Prediction of Contralateral Knee Arthroplasty") and which unquoted would
split into 8 tokens and be rejected. Quoting also stops `comments=True` from truncating
a path that contains a `#`. `shlex.quote(p) == p` for any path without shell
metacharacters, so space-free roots (and the `--dest-root .` form) are byte-identical to
the unquoted output.

Usage:
    python3 -m src.make_globus_batch \
        --source-root  <MRKR data root on the Nightingale collection, e.g. /mrkr/dicoms> \
        --dest-root    <destination folder on your endpoint, e.g. /MRKR_contra_tka/dicoms>

`--dest-root` is interpreted in the DESTINATION ENDPOINT's own path space, not as a macOS
path; see outputs/globus_transfer_runbook.md step 4. A dest path that starts with `/` is
absolute and makes `globus transfer` ignore the command-line DEST_PATH prefix, so use
`--dest-root .` when the base path is supplied on the command line instead.

Then (fill in the endpoint UUIDs from `globus endpoint search` / your Nightingale access docs):
    globus login
    globus transfer <SOURCE_ENDPOINT_UUID> <DEST_ENDPOINT_UUID> \
        --batch derived-data/cohort/globus_batch.txt \
        --label "MRKR contralateral TKA" --sync-level checksum --preserve-mtime

The output batch file lists DICOM paths and is git-ignored; do not commit it.
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from src.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a Globus batch-transfer file from the manifest.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--source-root", required=True,
                    help="MRKR DICOM root on the Nightingale/Globus source collection")
    ap.add_argument("--dest-root", required=True,
                    help="destination folder on your Globus endpoint (local or Drive collection)")
    ap.add_argument("--paths", default=None, help="override the manifest paths file")
    ap.add_argument("--out", default=None,
                    help="override the output batch file path (default: transfer.batch_file)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tcfg = cfg["transfer"]
    # Same config keys src/verify_transfer.py reads, so the batch that is generated and the
    # transfer that is verified can never drift onto different manifests.
    paths_file = Path(args.paths) if args.paths else cfg.path(tcfg["manifest_paths"])
    out_file = Path(args.out) if args.out else cfg.path(tcfg["batch_file"])

    if not paths_file.exists():
        raise SystemExit(f"paths file not found: {paths_file} (run src.manifest first)")

    src_root = args.source_root.rstrip("/")
    dst_root = args.dest_root.rstrip("/")
    seen, lines = set(), []
    for raw in paths_file.read_text().splitlines():
        rel = raw.strip().lstrip("/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        # Globus batch line: "<source path> <dest path>", each shlex-quoted so the CLI's
        # shlex.split() yields exactly two tokens even when a root contains spaces.
        lines.append(f"{shlex.quote(f'{src_root}/{rel}')} {shlex.quote(f'{dst_root}/{rel}')}")

    # Self-check with the CLI's own parser before the file is written: every line must
    # shlex-split into exactly (source, dest) and round-trip to the intended paths.
    for i, line in enumerate(lines):
        argv_line = shlex.split(line, comments=True)
        assert len(argv_line) == 2, (
            f"batch line {i} splits into {len(argv_line)} tokens, not 2: {line!r}")
        assert argv_line[0].startswith(f"{src_root}/") and argv_line[1].startswith(f"{dst_root}/"), \
            f"batch line {i} does not round-trip to the configured roots: {argv_line!r}"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_file} with {len(lines):,} transfer lines "
          f"(source_root={src_root}, dest_root={dst_root})")
    print(f"shlex check: all {len(lines):,} lines split into exactly 2 tokens "
          "(globus-cli parses --batch with shlex.split)")
    print("next: globus login  ->  globus transfer <SRC_UUID> <DST_UUID> "
          f"--batch {out_file} --label 'MRKR contralateral TKA' --sync-level checksum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
