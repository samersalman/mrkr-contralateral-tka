# Globus Transfer Runbook — Contralateral TKA DICOMs

Operator checklist for moving the locked 6,122-image cohort off Nightingale and into the
Google Drive folder that Colab will read. Everything here is a step **only you can do**
(browser OAuth, endpoint setup, starting the job). The batch-file generation and the
post-transfer verification are already scripted in this repo.

Verified against **globus-cli 3.42.0** (`globus version`) on macOS. Every command below was
checked against that version's `--help`.

---

## Placeholders (fill these in once, reuse everywhere)

| Placeholder | What it is | Where you get it |
| --- | --- | --- |
| `<SRC_UUID>` | Nightingale source collection UUID | **RESOLVED 2026-07-25: `16ff968b-105d-41cb-85a4-8f1cc79e0a88`** ("Nightingale Open Science Datasets", a GCS v5 **guest collection** — use `globus gcs collection show`, not `globus endpoint show`) |
| `<MRKR_ROOT>` | absolute path of the MRKR DICOM root **on the source collection** | **RESOLVED 2026-07-25: `/mrkr-emory-xray/images`** (the collection root also holds `/mrkr-emory-xray/tables/` and five unrelated datasets) |
| `<DST_UUID>` | your Globus Connect Personal collection UUID | `globus endpoint local-id` |
| `<DEST_ROOT>` | destination folder **expressed in the destination endpoint's own path space** | Step 4 (this is the #1 thing people get wrong) |
| `<TASK_ID>` | the transfer task UUID | printed by `globus transfer`, or `globus task list` |
| `<EMPI>` | one de-identified patient id from the manifest | `head -1 outputs/tables/image_transfer_manifest_paths.txt \| cut -d/ -f1` |

**Locked facts for this transfer**

- 6,122 unique DICOMs, 3,709 patients, 1 line per image in
  `outputs/tables/image_transfer_manifest_paths.txt` (git-ignored).
- Relative path layout, preserved end to end:
  `<empi_anon>/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`
- Final destination on this Mac:
  `$MRKR_DRIVE_ROOT`
- **Volume ~36 GB** (95% CI 29-43 GB), measured 2026-07-25 by `globus stat` on a seeded
  30-file random sample of the manifest: mean 5.9 MB, median 5.9 MB, range 1.8-15.4 MB.
  This supersedes the earlier **unmeasured ~90 GB** figure carried over from the plan, and it
  changes the storage decision in step 8 — see the note there. Step 6a still gives the
  definitive number from a real pilot transfer.

---

## Step 0.0 — Set the two shell variables this runbook uses

Every command below refers to `$MRKR_PROJECT` and `$MRKR_DRIVE_ROOT` instead of a hard-coded
home directory or Google account, so this file is safe to publish with the analysis repository
(protocol section 28). Set them once per shell and fill in your own account:

```bash
export MRKR_PROJECT="$HOME/Desktop/Radiographic Prediction of Contralateral Knee Arthroplasty"
export MRKR_DRIVE_ROOT="$HOME/Library/CloudStorage/GoogleDrive-<your-account>@gmail.com/My Drive/Radiographic Prediction of Contralateral Knee Arthroplasty/DICOMs-knee-imaging"

test -d "$MRKR_PROJECT" && test -d "$MRKR_DRIVE_ROOT" && echo "both roots resolve"
```

`MRKR_DRIVE_ROOT` is not decorative: `src/config.py` reads it in `Config.transfer_dest_root()`,
which is what `python3 -m src.verify_transfer` resolves its destination from (after `--dest-root`,
which still wins). Leave it unset and everything falls back to `transfer.dest_root` in
`config/feasibility.yaml`, which is what it has always been.

---

## Step 0 — Preflight (2 minutes, no login needed)

```bash
cd "$MRKR_PROJECT"
globus version                                              # expect 3.42.0
wc -l outputs/tables/image_transfer_manifest_paths.txt      # expect 6122
ls -d "$MRKR_DRIVE_ROOT"
df -h /System/Volumes/Data                                  # local free space
```

`globus --version` is not a flag; the command is `globus version`.

---

## Step 1 — Log in

```bash
globus login
```

Opens a browser, you consent, the CLI captures the tokens. On a remote/SSH shell use the
copy-paste flow instead:

```bash
globus login --no-local-server
```

Confirm:

```bash
globus whoami
globus whoami --linked-identities
globus session show
```

`globus whoami` must print your identity; `globus session show` must list it with a recent
authentication time. If either is empty, you are not logged in. To start clean:
`globus login --force`.

---

## Step 2 — Find the SOURCE (Nightingale collection + MRKR root)

The collection UUID and the DICOM root path come from your **Nightingale data-access
materials** (the access-granted email, the dataset page on nightingalescience.org, or the
README inside your Nightingale compute workspace). Do not guess them.

Search by name, then narrow by scope:

```bash
globus endpoint search "nightingale" --limit 25
globus endpoint search "MRKR" --limit 25
globus endpoint search "Emory knee" --limit 25

# collections that were shared with you, and ones you have already used
globus endpoint search --filter-scope shared-with-me --limit 50
globus endpoint search --filter-scope recently-used --limit 25
```

Inspect the candidate and note its default directory:

```bash
globus endpoint show <SRC_UUID>
```

Walk down to the DICOM root:

```bash
globus ls <SRC_UUID>:/
globus ls <SRC_UUID>:/<some-dir>/
globus ls -l <SRC_UUID>:<MRKR_ROOT>/
```

**Confirm the root is right** by listing one patient folder that is actually on the
manifest. Get a real id without pasting it into any tracked file:

```bash
EMPI=$(head -1 outputs/tables/image_transfer_manifest_paths.txt | cut -d/ -f1)
globus ls <SRC_UUID>:<MRKR_ROOT>/$EMPI/
globus ls -l --recursive --recursive-depth-limit 2 <SRC_UUID>:<MRKR_ROOT>/$EMPI/
```

You have the right `<MRKR_ROOT>` when that last command shows StudyUID directories
containing SeriesUID directories containing `*.dcm` files. If it errors with "path not
found", `<MRKR_ROOT>` is wrong; if it errors with "permission denied", see step 9.

Sanity-check a single full path end to end:

```bash
REL=$(head -1 outputs/tables/image_transfer_manifest_paths.txt)
globus stat <SRC_UUID>:<MRKR_ROOT>/$REL
```

---

## Step 3 — Set up the DESTINATION (Globus Connect Personal on macOS)

1. Download Globus Connect Personal for macOS from
   <https://www.globus.org/globus-connect-personal> and drag it to `/Applications`.
2. Launch it. It walks you through logging in with the same Globus identity from step 1 and
   naming a new collection (for example `samer-macbook`).
3. Open **Globus Connect Personal > Preferences > Access**. This list is the endpoint's
   entire visible filesystem. Decide now which of the two setups in step 4 you want, and
   configure the accessible path accordingly. Give the path **Read/Write** (uncheck
   read-only), leave "Shareable" off.
4. Grant the app **Full Disk Access** (System Settings > Privacy & Security > Full Disk
   Access). Without it, macOS blocks Globus Connect Personal from writing into
   `~/Library/CloudStorage/`, which is where Google Drive lives.
5. Leave the app running for the whole transfer. If it quits, the task goes INACTIVE and
   resumes when it comes back.

Read the UUID:

```bash
globus endpoint local-id
globus endpoint show "$(globus endpoint local-id)"
```

Prove the endpoint can see the destination folder before you transfer anything:

```bash
globus ls "$(globus endpoint local-id)":/
```

---

## Step 4 — THE #1 FAILURE MODE: `--dest-root` is endpoint-relative, not an absolute macOS path

A Globus Connect Personal collection exposes **only the directories you listed in
Preferences > Access**, and paths in a transfer are interpreted **inside that path space**.
Whatever `globus ls "$(globus endpoint local-id)":/` prints *is* the root. Do not reason
about it, list it.

Spaces in the path are **no longer a problem**. `globus transfer --batch` parses each line
with `shlex.split(line, comments=True)` and then expects **exactly two** arguments (source
path, dest path); `src/make_globus_batch.py` now emits both paths through `shlex.quote`, so
a `--dest-root` containing spaces (`My Drive`, `Radiographic Prediction of Contralateral
Knee Arthroplasty`) produces a clean 2-token line instead of the 8-token line that the CLI
used to reject. Space-free roots are unaffected: `shlex.quote` leaves them byte-identical.

What still matters is the **path space** and the **leading slash**.

Verified behaviour of each option (checked against globus-cli 3.42.0's own batch parser):

**Case (a) — GCP configured with `~/` (or `/`) accessible.**
The endpoint root maps to your home directory (or the filesystem root), so express the
Drive folder in *that* path space — with `~/` shared it is
`${MRKR_DRIVE_ROOT#$HOME}`,
with `/` shared it is the full `"$MRKR_DRIVE_ROOT"` path. Either can now be passed
directly, shell-quoted — take the form that matches what
`globus ls "$(globus endpoint local-id)":/` actually printed:

```bash
# GCP Preferences > Access = "/"  (endpoint root == filesystem root)
python3 -m src.make_globus_batch --source-root <MRKR_ROOT> \
  --dest-root "$MRKR_DRIVE_ROOT"

# GCP Preferences > Access = "~/"  (endpoint root == $HOME, so drop that prefix)
python3 -m src.make_globus_batch --source-root <MRKR_ROOT> \
  --dest-root "${MRKR_DRIVE_ROOT#$HOME}"
```

The alternative — batch paths relative, base path on the command line — also still works,
and is what you want if you would rather keep the long path out of the batch file:

```bash
python3 -m src.make_globus_batch --source-root <MRKR_ROOT> --dest-root .
# then transfer with the base path as a command-line prefix (step 6):
globus transfer <SRC_UUID> "<DST_UUID>:$MRKR_DRIVE_ROOT" \
  --batch derived-data/cohort/globus_batch.txt ...
```

`--dest-root .` makes every batch dest path relative, and the CLI joins it onto the
command-line `DEST_PATH`. **A dest path that starts with `/` is treated as absolute and the
command-line `DEST_PATH` prefix is ignored**, so `--dest-root .` is still required whenever
you supply the base path on the command line. Pick one mechanism, never both.

**Case (b) — GCP configured with the project Drive folder as the accessible root**
(Preferences > Access = `$(dirname "$MRKR_DRIVE_ROOT")`).
Then `globus ls "$(globus endpoint local-id)":/` prints `DICOMs-knee-imaging/`, and:

```bash
--dest-root /DICOMs-knee-imaging
```

Batch lines are two clean tokens, no spaces anywhere, no command-line prefix needed.
**This is still the recommended setup** — the shortest paths, the smallest batch file, and
nothing to get wrong. If instead you share the `DICOMs-knee-imaging` folder itself as the
accessible root, use `--dest-root /` and files land at
`/<empi>/<study>/<series>/<sop>.dcm`.

Decision rule: make `--dest-root` a path that **exists in the endpoint's path space** (list
it, do not reason about it), and use `--dest-root .` if and only if you are passing the base
path as a command-line `DEST_PATH`. Spaces are fine either way.

---

## Step 5 — Generate the batch file

```bash
cd "$MRKR_PROJECT"
python3 -m src.make_globus_batch --source-root <MRKR_ROOT> --dest-root <DEST_ROOT>
```

Expected stdout:

```text
wrote .../derived-data/cohort/globus_batch.txt with 6,122 transfer lines (source_root=..., dest_root=...)
shlex check: all 6,122 lines split into exactly 2 tokens (globus-cli parses --batch with shlex.split)
```

The module runs that token check on every line before it writes the file, so a bad
`--dest-root` fails loudly here rather than at `globus transfer`.

Sanity checks (`globus_batch.txt` is git-ignored; it holds DICOM paths, keep it local):

```bash
wc -l derived-data/cohort/globus_batch.txt        # must be 6122
head -2 derived-data/cohort/globus_batch.txt
python3 -c "import shlex,sys; bad=[i for i,l in enumerate(open(sys.argv[1])) if len(shlex.split(l, comments=True)) != 2]; print('OK: all lines split into exactly 2 tokens' if not bad else f'BAD lines: {bad[:5]}'); sys.exit(1 if bad else 0)" derived-data/cohort/globus_batch.txt
```

That command parses the file exactly the way globus-cli does and exits non-zero if any line
is wrong, so it can gate a script. Use it instead of `awk '{print NF}'`: awk counts
whitespace-separated fields, and a correctly quoted space-bearing destination path is one
shlex token but several awk fields, so awk would report a false failure.

`head -2` must show two lines, each holding a source path and a destination path:
`<MRKR_ROOT>/<empi>/<study>/<series>/<sop>.dcm <DEST_ROOT>/<empi>/<study>/<series>/<sop>.dcm`.
A path containing spaces appears wrapped in single quotes — that is correct and is what
makes it a single token.

---

## Step 6 — Trigger the transfer

### 6a. Pilot first (one patient, ~1 minute)

Proves permissions, path space, and Drive writes before you commit ~90 GB, and gives you a
real per-image size to extrapolate from.

```bash
EMPI=$(head -1 outputs/tables/image_transfer_manifest_paths.txt | cut -d/ -f1)
grep "/$EMPI/" derived-data/cohort/globus_batch.txt > derived-data/cohort/globus_batch_pilot.txt
wc -l derived-data/cohort/globus_batch_pilot.txt

globus transfer <SRC_UUID> <DST_UUID> --batch derived-data/cohort/globus_batch_pilot.txt \
  --label "MRKR pilot" --sync-level checksum --preserve-mtime --dry-run
```

`--dry-run` prints the submission document without submitting. If it looks right, rerun
without `--dry-run`, then confirm the files landed:

```bash
DEST="$MRKR_DRIVE_ROOT"
find "$DEST" -name '*.dcm' | wc -l
find "$DEST" -name '*.dcm' -exec ls -lh {} +
du -sh "$DEST"
```

Divide that total by the pilot file count and multiply by 6,122 to get the real projected
volume before step 6b. (`globus_batch_pilot.txt` lives in the git-ignored `derived-data/`
tree; delete it when you are done.)

### 6b. Full transfer

```bash
globus transfer <SRC_UUID> <DST_UUID> \
  --batch derived-data/cohort/globus_batch.txt \
  --label "MRKR contralateral TKA" \
  --sync-level checksum \
  --preserve-mtime
```

Case (a) variant, with the space-containing base path quoted on the command line:

```bash
globus transfer <SRC_UUID> "<DST_UUID>:$MRKR_DRIVE_ROOT" \
  --batch derived-data/cohort/globus_batch.txt \
  --label "MRKR contralateral TKA" \
  --sync-level checksum \
  --preserve-mtime
```

Notes on the flags:

- `--sync-level checksum` compares file contents, so re-running skips anything already
  correct. It is also the **only** sync level that is safe for restarting a failed
  transfer; the others can leave corrupted files in place.
- `--verify-checksum` is on by default: Globus checksums every file after writing it. That
  is the "Globus checksum-verified" half of the Track B acceptance criterion.
- `--preserve-mtime` keeps source modification times.
- Missing parent directories on the destination are created automatically.
- Optional: `--notify succeeded,failed` to get email on completion, `--encrypt-data` if
  your data-use agreement requires encryption in flight.

### 6c. Monitor

```bash
globus task list --limit 5
globus task show <TASK_ID>
globus task show <TASK_ID> --successful-transfers | tail
globus task wait <TASK_ID> --heartbeat --polling-interval 60 --timeout 86400
```

`globus task wait` exits 0 on success, 1 otherwise, so it can gate a shell script. Watch
`bytes_transferred`, `files_transferred`, `files_skipped`, and `faults` in `task show`.

### 6d. Cancel / resume

```bash
globus task cancel <TASK_ID>
globus task cancel --all
```

To resume, **re-issue the exact same `globus transfer` command**. With
`--sync-level checksum`, files already present and matching are skipped, so a resumed run
only moves what is missing or wrong. Nothing needs to be deleted first, and the batch file
does not need to be trimmed. Globus also retries transient faults on its own and parks a
task as INACTIVE (rather than failing it) when credentials expire or Globus Connect
Personal goes offline; `globus task pause-info <TASK_ID>` explains why.

---

## Step 7 — The Google Drive for Desktop caveat (read before preprocessing)

Google Drive for Desktop on this Mac runs in **streaming** mode: the folder lives under
`~/Library/CloudStorage/GoogleDrive-<account>/My Drive/` and files are placeholders backed
by the cloud until something reads their bytes. Two consequences:

**There are two transfers in series, not one.** Globus writes to the local Drive folder,
then Drive uploads to Google's servers. Colab reads the **cloud** copy. The transfer is not
usable from Colab until Drive finishes its own upload of the full ~90 GB, which can take far
longer than the Globus leg.

Watch the Drive upload:

- The Drive icon in the macOS menu bar shows "Syncing N items" / a completion state.
- Locally: `find "<...>/DICOMs-knee-imaging" -name '*.dcm' | wc -l` should reach 6122.
- Authoritatively, from Colab, after mounting Drive:
  `!find "/content/drive/MyDrive/Radiographic Prediction of Contralateral Knee Arthroplasty/DICOMs-knee-imaging" -name '*.dcm' | wc -l`
  Only when that prints 6122 has the cloud copy caught up.

**Placeholders are fine for Colab, fatal for local preprocessing.** Colab mounts the cloud
copy, so placeholder status on this Mac is irrelevant there. But `src/preprocess_images.py`
running locally reads local bytes; every file it opens triggers an on-demand download, which
is slow and can fail under Drive's per-day API limits.

To force local materialization: right-click the folder in Finder and choose **Available
offline**, or switch Drive to **Mirror files** in Drive Preferences (menu-bar icon > gear >
Preferences > Google Drive > Mirror files). Both keep a full local copy.

> **Disk-space note (revised 2026-07-25).** This Mac has ~50 GB free and the cohort measures
> ~36 GB, so an offline/mirror copy **does** fit — with only ~14 GB to spare. That is tight
> enough to be risky while Colab, Drive cache and swap are also competing for the disk, so
> prefer an external volume if you have one. Switching to mirror mode can also relocate the folder; if it does, pass the new
> location with `python3 -m src.verify_transfer --dest-root <new path>` (the value in
> `config/feasibility.yaml` `transfer.dest_root` is the streaming-mode path).

Note that the file-size check in `src/verify_transfer.py` catches truncation reliably but
cannot always distinguish a placeholder (macOS reports the logical size for non-materialized
files). The `--sample-dicom` check is the definitive test, because reading pixel data forces
Drive to hand over real bytes.

---

## Step 8 — The efficiency alternative (decide before step 6b)

Instead of Drive-then-Colab, you can transfer to a **plain non-synced local folder**,
preprocess locally with `python3 -m src.preprocess_images`, and push only the small
WebDataset shards (a few hundred MB) to Drive for training. That skips ~90 GB of Drive
upload entirely.

**Revised 2026-07-25 after measuring the real volume (~36 GB, not ~90 GB).** The local route
now fits on this Mac's internal disk: ~36 GB of DICOMs against ~50 GB free leaves ~14 GB
headroom. That is workable but not comfortable — macOS wants several GB for swap and Drive
keeps its own cache — so prefer an external SSD if you have one.

**Choose the local route if you have ~50 GB of free non-synced disk and want the shards
today**; it skips the Google Drive cloud upload entirely, which is the slowest leg of the
Drive path. **Choose the Drive route if you would rather not run near a full disk**, or if
you want the DICOMs available to Colab for any later re-run. Either way, run the step-6a
pilot first and re-check `df -h` against the extrapolated total before committing.

To take it, set `--dest-root` to that plain folder's path in the endpoint path space, run
steps 5 and 6 unchanged, then
`python3 -m src.verify_transfer --dest-root <that folder>`.

---

## Step 9 — Troubleshooting

**Consent / activation errors.** On a GCS v5 mapped collection the first access fails with
a consent-required error and the CLI prints the exact fix. Run the command it prints,
verbatim, including the quotes:

```bash
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/<COLLECTION_UUID>/data_access]'
```

You can request it up front instead:

```bash
globus login --gcs <SRC_UUID>
```

If the error is about **identities** rather than scopes (a high-assurance collection
demanding a specific institutional login):

```bash
globus session update <your-institutional-identity>
globus session update --policy <POLICY_UUID>
globus session show
```

**`--sync-level` semantics.** `exists` transfers only if the destination file is absent;
`size` if sizes differ; `mtime` if the source is newer; `checksum` if contents differ. Use
`checksum`. After a failed transfer, `checksum` is the only safe level, because the others
will happily keep a half-written file.

**Partial or failed transfer, per-file errors.**

```bash
globus task show <TASK_ID>
globus task event-list <TASK_ID> --limit 50
globus task event-list <TASK_ID> --filter-errors --limit 50
globus task show <TASK_ID> --skipped-errors
globus task pause-info <TASK_ID>
```

Event history is retained for about a month. Then fix the cause and re-issue the same
`globus transfer ... --sync-level checksum` command; only the outstanding files move.
`--skip-source-errors` makes the task continue past unreadable source paths instead of
failing, but only add it once you know which files are affected and why.

**Destination path rejected** ("Path not found", "Permission denied", "not authorized"):

1. `globus ls "$(globus endpoint local-id)":/` and confirm the path really exists in the
   endpoint's path space, not just on the Mac.
2. Check Globus Connect Personal > Preferences > Access: the path must be listed and must
   **not** be read-only.
3. Check macOS Full Disk Access for Globus Connect Personal (step 3.4). This is the usual
   cause of a permission error when writing into `~/Library/CloudStorage/`.
4. Re-read step 4: an absolute macOS path is almost never the correct `<DEST_ROOT>` — it
   must exist in the endpoint's own path space. (Spaces in it are fine; the batch file
   quotes them.)
5. Re-run the step-5 token check; every line must split into exactly 2 tokens:

   ```bash
   python3 -c "import shlex,sys; bad=[i for i,l in enumerate(open(sys.argv[1])) if len(shlex.split(l, comments=True)) != 2]; print('OK: all lines split into exactly 2 tokens' if not bad else f'BAD lines: {bad[:5]}'); sys.exit(1 if bad else 0)" derived-data/cohort/globus_batch.txt
   ```
6. If the destination is being written to the wrong place rather than rejected, check
   whether your batch dest paths start with `/`: an absolute batch dest path makes
   `globus transfer` ignore the command-line `DEST_PATH` prefix (step 4).

**Task goes INACTIVE.** Usually credentials expired or Globus Connect Personal is not
running. Restart the app, then `globus session show`; re-login if needed. The task resumes
on its own.

---

## Step 10 — Verify

```bash
cd "$MRKR_PROJECT"
python3 -m src.verify_transfer
```

Optional overrides: `--dest-root <path>` for a non-default destination,
`--sample-dicom 50` to read more files, `--sample-dicom 0` to skip the pydicom pass.

A pass looks like this:

```text
verify_transfer | INFO | manifest: 6122 unique relative paths (config expects 6122)
verify_transfer | INFO | dicom sample: 20 opened, 20 parsed with pixel data, 0 failed, 0 warned
verify_transfer | INFO | Manifest matches config          PASS 6,122 manifest paths vs 6,122 expected
verify_transfer | INFO | DICOM file count                 PASS 6,122 `*.dcm` files found vs 6,122 expected
verify_transfer | INFO | Every manifest path present      PASS 0 missing
verify_transfer | INFO | No unexpected extra DICOMs       PASS 0 extra
verify_transfer | INFO | No file below the minimum size   PASS 0 smaller than 1,024 B
verify_transfer | INFO | DICOM read sample                PASS 20/20 parsed with pixel data, 0 failed
VERDICT: PASS — 6,122/6,122 images present (100.00%)
```

Exit code 0 only when every check passes, so the preprocessing step can be gated on it
(`python3 -m src.verify_transfer && python3 -m src.preprocess_images`). The full aggregate
report is written to `outputs/transfer_verification.md`. Before the transfer starts it
reports `0 of 6,122 present, transfer not started` and exits 1, which is expected.
