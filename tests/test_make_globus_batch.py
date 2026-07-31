"""Unit tests for src/make_globus_batch.py — the Globus `--batch` quoting contract.

`globus transfer --batch` parses every line with ``shlex.split(line, comments=True)``
(globus_cli/utils.py:197 in globus-cli 3.42.0) and then requires EXACTLY two arguments.
These tests pin that contract from both sides:

* a destination root containing spaces (the real Google Drive path has two space-bearing
  components) must still yield exactly two tokens that round-trip to the intended paths;
* a space-free root must produce output BYTE-IDENTICAL to the pre-quoting format, so the
  fix cannot have silently changed the transfer that has already been documented.

Synthetic manifest only — no DICOM path from the real cohort is read.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from src.config import DEFAULT_CONFIG, load_config
from src.make_globus_batch import main

SRC_ROOT = "/mrkr/dicoms"
# Structurally identical to the real Google Drive destination — two space-bearing components
# and an account-bearing component, which unquoted split into 8 tokens — with the home
# directory and the account replaced by placeholders. Protocol section 28 publishes this
# repository, so no tracked file carries a personal path.
DRIVE_ROOT = ("/home/USER/CloudStorage/GoogleDrive-ACCOUNT@example.com/"
              "My Drive/Radiographic Prediction of Contralateral Knee Arthroplasty/"
              "DICOMs-knee-imaging")
# Wholly invented ids — never paste a real empi_anon into a tracked file.
RELS = ["SYNTH_PATIENT_A/1.2.841.1/1.2.840.1/1.2.826.0.dcm",
        "SYNTH_PATIENT_B/1.2.841.2/1.2.840.2/1.2.826.1.dcm",
        "SYNTH_PATIENT_C/1.2.841.3/1.2.840.3/1.2.826.2.dcm"]


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    p = tmp_path / "paths.txt"
    p.write_text("\n".join(RELS) + "\n")
    return p


def _run(manifest: Path, out: Path, dest_root: str, src_root: str = SRC_ROOT) -> list[str]:
    rc = main(["--config", str(DEFAULT_CONFIG), "--source-root", src_root,
               "--dest-root", dest_root, "--paths", str(manifest), "--out", str(out)])
    assert rc == 0
    return out.read_text().splitlines()


# --------------------------------------------------------------------------- #
# The space-bearing destination — the case that used to be rejected             #
# --------------------------------------------------------------------------- #
def test_space_bearing_dest_root_gives_exactly_two_shlex_tokens(manifest, tmp_path):
    lines = _run(manifest, tmp_path / "batch.txt", DRIVE_ROOT)
    assert len(lines) == len(RELS)
    for line in lines:
        assert len(shlex.split(line, comments=True)) == 2, \
            f"globus-cli would reject this line: {line!r}"


def test_space_bearing_dest_root_round_trips_to_the_intended_paths(manifest, tmp_path):
    lines = _run(manifest, tmp_path / "batch.txt", DRIVE_ROOT)
    for line, rel in zip(lines, RELS):
        src, dst = shlex.split(line, comments=True)
        assert src == f"{SRC_ROOT}/{rel}"
        assert dst == f"{DRIVE_ROOT}/{rel}"


def test_the_unquoted_form_would_have_been_rejected(manifest, tmp_path):
    """Guards the test above from being vacuous: prove the naive line really does break."""
    naive = f"{SRC_ROOT}/{RELS[0]} {DRIVE_ROOT}/{RELS[0]}"
    assert len(shlex.split(naive, comments=True)) == 8


def test_space_bearing_source_root_is_quoted_too(manifest, tmp_path):
    src = "/mrkr archive/knee dicoms"
    lines = _run(manifest, tmp_path / "batch.txt", "/DICOMs-knee-imaging", src_root=src)
    for line, rel in zip(lines, RELS):
        a, b = shlex.split(line, comments=True)
        assert a == f"{src}/{rel}" and b == f"/DICOMs-knee-imaging/{rel}"


def test_hash_in_a_path_is_not_truncated_by_the_comment_parser(tmp_path):
    """`comments=True` eats everything from a `#` onward — quoting is what stops it."""
    rel = "SYNTH#PATIENT_D/1.2.841.1/1.2.840.1/1.2.826.0.dcm"
    m = tmp_path / "paths.txt"
    m.write_text(rel + "\n")
    line = _run(m, tmp_path / "batch.txt", "/DICOMs-knee-imaging")[0]
    src, dst = shlex.split(line, comments=True)
    assert src == f"{SRC_ROOT}/{rel}" and dst == f"/DICOMs-knee-imaging/{rel}"
    assert shlex.split(f"{SRC_ROOT}/{rel} /DICOMs-knee-imaging/{rel}",
                       comments=True) != [src, dst], "unquoted, this path silently truncates"


# --------------------------------------------------------------------------- #
# No regression for the space-free roots that were already documented           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dest_root", ["/DICOMs-knee-imaging", ".", "/", "/data_2024/knee-x_rays"])
def test_space_free_roots_are_byte_identical_to_the_unquoted_format(manifest, tmp_path, dest_root):
    lines = _run(manifest, tmp_path / "batch.txt", dest_root)
    stripped = dest_root.rstrip("/")
    assert lines == [f"{SRC_ROOT}/{rel} {stripped}/{rel}" for rel in RELS]


def test_duplicate_and_blank_manifest_lines_are_dropped(tmp_path):
    m = tmp_path / "paths.txt"
    m.write_text(f"{RELS[0]}\n\n{RELS[0]}\n/{RELS[1]}\n")
    lines = _run(m, tmp_path / "batch.txt", "/DICOMs-knee-imaging")
    assert len(lines) == 2, "duplicates and blanks must not produce transfer lines"
    assert [shlex.split(l)[0] for l in lines] == [f"{SRC_ROOT}/{RELS[0]}", f"{SRC_ROOT}/{RELS[1]}"]


# --------------------------------------------------------------------------- #
# Config wiring: transfer.batch_file / transfer.manifest_paths are the defaults  #
# --------------------------------------------------------------------------- #
def test_default_paths_come_from_the_transfer_config_block(tmp_path):
    cfg = dict(load_config(DEFAULT_CONFIG))
    manifest = tmp_path / "from_config_manifest.txt"
    manifest.write_text("\n".join(RELS) + "\n")
    out = tmp_path / "nested" / "from_config_batch.txt"
    cfg["transfer"] = dict(cfg["transfer"]) | {"manifest_paths": str(manifest),
                                               "batch_file": str(out)}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    assert main(["--config", str(cfg_path), "--source-root", SRC_ROOT,
                 "--dest-root", "/DICOMs-knee-imaging"]) == 0
    assert out.exists(), "transfer.batch_file was not used as the default output path"
    assert len(out.read_text().splitlines()) == len(RELS), \
        "transfer.manifest_paths was not used as the default input"


def test_the_shipped_config_points_the_batch_file_at_the_gitignored_tree():
    tr = load_config(DEFAULT_CONFIG)["transfer"]
    assert tr["batch_file"].startswith("derived-data/"), \
        "the batch file lists DICOM paths and must stay under the git-ignored tree"
    assert tr["manifest_paths"].startswith("outputs/tables/")


# --------------------------------------------------------------------------- #
# Portable destination root (protocol section 28: the repository is public)     #
# --------------------------------------------------------------------------- #
def test_config_path_expands_environment_variables(monkeypatch, tmp_path):
    from src.config import PROJECT_ROOT, load_config as _load

    cfg = _load(DEFAULT_CONFIG)
    monkeypatch.setenv("MRKR_TEST_ROOT", "/some/mount")
    assert cfg.path("${MRKR_TEST_ROOT}/DICOMs") == Path("/some/mount/DICOMs")
    assert cfg.path("$MRKR_TEST_ROOT/DICOMs") == Path("/some/mount/DICOMs")
    # Backward compatibility: a plain relative path is untouched and still project-relative.
    assert cfg.path("outputs/tables") == PROJECT_ROOT / "outputs/tables"
    assert cfg.path("/already/absolute") == Path("/already/absolute")


def test_transfer_dest_root_defaults_to_config_and_yields_to_the_env(monkeypatch):
    from src.config import DEST_ROOT_ENV, load_config as _load

    cfg = _load(DEFAULT_CONFIG)
    monkeypatch.delenv(DEST_ROOT_ENV, raising=False)
    assert cfg.transfer_dest_root() == Path(cfg["transfer"]["dest_root"]), \
        "with the variable unset nothing may change for the current machine"
    monkeypatch.setenv(DEST_ROOT_ENV, "/mnt/drive/DICOMs-knee-imaging")
    assert cfg.transfer_dest_root() == Path("/mnt/drive/DICOMs-knee-imaging")
    monkeypatch.setenv(DEST_ROOT_ENV, "   ")
    assert cfg.transfer_dest_root() == Path(cfg["transfer"]["dest_root"]), \
        "a blank override must fall back rather than resolve to the project root"


def test_verify_transfer_resolves_the_destination_through_the_override(monkeypatch):
    """src/verify_transfer.py must keep working, and must honour the env override."""
    from src import verify_transfer
    from src.config import DEST_ROOT_ENV, load_config as _load

    assert "transfer_dest_root" in verify_transfer.main.__code__.co_names
    monkeypatch.setenv(DEST_ROOT_ENV, "/mnt/elsewhere")
    assert _load(DEFAULT_CONFIG).transfer_dest_root() == Path("/mnt/elsewhere")
