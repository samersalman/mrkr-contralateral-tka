"""Configuration loader for the feasibility pipeline.

All modules import this to read config/feasibility.yaml and to resolve paths
relative to the project root. This module has NO side effects on import.

Usage:
    from src.config import load_config, PROJECT_ROOT
    cfg = load_config()                      # dict
    cpt_csv = cfg.source_path("cpt")         # absolute path to MRKR_CPT.csv
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root = parent directory of this file's package (src/..).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "feasibility.yaml"

# Environment override for the bulk-transfer destination (protocol section 28: the
# analysis repository is public, so no tracked file should have to carry somebody's
# home directory or account name). Unset -> the configured default is used unchanged.
DEST_ROOT_ENV = "MRKR_DRIVE_ROOT"


class Config(dict):
    """Thin dict wrapper with path-resolution helpers.

    Keeps the raw YAML structure accessible via normal dict access while adding
    a few convenience methods that resolve paths against PROJECT_ROOT.
    """

    def path(self, rel: str) -> Path:
        """Resolve a project-relative path to an absolute Path.

        ``${VAR}`` / ``$VAR`` references are expanded from the environment first, so a
        published configuration can carry ``"${MRKR_DRIVE_ROOT}/DICOMs"`` instead of a
        personal absolute path. A value containing no ``$`` is returned exactly as
        before, so this is backward-compatible; an undefined variable is left literal by
        ``os.path.expandvars`` and will surface as a missing path rather than silently
        resolving somewhere unexpected.
        """
        p = Path(os.path.expandvars(str(rel)))
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def transfer_dest_root(self) -> Path:
        """Destination root for the bulk DICOM transfer.

        Precedence: the ``MRKR_DRIVE_ROOT`` environment variable, then
        ``transfer.dest_root`` from the config. The configured value is still the
        default, so nothing changes for the current machine; setting the variable lets
        a second user run the same commands without editing a tracked file.
        """
        env = os.environ.get(DEST_ROOT_ENV, "").strip()
        return self.path(env or self["transfer"]["dest_root"])

    def source_path(self, key: str) -> Path:
        """Absolute path to a source CSV named in config['source_files'][key]."""
        fname = self["source_files"][key]["filename"]
        return self.path(self["paths"]["metadata_dir"]) / fname

    def parquet_path(self, key: str) -> Path:
        """Absolute path to the typed Parquet for a source table."""
        return self.path(self["paths"]["source_parquet_dir"]) / f"{key}.parquet"

    def out(self, rel_key: str) -> Path:
        """Absolute path for a configured output location, e.g. out('tables_dir')."""
        return self.path(self["paths"][rel_key])


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load feasibility.yaml and return a Config (dict subclass)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return Config(data)


def ensure_dirs(cfg: Config) -> None:
    """Create the standard output/derived directories if missing (idempotent)."""
    for key in ("source_parquet_dir", "cohort_dir", "tables_dir", "figures_dir", "logs_dir"):
        cfg.path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
