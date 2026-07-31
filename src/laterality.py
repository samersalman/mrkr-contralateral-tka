"""Pure, unit-testable CPT / laterality parsing and timeline helpers.

Phase-1 metadata-only feasibility module for the MRKR Contralateral TKA study.
Every function here is PURE: it reads no files, opens no DICOMs, performs no I/O,
and has no side effects on import. Token lists and timeline constants are passed
as explicit arguments (defaulting to the study's verified values) so the core
logic can be exercised with synthetic inputs alone. The one convenience wrapper
that reads ``config/feasibility.yaml`` (``parse_modifier_from_config``) does so
lazily, only when called.

Downstream modules (index_tka / preliminary_counts) import ``normalize_cpt``,
``parse_modifier``, ``contralateral_side`` and the date helpers below.

Verified raw ``cpt_group_modifier`` values for CPT 27447 that these functions
must handle: NULL/blank (~61%), ``RT``, ``LT``, ``50``, multi-token strings such
as ``RT XP`` / ``74 LT`` / ``LT XU`` / ``59 RT`` / ``22 LT`` / ``LT XP`` /
``LT XE`` / ``73 LT``, and non-laterality tokens like ``22``. No string carrying
BOTH RT and LT was observed, but ``conflicting`` is still handled defensively.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Iterable

# Default study tokens (mirrors config/feasibility.yaml -> laterality:). These
# are the defaults for the pure functions; callers may override explicitly.
DEFAULT_RIGHT_TOKENS: tuple[str, ...] = ("RT",)
DEFAULT_LEFT_TOKENS: tuple[str, ...] = ("LT",)
DEFAULT_BILATERAL_TOKENS: tuple[str, ...] = ("50",)

# Canonical side codes (mirrors config side_values).
SIDE_RIGHT = "R"
SIDE_LEFT = "L"
SIDE_BILATERAL = "B"
SIDE_UNKNOWN = "U"


# ---------------------------------------------------------------------------
# CPT code normalization
# ---------------------------------------------------------------------------
def normalize_cpt(code: object) -> str | None:
    """Normalize a raw CPT value to a canonical 5-character code, else ``None``.

    Rules (kept deliberately simple; this study centers on numeric 5-char CPT):
      * ``None`` / blank / whitespace-only -> ``None``.
      * A purely numeric value is zero-padded on the left to width 5, so
        ``'27447'`` and ``' 27447 '`` -> ``'27447'`` and a short but otherwise
        valid numeric code is padded (e.g. ``'447'`` -> ``'00447'``).
      * A numeric value longer than 5 digits cannot be a 5-char code -> ``None``.
      * A HCPCS-style token of exactly 4 digits + 1 trailing letter (e.g.
        ``'0074T'``) is a valid 5-char CPT-like token and is returned uppercased.
        (For THIS study the focus is numeric 5-char; letter-suffix codes are
        accepted for completeness but are not the primary path.)
      * Anything else (embedded punctuation, wrong shape, junk) -> ``None``.

    Args:
        code: raw value from the CPT code column (str/int/None/etc.).

    Returns:
        The 5-character CPT string, or ``None`` if it cannot be one.
    """
    if code is None:
        return None
    # Guard against float NaN (a common "missing" representation).
    if isinstance(code, float) and math.isnan(code):
        return None

    s = str(code).strip()
    if not s:
        return None

    # Primary path: purely numeric CPT.
    if s.isdigit():
        if len(s) > 5:
            return None
        return s.zfill(5)

    # Secondary: HCPCS-style 4 digits + 1 trailing letter (exactly 5 chars).
    if len(s) == 5 and s[:4].isdigit() and s[4].isalpha():
        return s.upper()

    return None


# ---------------------------------------------------------------------------
# Laterality-modifier parsing
# ---------------------------------------------------------------------------
def parse_modifier(
    raw: object,
    right_tokens: Iterable[str] = DEFAULT_RIGHT_TOKENS,
    left_tokens: Iterable[str] = DEFAULT_LEFT_TOKENS,
    bilateral_tokens: Iterable[str] = DEFAULT_BILATERAL_TOKENS,
) -> tuple[str, str]:
    """Parse a raw CPT modifier string into ``(side, quality_flag)``.

    Tokens are whitespace-delimited and matched case-insensitively; leading,
    trailing, and repeated internal spaces are tolerated. ``side`` is one of
    ``R`` / ``L`` / ``B`` / ``U``; ``quality_flag`` is one of the seven
    config-declared flags.

    Decision table (evaluated in this order):
      * ``None`` / empty / whitespace / float NaN     -> ``('U', 'missing')``
      * BOTH a right and a left token present anywhere -> ``('U', 'conflicting')``
      * single token == right token (e.g. ``RT``)      -> ``('R', 'single_rt')``
      * single token == left token  (e.g. ``LT``)      -> ``('L', 'single_lt')``
      * single token == bilateral   (e.g. ``50``)      -> ``('B', 'bilateral_50')``
      * single non-laterality token (e.g. ``22``)      -> ``('U', 'uninterpretable')``
      * multi-token containing a bilateral token       -> ``('B', 'bilateral_50')``
      * multi-token with exactly one of right/left     -> ``('R'|'L', 'multi_single_side')``
      * multi-token with no laterality/bilateral token -> ``('U', 'uninterpretable')``

    Precedence notes (documented, resolving cases not in the raw data):
      * ``conflicting`` (RT and LT both present) dominates everything except
        ``missing``; this also enforces the "not a RT/LT conflict" caveat on the
        bilateral rule, so e.g. ``'50 RT LT'`` -> ``conflicting``.
      * A bilateral modifier alongside a single side but no conflict
        (e.g. ``'50 RT'``) is classified ``bilateral_50`` -- modifier 50 signals a
        bilateral procedure and takes precedence over ``multi_single_side``.

    Args:
        raw: the raw modifier value (e.g. ``'RT XP'``, ``'74 LT'``, ``None``).
        right_tokens: tokens meaning "right" (default ``('RT',)``).
        left_tokens: tokens meaning "left" (default ``('LT',)``).
        bilateral_tokens: tokens meaning "bilateral" (default ``('50',)``).

    Returns:
        ``(side, quality_flag)`` tuple.
    """
    # Missing: None, float NaN, empty, or whitespace-only.
    if raw is None:
        return (SIDE_UNKNOWN, "missing")
    if isinstance(raw, float) and math.isnan(raw):
        return (SIDE_UNKNOWN, "missing")
    s = str(raw).strip()
    if not s:
        return (SIDE_UNKNOWN, "missing")

    tokens = [t.upper() for t in s.split()]
    right_set = {str(t).upper() for t in right_tokens}
    left_set = {str(t).upper() for t in left_tokens}
    bilateral_set = {str(t).upper() for t in bilateral_tokens}

    has_right = any(t in right_set for t in tokens)
    has_left = any(t in left_set for t in tokens)
    has_bilateral = any(t in bilateral_set for t in tokens)

    # Both sides present anywhere -> unusable (dominates all but 'missing').
    if has_right and has_left:
        return (SIDE_UNKNOWN, "conflicting")

    # Single-token cases (exact laterality / bilateral / junk).
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in right_set:
            return (SIDE_RIGHT, "single_rt")
        if tok in left_set:
            return (SIDE_LEFT, "single_lt")
        if tok in bilateral_set:
            return (SIDE_BILATERAL, "bilateral_50")
        return (SIDE_UNKNOWN, "uninterpretable")

    # Multi-token from here (and NOT a RT/LT conflict).
    # Bilateral modifier present -> bilateral (takes precedence over single side).
    if has_bilateral:
        return (SIDE_BILATERAL, "bilateral_50")

    # Exactly one of right/left among multiple tokens -> that side.
    if has_right:
        return (SIDE_RIGHT, "multi_single_side")
    if has_left:
        return (SIDE_LEFT, "multi_single_side")

    # No laterality or bilateral token at all -> uninterpretable.
    return (SIDE_UNKNOWN, "uninterpretable")


def parse_modifier_from_config(raw: object, cfg: object | None = None) -> tuple[str, str]:
    """Thin convenience wrapper: parse ``raw`` using tokens from the config file.

    Loads ``config/feasibility.yaml`` lazily (only when called) if ``cfg`` is not
    supplied, then delegates to the pure :func:`parse_modifier`. This is the only
    function in the module that touches the filesystem, and never on import.

    Args:
        raw: the raw modifier value.
        cfg: an already-loaded config mapping; if ``None`` it is loaded via
            ``src.config.load_config``.

    Returns:
        ``(side, quality_flag)`` from :func:`parse_modifier`.
    """
    if cfg is None:
        from src.config import load_config  # lazy import: no I/O at module import

        cfg = load_config()
    lat = cfg["laterality"]
    return parse_modifier(
        raw,
        right_tokens=lat.get("right_tokens", DEFAULT_RIGHT_TOKENS),
        left_tokens=lat.get("left_tokens", DEFAULT_LEFT_TOKENS),
        bilateral_tokens=lat.get("bilateral_tokens", DEFAULT_BILATERAL_TOKENS),
    )


def contralateral_side(index_side: object) -> str | None:
    """Return the opposite knee side, or ``None`` when not a single side.

    ``'R'`` -> ``'L'`` and ``'L'`` -> ``'R'`` (input is stripped/upper-cased for
    robustness). Any other value -- ``'B'`` (bilateral), ``'U'`` (unknown),
    ``None``, blanks -- returns ``None`` because no single contralateral side is
    defined.

    Args:
        index_side: the index-knee side code.

    Returns:
        ``'L'`` / ``'R'`` / ``None``.
    """
    if index_side is None:
        return None
    s = str(index_side).strip().upper()
    if s == SIDE_RIGHT:
        return SIDE_LEFT
    if s == SIDE_LEFT:
        return SIDE_RIGHT
    return None


# ---------------------------------------------------------------------------
# Pure date / timeline helpers (operate on datetime.date)
# ---------------------------------------------------------------------------
def days_between(d0: date | None, d1: date | None) -> int | None:
    """Signed day count ``d1 - d0``; ``None`` if either date is ``None``.

    Positive when ``d1`` is after ``d0``, negative when before, zero when equal.
    """
    if d0 is None or d1 is None:
        return None
    return (d1 - d0).days


def add_days(d: date | None, n: int) -> date | None:
    """Return ``d`` shifted by ``n`` days (negative allowed); ``None`` passes through."""
    if d is None:
        return None
    return d + timedelta(days=int(n))


def landmark_date(index_date: date | None, landmark_days: int = 90) -> date | None:
    """Landmark (time-origin) date = ``index_date + landmark_days`` (default day 90).

    Example: ``landmark_date(date(2018, 1, 1))`` -> ``date(2018, 4, 1)``.
    """
    return add_days(index_date, landmark_days)


def horizon_date(
    index_date: date | None,
    years: float = 5,
    days_per_year: float = 365.25,
) -> date | None:
    """Study-horizon date = ``index_date + round(years * days_per_year)`` days.

    With the study defaults (5 years, 365.25 days/year) the offset is
    ``round(1826.25) == 1826`` days, so ``horizon_date(date(2018, 1, 1), 5)``
    -> ``date(2023, 1, 1)``.
    """
    if index_date is None:
        return None
    n = round(years * days_per_year)
    return add_days(index_date, n)


def within(days: int | None, lo: int, hi: int) -> bool:
    """Inclusive membership test ``lo <= days <= hi``; ``None`` days -> ``False``."""
    if days is None:
        return False
    return lo <= days <= hi


def last_observation(dates: Iterable[date | None]) -> date | None:
    """Latest date in an iterable, ignoring ``None``; ``None`` if none remain.

    Used for the observation-through-landmark rule (max across CPT/ICD/pain/image
    dates). An empty iterable or one that is all ``None`` returns ``None``.
    """
    valid = [d for d in dates if d is not None]
    return max(valid) if valid else None
