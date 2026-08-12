"""Protocol section 22: image interpretability, occlusion, and leakage controls.

Why this module exists
----------------------
The study protocol (section 22) pre-specified Grad-CAM / integrated-gradient attribution
over a stratified TP/FP/TN/FN test sample, occlusion by anatomic region, saliency sanity
checks after randomising the model, and a masked-text/marker/border prediction-change
test. None of it was run before the first submission and no deviation entry covered the
gap; the completed CLAIM checklist asserted the opposite. The *Journal of Imaging*
academic editor independently called the same package Essential, and was explicit that
attribution *pictures* are not enough: what is wanted is "a quantitative measure such as
the proportion of attribution within a predefined joint region and an occlusion
experiment". This module produces that quantitative measure.

The one methodological rule everything here obeys
-------------------------------------------------
The published per-patient hazards in ``derived-data/cohort/test_hazards_*.npz`` were
computed on Colab CUDA under float16 autocast. This module runs local float32. Aggregate
discrimination reproduces to <=5.4e-04, but **individual predicted risks reproduce only to
about 1e-02**. A delta formed by subtracting a published risk from a locally recomputed
one would therefore be measuring float16-versus-float32, not anatomy.

So: **every baseline in this module is recomputed locally, in the same process, on the
same device, with the same code path as the perturbed condition.** No published number is
ever an arm of a comparison. ``ArmRunner.score`` is the only way a prediction enters this
module, and the unperturbed condition goes through it exactly like every other condition.

Nothing is fitted. The frozen checkpoints, the frozen ensemble rule (average the hazards
across the five pre-specified seeds) and the frozen validation-fitted recalibration are
applied as-is, through the same primitives ``src/score_test.py`` uses. The published
``.npz`` files are never written.

The regions
-----------
Every crop is a 512x512 uint8 image built by ``src/preprocess_images.crop_stages``: a
square box centred on the collimated field, resized with LANCZOS, its outer 31 px zeroed,
and mirrored so that every knee reads as a LEFT knee. Three facts follow, and they are
what make a fixed geometric region defensible rather than arbitrary:

1. the crop is *centred*, so the tibiofemoral joint is central by construction;
2. the border band is a fixed 31 px frame = 22.75% of the area (config
   ``preprocess.mask_border_frac`` 0.06 x 512 = 31, and 1 - (450/512)^2 = 0.2275);
3. because right knees are mirrored to read as left knees under the radiological display
   convention (``bilateral_display_convention: radiological``, patient RIGHT on image
   LEFT), the **medial** compartment is always on the image LEFT and the **lateral**
   compartment always on the image RIGHT. This is verifiable on the images themselves:
   the fibular head, which is lateral by definition, sits on the image right.

``JOINT_BOX`` is the central half of the image in each dimension - rows and columns
[128, 384) - which is 25.00% of the area. A model-free matched-filter detector
(:func:`detect_joint_row`, a dark horizontal line with bright bone above and below) places
the tibiofemoral joint line inside that row band on 91.0% of the 858 frontal test crops.
The region was fixed on that geometric argument, not tuned against any attribution map.

Run::

    PYTHONPATH="$PWD" python -m src.interpretability --stage all \\
        --shard-dir <staged shards-test> --ckpt-dir <staged ckpt> --cache-dir <cache-test>
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.model_clinical import ipcw_auc, ipcw_labels_weights
from src.score_test import (
    EXPECTED_TEST_CROPS,
    EXPECTED_TEST_PATIENTS,
    assert_frozen,
    build_test_frames,
)
from src.train_model import (
    EDGES,
    N_INTERVALS,
    SEALED_SPLIT,
    FrozenContracts,
    PatientViewDataset,
    SurvivalFusionNet,
    TrainSettings,
    apply_recalibration,
    average_hazard,
    load_seed_model,
    load_sidecar,
    materialize_split,
    read_json_retrying,
    require_torch,
    risk_at_horizon,
    risk_score,
    seed_everything,
)

MODULE = "interpretability"

# --------------------------------------------------------------------------- #
# Frozen geometry. Nothing below is tuned; every number is either config-derived
# or a plain fraction of the 512 px crop.
# --------------------------------------------------------------------------- #
OUT_SIZE = 512
BORDER_PX = 31                                   # round(0.06 * 512), protocol section 13
JOINT_BOX = (128, 384, 128, 384)                 # (r0, r1, c0, c1): central half, 25.00%
PF_BOX = (128, 256, 192, 320)                    # patellar projection zone, 6.25%
HORIZON_DAYS = 1825                              # the reported 5-year horizon
PRIMARY_ARM = "m2_frontal"
MULTIVIEW_ARMS = ("m3_image", "m4_fusion")
CAM_LAYER = "encoder.features.norm5"             # densenet121 final BN, 16x16 x 1024


def region_masks(out_size: int = OUT_SIZE, border_px: int = BORDER_PX) -> dict[str, np.ndarray]:
    """The predefined regions, as boolean (out_size, out_size) masks.

    ``joint``/``border``/``peripheral`` PARTITION the image exactly; the rest are
    sub-regions used by the occlusion experiment and may overlap each other.
    """
    n = int(out_size)
    b = int(border_px)
    r0, r1, c0, c1 = JOINT_BOX
    pr0, pr1, pc0, pc1 = PF_BOX
    assert b < r0 and b < c0, "the joint box must not touch the border band"
    joint = np.zeros((n, n), dtype=bool); joint[r0:r1, c0:c1] = True
    border = np.zeros((n, n), dtype=bool)
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    medial = joint.copy(); medial[:, n // 2:] = False          # image LEFT  = medial
    lateral = joint.copy(); lateral[:, :n // 2] = False        # image RIGHT = lateral
    pf = np.zeros((n, n), dtype=bool); pf[pr0:pr1, pc0:pc1] = True
    out = {"joint": joint, "border": border,
           "peripheral": ~(joint | border),
           "medial": medial, "lateral": lateral, "patellofemoral": pf}
    assert not (out["joint"] & out["border"]).any()
    assert (out["joint"] | out["border"] | out["peripheral"]).all()
    return out


def region_area_fractions(out_size: int = OUT_SIZE, border_px: int = BORDER_PX) -> dict[str, float]:
    """Area of each region as a fraction of the image - the CHANCE level for B2."""
    m = region_masks(out_size, border_px)
    return {k: float(v.sum()) / float(out_size * out_size) for k, v in m.items()}


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=float), np.ones(int(k)) / float(k), mode="same")


def detect_joint_row(img: np.ndarray, *, half_col: int = 96, lo: int = 120,
                     hi: int = 400) -> int:
    """Row of the tibiofemoral joint space in one frontal crop. Model-free.

    The joint space is a dark horizontal line with bright bone above and below. The mean
    column profile is detrended (a 121 px moving average removes the smooth thigh-to-calf
    intensity gradient) and scored with a matched filter that rewards bright at +-(12..34)
    px and dark within +-8 px. Exists only to VALIDATE :data:`JOINT_BOX`; no attribution
    result depends on it.
    """
    a = np.asarray(img, dtype=np.float32)
    n = a.shape[0]
    mid = n // 2
    prof = a[:, mid - half_col:mid + half_col].mean(axis=1)
    det = _moving_average(prof - _moving_average(prof, 121), 5)
    best, best_i = -np.inf, int(lo)
    for i in range(int(lo), int(hi)):
        up = det[max(0, i - 34):i - 11].mean()
        dn = det[i + 12:min(n, i + 35)].mean()
        ctr = det[i - 8:i + 9].mean()
        r = float(up + dn - 2.0 * ctr)
        if r > best:
            best, best_i = r, i
    return int(best_i)


# =========================================================================== #
# 1. CONTEXT: the frozen inputs, loaded once                                   #
# =========================================================================== #
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a")
        fh._mrkr = True                                        # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"))
        lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True                                        # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        lg.addHandler(sh)
    return lg


@dataclass
class InterpContext:
    """Everything the frozen models need, plus the split's images and labels."""

    cfg: Config
    settings: TrainSettings                 # contract-exact; used to LOAD checkpoints
    runtime: TrainSettings                  # same, with num_workers 0; used to PREDICT
    device: object
    train_arms: dict
    contracts: FrozenContracts
    frames: dict
    designs: dict
    stats: dict
    labels: pd.DataFrame
    npy: Path
    index: pd.DataFrame
    log: logging.Logger
    out_dir: Path
    table_dir: Path
    images: np.ndarray = field(default=None, repr=False)

    @property
    def max_elems(self) -> int:
        return int(self.labels.groupby("empi_anon").size().max())


def build_context(config: str, shard_dir: Path, ckpt_dir: Path, cache_dir: Path,
                  device_name: str | None, out_dir: Path, table_dir: Path) -> InterpContext:
    """Load the frozen record, redirect the three staging paths, materialise the split.

    ``settings`` MUST carry the recorded ``grad_accum_steps`` and ``num_workers``: both are
    inside the training-contract hash, and ``load_seed_model`` refuses a checkpoint whose
    hash does not match with a message that blames the model rather than the config.
    """
    require_torch()
    import torch

    cfg: Config = load_config(config)
    log = setup_logging(Path(out_dir) / "interpretability.log")
    cohort_dir = cfg.path(cfg["paths"]["cohort_dir"])
    train_arms = read_json_retrying(cohort_dir / "train_arms.json")
    tc = train_arms.get("training_contract") or {}
    settings = TrainSettings(cfg, grad_accum_override=tc.get("grad_accum_steps"),
                             num_workers_override=tc.get("num_workers"))
    settings.ckpt_dir = Path(ckpt_dir).expanduser()
    settings.shard_dir = Path(shard_dir).expanduser()
    settings.cache_dir = Path(cache_dir).expanduser()
    assert_frozen(train_arms, settings, log)

    # A SEPARATE settings object for the DataLoader. num_workers is inside the contract
    # hash, so mutating the object the checkpoints are loaded against would make every
    # load_seed_model call fail. Worker count changes nothing at eval time (no
    # augmentation, no shuffling), and 0 avoids spawning two processes per condition on a
    # machine that is already contended.
    runtime = copy.copy(settings)
    runtime.num_workers = 0

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    assert device.type != "cuda" or "cuda" in settings.amp_devices
    seed_everything(int(cfg["reproducibility"]["random_seed"]))

    labels = load_sidecar(settings.shard_dir, splits=[SEALED_SPLIT],
                          expected=EXPECTED_TEST_CROPS,
                          expected_patients=EXPECTED_TEST_PATIENTS)
    npy, index = materialize_split(SEALED_SPLIT, shard_dir=settings.shard_dir,
                                   cache_dir=settings.cache_dir, labels=labels,
                                   out_size=settings.out_size, log=log, allow_sealed=True)
    contracts = FrozenContracts(cohort_dir)
    frames, designs, stats = build_test_frames(contracts, log)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(table_dir).mkdir(parents=True, exist_ok=True)
    log.info("device %s | float32 (no autocast) | border band %d px | joint box %s",
             device, settings.border_px, JOINT_BOX)
    return InterpContext(cfg=cfg, settings=settings, runtime=runtime, device=device,
                         train_arms=train_arms, contracts=contracts, frames=frames,
                         designs=designs, stats=stats, labels=labels, npy=npy, index=index,
                         log=log, out_dir=Path(out_dir), table_dir=Path(table_dir),
                         images=np.load(npy, mmap_mode="r"))


# =========================================================================== #
# 2. PIXEL OPERATIONS: the perturbations, applied to the uint8 crop            #
# =========================================================================== #
@dataclass(frozen=True)
class PixelOp:
    """One reproducible perturbation of a 512x512 uint8 crop.

    ``zero`` names regions to blank; ``keep`` (if given) blanks everything OUTSIDE the
    union of the named regions. ``fill`` is 0 by default because 0 is what the pipeline
    itself writes into the border band and into out-of-bounds padding, so a zero patch is
    the one occlusion value the encoder has already seen thousands of times in training.
    ``mean_fill`` substitutes the mean of the RETAINED pixels instead, which is the
    Zeiler-Fergus grey-patch convention and is reported as a sensitivity analysis.
    """

    name: str
    zero: tuple[str, ...] = ()
    keep: tuple[str, ...] = ()
    fill: int = 0
    mean_fill: bool = False
    border_px: int = 0                    # widen the zeroed border band to this many px
    mask_residual_markers: bool = False
    frontal_only: bool = True             # medial/lateral only mean something on a frontal

    def describe(self) -> str:
        bits = []
        if self.zero:
            bits.append("zero " + "+".join(self.zero))
        if self.keep:
            bits.append("keep only " + "+".join(self.keep))
        if self.border_px:
            bits.append(f"border widened to {self.border_px} px")
        if self.mask_residual_markers:
            bits.append("residual saturated blobs blanked")
        if self.mean_fill:
            bits.append("mean fill")
        return "; ".join(bits) or "unperturbed"


IDENTITY = PixelOp("baseline")


def _residual_marker_mask(img: np.ndarray, *, sat: int, lo: int, hi_frac: float) -> np.ndarray:
    """Saturated, small, non-largest connected components: the residual-marker signature.

    Exactly the criterion ``src.crop_qa.residual_marker_scan`` counts with, so the pixels
    blanked here are the pixels that audit calls a surviving marker. It is an UPPER bound:
    saturated bone edges share the signature, which is why B6 is reported as a bound.
    """
    from scipy import ndimage as ndi

    hot = np.asarray(img) >= int(sat)
    if not hot.any():
        return np.zeros_like(hot, dtype=bool)
    lab, n = ndi.label(hot)
    if n == 0:
        return np.zeros_like(hot, dtype=bool)
    sizes = ndi.sum(hot, lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    area = float(img.size)
    out = np.zeros_like(hot, dtype=bool)
    for k in range(1, n + 1):
        if k == biggest:
            continue
        if int(lo) <= float(sizes[k - 1]) <= float(hi_frac) * area:
            out |= (lab == k)
    return ndi.binary_dilation(out, iterations=2)


def apply_pixel_op(img_u8: np.ndarray, op: PixelOp, masks: dict[str, np.ndarray],
                   *, view: str, marker_params: dict) -> np.ndarray:
    """Apply one :class:`PixelOp` to one (H, W) uint8 crop. Pure; returns a new array."""
    a = np.asarray(img_u8).copy()
    if op.frontal_only and view != "frontal" and (op.zero or op.keep):
        return a                       # medial/lateral/PF are undefined off the frontal view
    sel = np.zeros(a.shape, dtype=bool)
    for r in op.zero:
        sel |= masks[r]
    if op.keep:
        keep = np.zeros(a.shape, dtype=bool)
        for r in op.keep:
            keep |= masks[r]
        sel |= ~keep
    if op.border_px:
        b = int(op.border_px)
        wide = np.zeros(a.shape, dtype=bool)
        wide[:b, :] = wide[-b:, :] = True
        wide[:, :b] = wide[:, -b:] = True
        sel |= wide
    if op.mask_residual_markers:
        sel |= _residual_marker_mask(a, **marker_params)
    if not sel.any():
        return a
    fill = int(round(float(a[~sel].mean()))) if (op.mean_fill and (~sel).any()) else int(op.fill)
    a[sel] = np.uint8(np.clip(fill, 0, 255))
    return a


class PerturbedViewDataset(PatientViewDataset):
    """``PatientViewDataset`` with one :class:`PixelOp` applied to every real crop.

    Subclassed rather than patched so ``src/train_model.py`` is untouched and the label,
    clinical-design and padding logic stay bit-identical to the sealed read.
    """

    def configure(self, op: PixelOp, masks: dict[str, np.ndarray], marker_params: dict,
                  view_names: list[str]):
        self.op = op
        self.masks = masks
        self.marker_params = marker_params
        self.view_names = list(view_names)
        return self

    def __getitem__(self, i):
        item = super().__getitem__(i)
        if getattr(self, "op", IDENTITY) is IDENTITY or self.op.name == "baseline":
            return item
        import torch

        imgs = item["images"].numpy()
        vids = item["view_id"].numpy()
        keep = item["mask"].numpy()
        out = imgs.copy()
        for j in range(imgs.shape[0]):
            if not keep[j]:
                continue
            out[j] = apply_pixel_op(imgs[j], self.op, self.masks,
                                    view=self.view_names[int(vids[j])],
                                    marker_params=self.marker_params)
        item["images"] = torch.from_numpy(out)
        return item


# =========================================================================== #
# 3. ONE ARM, SCORED LOCALLY UNDER ANY PIXEL OPERATION                         #
# =========================================================================== #
def five_year_risk(hazards: np.ndarray, recal: dict | None,
                   horizon: float = HORIZON_DAYS) -> np.ndarray:
    """Predicted risk at the horizon, with the frozen validation recalibration applied.

    ``src/score_test.py`` writes RAW ensemble hazards; the frozen recalibration is applied
    downstream in ``src/eval_models.trained_arm_scores``. This reproduces that, so the
    numbers here are on the same scale as the manuscript's predicted risks.
    """
    p = risk_at_horizon(hazards, float(horizon), edges=EDGES)
    if recal is None:
        return np.asarray(p, dtype=float)
    return np.asarray(apply_recalibration(p, recal[str(float(horizon))]), dtype=float)


def published_risk(ctx: InterpContext, arm: str, pids: np.ndarray,
                   horizon: float = HORIZON_DAYS) -> np.ndarray:
    """The MANUSCRIPT's per-patient 5-year risk for one arm, read from the published npz.

    Read-only. This exists for exactly one purpose: **case selection**. Local float32 and
    the published Colab float16-autocast run agree on aggregate discrimination to 6.7e-05
    but on individual risks only to about 8e-03, so thresholding a locally recomputed risk
    can put a patient in a different TP/FP/TN/FN cell than the published risk table does.
    Every example case in this module is therefore assigned its cell from THIS number,
    while every delta is computed locally against a locally computed baseline.

    The npz holds RAW ensemble hazards - ``src/score_test.py`` never calls
    ``apply_recalibration`` despite what its docstring says; the frozen validation
    recalibration is applied downstream in ``eval_models.trained_arm_scores``. It is
    applied here, so the returned risk is on the same scale as every published table.
    """
    path = ctx.cfg.path(ctx.cfg["paths"]["cohort_dir"]) / f"test_hazards_{arm}.npz"
    with np.load(path, allow_pickle=False) as z:
        pub_pids = np.asarray(z["empi_anon"]).astype(str)
        hz = np.asarray(z["hazards"], dtype=float)
        assert str(z["arm"].item()) == arm and str(z["split"].item()) == SEALED_SPLIT
    recal = dict(ctx.train_arms["arms"][arm]["recalibration"])
    r = five_year_risk(hz, recal, horizon)
    pos = pd.Index(pub_pids).get_indexer(np.asarray(pids).astype(str))
    assert (pos >= 0).all(), f"{path} does not carry every locally scored patient"
    return r[pos]


@dataclass
class Prediction:
    """One condition's local prediction for one arm. Baselines and perturbations share it."""

    condition: str
    arm: str
    pids: np.ndarray
    hazards: np.ndarray
    risk: np.ndarray
    rank: np.ndarray
    time: np.ndarray
    event: np.ndarray
    seconds: float = 0.0


class ArmRunner:
    """Loads one frozen arm's five seeds once, then scores it under any :class:`PixelOp`.

    Every condition, INCLUDING the unperturbed baseline, goes through :meth:`score`. That
    is the whole point: the baseline is a local float32 number produced by the same code
    on the same device in the same process, so a difference between two conditions cannot
    be a float16-versus-float32 artefact.
    """

    def __init__(self, ctx: InterpContext, arm: str):
        import torch

        self.ctx = ctx
        self.arm = arm
        self.summary = ctx.train_arms["arms"][arm]
        self.design = str(self.summary["design"])
        self.spec_views = list(self.summary["views"])
        self.seeds = [int(s) for s in self.summary["seeds"]]
        self.recal = dict(self.summary["recalibration"])
        self.masks = region_masks(ctx.settings.out_size, ctx.settings.border_px)
        p = ctx.cfg["preprocess"]
        self.marker_params = {"sat": int(p.get("marker_sat_level", 250)),
                              "lo": int(p.get("marker_min_px", 20)),
                              "hi_frac": float(p.get("marker_max_area_frac", 0.01))}
        self.nets = []
        for seed in self.seeds:
            net, _ = load_seed_model(
                arm, seed,
                spec={"mode": self.summary["mode"], "arch": self.summary["arch"],
                      "views": self.spec_views, "design": self.design},
                n_clinical=int(self.summary["n_clinical"]), settings=ctx.settings,
                device=ctx.device, contract_hash=ctx.settings.contract_hash())
            self.nets.append(net)
        ctx.log.info("%s: %d frozen seeds loaded (%s, views %s)", arm, len(self.nets),
                     self.summary["mode"], self.spec_views)

    # -- data ------------------------------------------------------------------ #
    def dataset(self, op: PixelOp, views_allowed=None) -> PerturbedViewDataset:
        ctx = self.ctx
        ds = PerturbedViewDataset(
            SEALED_SPLIT, ctx.frames[self.design], ctx.designs[self.design], train=False,
            npy_path=ctx.npy, index=ctx.index, clin_stats=ctx.stats[self.design],
            views=list(ctx.settings.views),
            views_allowed=list(self.spec_views if views_allowed is None else views_allowed),
            max_elems=ctx.max_elems, design=self.design, out_size=ctx.settings.out_size,
            aug=None, border_px=ctx.settings.border_px, allow_sealed=True)
        return ds.configure(op, self.masks, self.marker_params, list(ctx.settings.views))

    # -- prediction ------------------------------------------------------------- #
    def score(self, op: PixelOp, nets=None, condition: str | None = None,
              views_allowed=None) -> Prediction:
        """Ensemble hazards for one condition. Nothing is fitted; five seeds are averaged.

        ``nets`` overrides the frozen ensemble and exists only for the sanity checks,
        where a single deliberately corrupted network is scored. ``views_allowed`` withholds
        whole views from a multi-view arm, which is the view-ablation experiment.
        """
        import torch
        from torch.utils.data import DataLoader

        t0 = time.time()
        ds = self.dataset(op, views_allowed=views_allowed)
        per_seed = []
        for net in (self.nets if nets is None else list(nets)):
            loader = DataLoader(ds, batch_size=self.ctx.runtime.micro_batch_size,
                                shuffle=False, num_workers=0)
            out = np.zeros((len(ds), N_INTERVALS), dtype=np.float64)
            net.eval()
            with torch.no_grad():
                for batch in loader:
                    b = {k: (v.to(self.ctx.device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                    logits = net(b)
                    out[b["idx"].cpu().numpy()] = torch.sigmoid(
                        logits.float()).cpu().numpy()
            per_seed.append(out)
        ens = average_hazard(per_seed)
        return Prediction(condition=condition or op.name, arm=self.arm,
                          pids=np.asarray(ds.pids, dtype=object).astype("U"),
                          hazards=ens, risk=five_year_risk(ens, self.recal),
                          rank=risk_score(ens), time=ds.time.astype(float),
                          event=ds.event.astype(int), seconds=time.time() - t0)


# =========================================================================== #
# 4. DISCRIMINATION, ON THE SAME IPCW MACHINERY THE PAPER USES                 #
# =========================================================================== #
def ipcw_auroc(pred: Prediction, ctx: InterpContext, horizon: float = HORIZON_DAYS,
               score: str = "risk") -> float:
    y, w = ipcw_labels_weights(pred.time, pred.event, float(horizon),
                               ctx.contracts.g_grid, ctx.contracts.g_vals)
    s = pred.risk if score == "risk" else pred.rank
    return float(ipcw_auc(y, w, s))


def restrict(pred: Prediction, pids) -> Prediction:
    """One prediction on a subset of its patients, in the order given. Nothing is refitted."""
    want = np.asarray(pids).astype(str)
    pos = pd.Index(pred.pids).get_indexer(want)
    assert (pos >= 0).all(), "restrict() was given a patient this condition did not score"
    return Prediction(condition=pred.condition, arm=pred.arm, pids=want,
                      hazards=pred.hazards[pos], risk=pred.risk[pos], rank=pred.rank[pos],
                      time=pred.time[pos], event=pred.event[pos], seconds=pred.seconds)


def paired_auroc_delta(base: Prediction, other: Prediction, ctx: InterpContext,
                       *, n_boot: int = 2000, seed: int = 20250720,
                       horizon: float = HORIZON_DAYS) -> dict:
    """Paired patient-level bootstrap of AUROC(perturbed) - AUROC(baseline).

    Paired by construction: both arms of the contrast are the SAME patients in the SAME
    row order (the two Predictions come from the same dataset), and one resample of
    patient positions is applied to both, exactly as ``eval_models.bootstrap_draw`` does
    for the published contrasts.
    """
    assert np.array_equal(base.pids, other.pids), "paired contrast needs identical rosters"
    y, w = ipcw_labels_weights(base.time, base.event, float(horizon),
                               ctx.contracts.g_grid, ctx.contracts.g_vals)
    a0, a1 = ipcw_auc(y, w, base.risk), ipcw_auc(y, w, other.risk)
    rng = np.random.default_rng(int(seed))
    n = base.pids.size
    draw = rng.integers(0, n, size=(int(n_boot), n))
    d = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        i = draw[b]
        yy, ww = y[i], w[i]
        if not ((yy == 1).any() and (yy == 0).any()):
            continue
        d[b] = ipcw_auc(yy, ww, other.risk[i]) - ipcw_auc(yy, ww, base.risk[i])
    ok = np.isfinite(d)
    lo, hi = (np.nanpercentile(d[ok], 2.5), np.nanpercentile(d[ok], 97.5)) if ok.sum() else (np.nan, np.nan)
    p = float(2.0 * min((d[ok] <= 0).mean(), (d[ok] >= 0).mean())) if ok.sum() else float("nan")
    return {"auc_baseline": float(a0), "auc_condition": float(a1),
            "delta_auc": float(a1 - a0), "delta_lo": float(lo), "delta_hi": float(hi),
            "p_boot": min(1.0, p), "n_boot_ok": int(ok.sum())}


# =========================================================================== #
# 5. ATTRIBUTION                                                               #
# =========================================================================== #
def risk_target(logits, horizon: float = HORIZON_DAYS):
    """Differentiable 5-year risk from interval logits: the torch twin of
    ``train_model.risk_at_horizon``, so attribution explains the reported quantity.

    The frozen recalibration is a strictly monotone scalar map applied AFTER ensembling,
    so it rescales every attribution by a common factor and cannot move mass between
    regions. Attribution therefore targets the un-recalibrated risk.
    """
    import torch

    h = torch.sigmoid(logits)
    k = int(np.searchsorted(EDGES[1:], float(horizon), side="right"))
    if k >= h.shape[1]:
        return 1.0 - torch.cumprod(1.0 - h, dim=1)[:, -1]
    S = torch.cumprod(1.0 - h[:, :k], dim=1)[:, -1] if k > 0 else torch.ones_like(h[:, 0])
    frac = float((float(horizon) - EDGES[k]) / (EDGES[k + 1] - EDGES[k]))
    return 1.0 - S * torch.pow(1.0 - h[:, k], frac)


def _real_slot_index(mask):
    """(b, e) of every real slot, in the exact order ``embed_images`` feeds the encoder.

    THIS IS THE REINDEXING TRAP. With ``encode_masked_only=True`` the encoder never sees a
    padded slot: ``embed_images`` flattens (B, E) to B*E, selects the masked positions with
    ``nonzero`` and runs the encoder on that gathered tensor only. A hook on the encoder
    therefore returns ``n_real`` feature maps, NOT ``B*E`` of them, and row r of that
    tensor belongs to slot (sel[r] // E, sel[r] % E) - not to slot r. Mapping CAMs back by
    position instead of through ``sel`` silently attributes one patient's map to another
    whenever any patient in the batch has fewer crops than the padding width.
    """
    B, E = mask.shape
    sel = mask.reshape(B * E).nonzero(as_tuple=True)[0]
    return sel, (sel // E), (sel % E)


def gradcam(net, batch, device, *, layer: str = CAM_LAYER, horizon: float = HORIZON_DAYS):
    """Grad-CAM at the encoder's last normalisation layer.

    Returns ``(sel_b, sel_e, cam)`` where ``cam`` is (n_real, h, w), one map per REAL view
    slot, aligned through :func:`_real_slot_index`. Channel weights are the spatial mean of
    the gradient of the 5-year risk with respect to the activations (Selvaraju et al.), the
    weighted sum is rectified, and the map is left un-normalised so the caller can decide.
    """
    import torch

    mod = net
    for part in layer.split("."):
        mod = getattr(mod, part)
    store = {}

    def fwd(_m, _i, o):
        store["act"] = o
        o.retain_grad()

    h = mod.register_forward_hook(fwd)
    try:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        b["images"] = b["images"].float()
        net.zero_grad(set_to_none=True)
        logits = net(b)
        risk_target(logits, horizon).sum().backward()
        act = store["act"]
        grad = act.grad
        assert grad is not None, "no gradient reached the CAM layer"
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * act).sum(dim=1))
    finally:
        h.remove()
    sel, sb, se = _real_slot_index(b["mask"])
    assert cam.shape[0] == sel.numel(), (
        f"the CAM layer produced {cam.shape[0]} maps for {sel.numel()} real slots; the "
        f"encode_masked_only gather was not accounted for")
    return sb.cpu().numpy(), se.cpu().numpy(), cam.detach().float().cpu().numpy()


def integrated_gradients(net, batch, device, *, steps: int = 32,
                         horizon: float = HORIZON_DAYS):
    """Integrated gradients on the raw uint8 pixel scale, zero baseline.

    A zero image is the RIGHT baseline here rather than a convenience: 0 is exactly what
    the pipeline writes into the masked border band and into out-of-bounds padding, so the
    all-zero image is inside the model's input distribution. ``to_model_input``
    (``train_model.py:852``) is deliberately out-of-place, so the graph survives.

    **Consequence for the border band, stated precisely.** ``ig = (x - 0) * mean_grad``, so
    a pixel that is exactly zero receives exactly zero attribution - by construction,
    wherever the band is exactly zero. That is an identity, not a finding, and it does NOT
    make the band's IG mass exactly zero over the cohort: the band is not exactly zero on
    11.9% of test crops, so 18 of the 230 IG crops carry a non-zero band mass (max
    4.90e-04) and the cohort mean is 5.6e-06 - essentially zero, not zero. Say
    "essentially zero"; the border finding is the occlusion result, not this one.

    Returns ``(sel_b, sel_e, ig)`` with ``ig`` (n_real, H, W) signed attributions.
    """
    import torch

    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    x = b["images"].float()
    total = torch.zeros_like(x)
    for s in range(int(steps)):
        alpha = (s + 0.5) / float(steps)                 # midpoint rule
        xi = (alpha * x).detach().requires_grad_(True)
        bb = dict(b); bb["images"] = xi
        net.zero_grad(set_to_none=True)
        risk_target(net(bb), horizon).sum().backward()
        total = total + xi.grad.detach()
    ig = (x * total / float(steps))                      # (x - 0) * mean gradient
    sel, sb, se = _real_slot_index(b["mask"])
    flat = ig.reshape(-1, ig.shape[-2], ig.shape[-1]).index_select(0, sel)
    return sb.cpu().numpy(), se.cpu().numpy(), flat.detach().float().cpu().numpy()


def attention_weights(net, batch, device):
    """Per-view attention from ``forward(..., return_attention=True)``.

    Already computed inside ``MaskedAttentionPool`` on every forward pass and never
    persisted. Padded slots are -inf before the softmax, so the weights over the real
    slots of one patient sum to 1 and are directly comparable across patients.
    """
    import torch

    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        _, attn = net(b, return_attention=True)
    return attn.float().cpu().numpy(), b["view_id"].cpu().numpy(), b["mask"].cpu().numpy()


def upsample_cam(cam: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    """(h, w) -> (size, size) bilinear. Grad-CAM's native cell is 32 px at this input."""
    import torch

    t = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                         align_corners=False)
    return up[0, 0].numpy()


def attribution_fractions(a: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of total absolute attribution mass falling in each region."""
    m = np.abs(np.asarray(a, dtype=np.float64))
    tot = float(m.sum())
    if tot <= 0:
        return {k: float("nan") for k in masks}
    return {k: float(m[v].sum() / tot) for k, v in masks.items()}


# =========================================================================== #
# 6. SANITY-CHECK MODELS (Adebayo et al. parameter randomisation)              #
# =========================================================================== #
# The encoder's stages, output-side last. Cascading randomisation (Adebayo et al. 2018)
# re-initialises the top of the network first and then works downwards, so every scope
# below also randomises everything listed after it, plus the head.
ENCODER_STAGES = ["conv0", "norm0", "denseblock1", "transition1", "denseblock2",
                  "transition2", "denseblock3", "transition3", "denseblock4", "norm5"]


def randomized_net(ctx: InterpContext, runner: ArmRunner, *, scope: str, seed: int = 0):
    """A copy of one frozen seed with part or all of its parameters re-randomised.

    ``scope`` is ``head`` (everything that is not the encoder: survival head, projection,
    view embedding, attention pool, image LayerNorm), an encoder stage name such as
    ``denseblock4`` (that stage and everything above it, plus the head), or ``all``.
    This is the parameter-randomisation half of the Adebayo sanity check: an attribution
    method that is really reading the learned function must degrade as the function is
    destroyed, and one that does not is an edge detector.
    """
    import torch

    torch.manual_seed(int(seed))
    fresh = SurvivalFusionNet(
        n_intervals=N_INTERVALS, n_clinical=int(runner.summary["n_clinical"]),
        n_views=len(ctx.settings.views), mode=str(runner.summary["mode"]),
        arch=str(runner.summary["arch"]), pretrained=False,
        view_emb_dim=ctx.settings.view_emb_dim).to(ctx.device).eval()
    if scope == "all":
        return fresh
    assert scope == "head" or scope in ENCODER_STAGES, f"unknown scope {scope!r}"
    take_stages = ([] if scope == "head"
                   else ENCODER_STAGES[ENCODER_STAGES.index(scope):])
    net = copy.deepcopy(runner.nets[0]).to(ctx.device).eval()
    src_sd, dst_sd = fresh.state_dict(), net.state_dict()
    hit = 0
    for k in dst_sd:
        if not k.startswith("encoder."):
            hit += 1
            dst_sd[k] = src_sd[k].clone()
        elif any(k.startswith(f"encoder.features.{s}.") or k == f"encoder.features.{s}"
                 or k.startswith(f"encoder.features.{s}") for s in take_stages):
            hit += 1
            dst_sd[k] = src_sd[k].clone()
    assert hit > 0, f"randomisation scope {scope!r} matched no parameter"
    net.load_state_dict(dst_sd)
    return net


# =========================================================================== #
# 7. STAGES                                                                    #
# =========================================================================== #
def limb_landmarks(img: np.ndarray, border_px: int = BORDER_PX) -> tuple[float, float]:
    """(centre column, width) of the imaged limb in one crop. Model-free.

    Uses the project's own Otsu threshold (``src.preprocess_images.otsu_threshold``) to
    separate limb from unexposed background, then takes the widest contiguous run of
    columns whose limb count exceeds a fifth of the maximum, so a collimator edge or a
    stray marker cannot drag the centre.
    """
    from src.preprocess_images import otsu_threshold

    b = int(border_px)
    inner = np.asarray(img, dtype=np.float32)[b:-b, b:-b] / 255.0
    thr = otsu_threshold(inner)
    prof = (inner > thr).sum(axis=0).astype(float)
    if prof.max() <= 0:
        return float(img.shape[1] / 2), float(inner.shape[1])
    on = prof > 0.2 * prof.max()
    best, cur = (0, 0, 0), None
    for i, v in enumerate(on):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[0]:
                best = (i - cur, cur, i)
            cur = None
    if cur is not None and len(on) - cur > best[0]:
        best = (len(on) - cur, cur, len(on))
    _, c0, c1 = best
    return float(0.5 * (c0 + c1) + b), float(c1 - c0)


def registered_joint_mask(img: np.ndarray, *, half_h: int = 96, half_w: int = 144,
                          border_px: int = BORDER_PX) -> np.ndarray:
    """A per-image tibiofemoral ROI: the fixed box RE-CENTRED on this crop's own anatomy.

    The fixed :data:`JOINT_BOX` is pre-declared and needs no detector, but it assumes the
    knee sits at the crop centre, and the crops are only approximately centred - the joint
    line lands inside the fixed rows on 91% of frontal crops while the limb centre column
    varies far more. This is the sensitivity analysis: same box size, placed on the joint
    line and limb centre this image actually has, clipped to the unmasked interior.
    """
    n = int(img.shape[0])
    b = int(border_px)
    jr = detect_joint_row(img)
    cc, _ = limb_landmarks(img, b)
    r0, r1 = int(np.clip(jr - half_h, b, n - b)), int(np.clip(jr + half_h, b, n - b))
    c0, c1 = int(np.clip(cc - half_w, b, n - b)), int(np.clip(cc + half_w, b, n - b))
    m = np.zeros((n, n), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def stage_registered(ctx: InterpContext, arm: str = PRIMARY_ARM) -> pd.DataFrame:
    """B2 sensitivity: the joint fraction under a PER-IMAGE anatomically registered ROI.

    Re-scores the Grad-CAM maps that ``--stage attribution`` already wrote, so it costs one
    pass over an npz and no forward passes at all. Enrichment is computed per image against
    that image's own ROI area, because clipping at the border band makes the registered box
    slightly smaller on some crops.
    """
    z = np.load(ctx.out_dir / f"attribution_maps_{arm}.npz", allow_pickle=False)
    assert "image_row" in z and "gradcam_native" in z, (
        "re-run --stage attribution: this map file predates the native-resolution archive")
    cams, rows_img = z["gradcam_native"], z["image_row"]
    fixed = region_masks(ctx.settings.out_size, ctx.settings.border_px)["joint"]
    out = []
    for i in range(cams.shape[0]):
        img = np.asarray(ctx.images[int(rows_img[i])])
        reg = registered_joint_mask(img, border_px=ctx.settings.border_px)
        a = np.abs(upsample_cam(cams[i], ctx.settings.out_size).astype(np.float64))
        tot = float(a.sum())
        if tot <= 0:
            continue
        ar = float(reg.sum()) / float(reg.size)
        out.append({"arm": arm, "image_row": int(rows_img[i]),
                    "empi_anon": str(z["empi_anon"][i]),
                    "joint_row": int(detect_joint_row(img)),
                    "limb_centre_col": float(limb_landmarks(img, ctx.settings.border_px)[0]),
                    "roi_area_fraction": ar,
                    "frac_joint_registered": float(a[reg].sum() / tot),
                    "frac_joint_fixed": float(a[fixed].sum() / tot),
                    "iou_registered_vs_fixed": float((reg & fixed).sum() / (reg | fixed).sum())})
    df = pd.DataFrame(out)
    df["enrichment_registered"] = df["frac_joint_registered"] / df["roi_area_fraction"]
    df["enrichment_fixed"] = df["frac_joint_fixed"] / 0.25
    df.to_csv(ctx.table_dir / "interp_attribution_registered.csv", index=False)
    ctx.log.info("registered ROI (n=%d): joint fraction %.4f (fixed box %.4f); mean ROI area "
                 "%.4f; enrichment %.3f vs %.3f; mean IoU with the fixed box %.3f",
                 len(df), df["frac_joint_registered"].mean(), df["frac_joint_fixed"].mean(),
                 df["roi_area_fraction"].mean(), df["enrichment_registered"].mean(),
                 df["enrichment_fixed"].mean(), df["iou_registered_vs_fixed"].mean())
    return df


def _iter_batches(ds, batch_size: int):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)


PAIRED_ATTENTION_COLUMNS = ["arm", "comparison", "view_a", "view_b", "view_c", "n_patients",
                            "weight_a_mean", "weight_a_sd", "weight_b_mean", "weight_b_sd",
                            "weight_c_mean", "weight_c_sd", "pct_b_outweighs_a", "note"]


def paired_attention_rows(arm: str, views, share: np.ndarray, n_slots: np.ndarray, *,
                          min_patients: int = 10) -> list[dict]:
    """The rows of ``interp_attention_paired.csv``, from the arrays the npz already holds.

    Pure and re-runnable: ``share`` and ``n_slots`` are exactly the two arrays
    :func:`stage_attention` writes into ``outputs/interpretability/attention_{arm}.npz``,
    so the table can be rebuilt without re-scoring an image.

    **Every column holds the quantity its name promises, and that is the point of this
    function.** The first version of this table wrote the LATERAL MEAN into ``weight_a_sd``
    on the "all three views present" row (0.4580 for m3_image, 0.4920 for m4_fusion),
    because the three-view summary was squeezed into a schema built for a two-view
    contrast. A figure agent reading that column would have drawn a 0.458 error bar on a
    weight of 0.213. The three-view summary now owns ``view_c`` / ``weight_c_mean`` /
    ``weight_c_sd``, every ``*_sd`` is a standard deviation (``ddof=1``) of the column
    beside it, and the pairwise rows leave the ``_c`` columns empty.

    Two row types, and they have different denominators:

    * ``pairwise_one_crop_each`` - patients holding EXACTLY one crop of each of two views
      and nothing else, so the shares are per-slot weights and a patient with two frontals
      cannot inflate the frontal total. ``weight_a_mean + weight_b_mean == 1``.
    * ``all_three_views_present`` - patients with at least one crop of all three views.
      ``weight_a_mean + weight_b_mean + weight_c_mean == 1``.
    """
    views = list(views)
    share = np.asarray(share, dtype=float)
    n_slots = np.asarray(n_slots, dtype=float)
    out: list[dict] = []

    def _sd(x):
        return float(np.std(x, ddof=1)) if x.size > 1 else np.nan

    for a_i, b_i in ((0, 1), (0, 2), (1, 2)):
        m = ((np.abs(n_slots[:, a_i] - 1) < 1e-9) & (np.abs(n_slots[:, b_i] - 1) < 1e-9)
             & (np.abs(n_slots.sum(axis=1) - 2) < 1e-9))
        if m.sum() < min_patients:
            continue
        out.append({"arm": arm, "comparison": "pairwise_one_crop_each",
                    "view_a": views[a_i], "view_b": views[b_i], "view_c": "",
                    "n_patients": int(m.sum()),
                    "weight_a_mean": float(share[m, a_i].mean()),
                    "weight_a_sd": _sd(share[m, a_i]),
                    "weight_b_mean": float(share[m, b_i].mean()),
                    "weight_b_sd": _sd(share[m, b_i]),
                    "weight_c_mean": np.nan, "weight_c_sd": np.nan,
                    "pct_b_outweighs_a": float((share[m, b_i] > share[m, a_i]).mean() * 100),
                    "note": (f"patients holding exactly one {views[a_i]} crop and one "
                             f"{views[b_i]} crop and nothing else; the two weights sum to 1 "
                             f"within each patient, so weight_a_sd == weight_b_sd by "
                             f"construction. Every *_sd is a standard deviation (ddof=1).")})
    all3 = (n_slots > 0).all(axis=1)
    if all3.sum() >= min_patients and len(views) >= 3:
        out.append({"arm": arm, "comparison": "all_three_views_present",
                    "view_a": views[0], "view_b": views[1], "view_c": views[2],
                    "n_patients": int(all3.sum()),
                    "weight_a_mean": float(share[all3, 0].mean()),
                    "weight_a_sd": _sd(share[all3, 0]),
                    "weight_b_mean": float(share[all3, 1].mean()),
                    "weight_b_sd": _sd(share[all3, 1]),
                    "weight_c_mean": float(share[all3, 2].mean()),
                    "weight_c_sd": _sd(share[all3, 2]),
                    "pct_b_outweighs_a": float(
                        (share[all3, 1] > share[all3, 0]).mean() * 100),
                    "note": ("patients with at least one crop of all three views; the three "
                             "means sum to 1 within each patient. Quote all three to 3 dp "
                             "from this row - rounding them independently from elsewhere is "
                             "how a set that sums to 1.000 becomes 1.002.")})
    return out


def by_view_attention_rows(arm: str, views, share: np.ndarray, n_slots: np.ndarray) -> list[dict]:
    """The rows of ``interp_attention_by_view.csv``, from the arrays the npz already holds.

    Pure, for the same reason :func:`paired_attention_rows` is.

    ``n_patients_with_2plus_distinct_views`` counts patients who hold **this** view *and*
    at least two DISTINCT views. It is not the number of patients with more than one view
    and it is not the number of patients with more than one crop; those are three
    different counts on this cohort (316 / 316 / 373 on the 740-patient multi-view roster)
    and the ``note`` column says so on every row, because conflating them is how "315
    patients who have more than one view" got written down.
    """
    views = list(views)
    share = np.asarray(share, dtype=float)
    n_slots = np.asarray(n_slots, dtype=float)
    present = n_slots > 0
    multi = present.sum(axis=1) > 1
    note = ("n_patients_with_2plus_distinct_views counts patients holding THIS view and at "
            "least two distinct views; it is not a count of patients with >1 crop (a "
            "patient with two frontals and nothing else is excluded) and it is not the "
            "view-availability subgroup of outputs/tables/test_subgroups.csv, which is "
            "labelled from the selected study's view_set before preprocessing.")
    out: list[dict] = []
    for vi, v in enumerate(views):
        has = present[:, vi]
        out.append({
            "arm": arm, "view": v,
            "n_patients_with_view": int(has.sum()),
            "n_slots_mean": float(n_slots[has, vi].mean()) if has.any() else np.nan,
            "attention_share_mean": float(share[has, vi].mean()) if has.any() else np.nan,
            "attention_share_sd": float(share[has, vi].std(ddof=1)) if has.sum() > 1 else np.nan,
            "attention_share_median": float(np.median(share[has, vi])) if has.any() else np.nan,
            "attention_per_slot_mean": float(
                (share[has, vi] / np.maximum(n_slots[has, vi], 1)).mean()) if has.any() else np.nan,
            "n_patients_with_2plus_distinct_views": int((has & multi).sum()),
            "attention_share_mean_2plus_distinct_views": float(
                share[has & multi, vi].mean()) if (has & multi).any() else np.nan,
            "note": note,
        })
    return out


def stage_attention(ctx: InterpContext, arms=MULTIVIEW_ARMS, batch_size: int = 8) -> pd.DataFrame:
    """B7. Per-view attention weights for the multi-view arms, persisted for the first time."""
    views = list(ctx.settings.views)
    rows: list[dict] = []
    paired: list[dict] = []
    for arm in arms:
        runner = ArmRunner(ctx, arm)
        ds = runner.dataset(IDENTITY)
        acc = np.zeros((len(ds), len(views)))
        cnt = np.zeros((len(ds), len(views)))
        for si, net in enumerate(runner.nets):
            for batch in _iter_batches(ds, batch_size):
                a, vid, msk = attention_weights(net, batch, ctx.device)
                idx = batch["idx"].numpy()
                for r in range(a.shape[0]):
                    for e in range(a.shape[1]):
                        if msk[r, e]:
                            acc[idx[r], vid[r, e]] += a[r, e]
                            cnt[idx[r], vid[r, e]] += 1
            ctx.log.info("  %s attention: seed %d/%d done", arm, si + 1, len(runner.nets))
        share = acc / max(len(runner.nets), 1)            # per-patient share summing to 1
        present = cnt > 0
        np.savez(ctx.out_dir / f"attention_{arm}.npz",
                 empi_anon=np.asarray(ds.pids, dtype=object).astype("U"),
                 share=share, n_slots=cnt / max(len(runner.nets), 1),
                 views=np.asarray(views, dtype=object).astype("U"),
                 event=ds.event.astype(int), time=ds.time.astype(float))
        rows.extend(by_view_attention_rows(arm, views, share, cnt / max(len(runner.nets), 1)))
        # A within-patient contrast on the cleanest possible comparison: patients holding
        # EXACTLY one crop of each of two views, so the shares are per-slot weights and a
        # patient with two frontals cannot inflate the frontal total. Built by the pure
        # helper so the table is rebuildable from the npz alone.
        for r in paired_attention_rows(arm, views, share, cnt / max(len(runner.nets), 1)):
            paired.append(r)
            ctx.log.info("  %s %s: %s %.3f (sd %.3f) / %s %.3f (sd %.3f)%s, n=%d", arm,
                         r["comparison"], r["view_a"], r["weight_a_mean"], r["weight_a_sd"],
                         r["view_b"], r["weight_b_mean"], r["weight_b_sd"],
                         (f" / {r['view_c']} {r['weight_c_mean']:.3f} "
                          f"(sd {r['weight_c_sd']:.3f})") if r["view_c"] else "",
                         r["n_patients"])
        del runner
    df = pd.DataFrame(rows)
    df.to_csv(ctx.table_dir / "interp_attention_by_view.csv", index=False)
    pd.DataFrame(paired, columns=PAIRED_ATTENTION_COLUMNS).to_csv(
        ctx.table_dir / "interp_attention_paired.csv", index=False)
    ctx.log.info("wrote %s and %s", ctx.table_dir / "interp_attention_by_view.csv",
                 ctx.table_dir / "interp_attention_paired.csv")
    return df


VIEW_ABLATIONS = {"all_views": ["frontal", "lateral", "sunrise"],
                  "frontal_only": ["frontal"],
                  "drop_sunrise": ["frontal", "lateral"],
                  "non_frontal_only": ["lateral", "sunrise"]}

#: Rides on every row of ``interp_view_ablation.csv``. ``n_patients_common`` is a PAIRED
#: DENOMINATOR, not a cohort description, and it has been misread as one: the drafted
#: Results text said "the 315 patients who have more than one view", which is a different
#: and wrong statement. Four counts on this cohort are close together and all different.
VIEW_ABLATION_NOTE = (
    "n_patients_common is a PAIRED-ANALYSIS DENOMINATOR, not a count of multi-view "
    "patients: it is the intersection of the rosters of all four view conditions, i.e. the "
    "patients every condition in this contrast can score, which requires at least one "
    "frontal crop (for frontal_only) and at least one non-frontal crop (for "
    "non_frontal_only). On this test split that is 315. The neighbouring counts are "
    "different quantities and must not be substituted for it: 316 patients hold two or "
    "more DISTINCT views (315 of them also hold a frontal, 1 does not); 321 of the 740 "
    "scored patients contribute at least one non-frontal crop; 6 have no frontal crop at "
    "all; and the 322 in the 'Multiple views' row of outputs/tables/test_subgroups.csv is a "
    "pre-preprocessing label (final_cohort.view_set != 'frontal') that additionally counts "
    "one patient whose lateral crop did not survive preprocessing. n_patients_scored is "
    "the roster of that single condition; every AUROC and Δ in this table is computed on "
    "n_patients_common.")


def stage_view_ablation(ctx: InterpContext, arms=MULTIVIEW_ARMS, *,
                        n_boot: int = 2000, reference_arm: str = PRIMARY_ARM) -> pd.DataFrame:
    """B7's companion: withhold whole views from a multi-view model and see what moves.

    The attention weights say where the aggregator LOOKS. This says what the extra views
    are WORTH, using the same frozen network: the multi-view arm is re-scored with the
    lateral and sunrise crops withheld at input, so nothing is retrained and the contrast
    is within one model rather than across two arms.

    Every contrast is restricted to the patients ALL of its conditions can score - a
    frontal-only input set cannot score the six patients who have no frontal crop - and is
    paired on that common roster.
    """
    rows = []
    for arm in arms:
        runner = ArmRunner(ctx, arm)
        preds = {}
        for name, views in VIEW_ABLATIONS.items():
            preds[name] = runner.score(IDENTITY, condition=name, views_allowed=views)
            ctx.log.info("  %s / %-17s n=%d AUROC %.4f mean risk %.4f [%.0f s]",
                         arm, name, preds[name].pids.size,
                         ipcw_auroc(preds[name], ctx), float(preds[name].risk.mean()),
                         preds[name].seconds)
        common = sorted(set.intersection(*[set(p.pids.tolist()) for p in preds.values()]))
        base = restrict(preds["all_views"], common)
        for name, p in preds.items():
            q = restrict(p, common)
            d = paired_auroc_delta(base, q, ctx, n_boot=(0 if name == "all_views" else n_boot))
            dr = q.risk - base.risk
            rows.append({"arm": arm, "condition": name,
                         "views_supplied": "+".join(VIEW_ABLATIONS[name]),
                         "n_patients_scored": int(p.pids.size),
                         "n_patients_common": len(common),
                         "auroc_5y": d["auc_condition"], "delta_auroc": d["delta_auc"],
                         "delta_auroc_lo": d["delta_lo"], "delta_auroc_hi": d["delta_hi"],
                         "p_boot": d["p_boot"],
                         "mean_risk_5y": float(q.risk.mean()),
                         "mean_delta_risk_5y": float(dr.mean()),
                         "mean_abs_delta_risk_5y": float(np.abs(dr).mean()),
                         "spearman_vs_all_views": float(
                             pd.Series(q.risk).corr(pd.Series(base.risk), method="spearman")),
                         "note": VIEW_ABLATION_NOTE})
        # The comparison that matters for the paper's thesis: the arm TRAINED on the
        # frontal view alone, scored on exactly these patients. Withholding views from a
        # multi-view network only says what that network learned to lean on; it says
        # nothing about how much information a frontal radiograph carries.
        if reference_arm and reference_arm not in arms:
            ref = ArmRunner(ctx, reference_arm)
            rq = restrict(ref.score(IDENTITY, condition=reference_arm), common)
            d = paired_auroc_delta(base, rq, ctx, n_boot=n_boot)
            rows.append({"arm": f"{reference_arm} (reference)", "condition": "trained_frontal_only",
                         "views_supplied": "frontal", "n_patients_scored": int(rq.pids.size),
                         "n_patients_common": len(common), "auroc_5y": d["auc_condition"],
                         "delta_auroc": d["delta_auc"], "delta_auroc_lo": d["delta_lo"],
                         "delta_auroc_hi": d["delta_hi"], "p_boot": d["p_boot"],
                         "mean_risk_5y": float(rq.risk.mean()),
                         "mean_delta_risk_5y": float((rq.risk - base.risk).mean()),
                         "mean_abs_delta_risk_5y": float(np.abs(rq.risk - base.risk).mean()),
                         "spearman_vs_all_views": float(
                             pd.Series(rq.risk).corr(pd.Series(base.risk), method="spearman")),
                         "note": VIEW_ABLATION_NOTE})
            ctx.log.info("  %s TRAINED on frontal alone, same %d patients: AUROC %.4f "
                         "(delta vs %s all views %+.4f [%+.4f, %+.4f])", reference_arm,
                         len(common), d["auc_condition"], arm, d["delta_auc"], d["delta_lo"],
                         d["delta_hi"])
            del ref
        del runner
    df = pd.DataFrame(rows)
    df.to_csv(ctx.table_dir / "interp_view_ablation.csv", index=False)
    ctx.log.info("wrote %s", ctx.table_dir / "interp_view_ablation.csv")
    return df


#: Conditions that blank or retain an anatomic region. Their Δ AUROC intervals are wide,
#: they are not a compartment ranking, and the note column says so on every one of them.
ANATOMIC_CONDITIONS = ("occlude_medial", "occlude_lateral", "occlude_patellofemoral",
                       "occlude_joint", "keep_joint_only", "occlude_joint_meanfill",
                       "keep_joint_only_meanfill")


def risk_degeneracy(risk) -> tuple[int, float, int]:
    """``(largest number of patients sharing one identical risk, that risk, n distinct)``.

    A perturbation that destroys the input does not produce a weak predictor; it produces
    a **constant** one, and a constant predictor's AUROC is 0.5 by construction rather
    than by measurement. This is the number that tells the two apart, and it is written
    into ``interp_occlusion.csv`` for every condition so nobody has to take the note's word
    for it.
    """
    vals, counts = np.unique(np.asarray(risk, dtype=float), return_counts=True)
    j = int(np.argmax(counts))
    return int(counts[j]), float(vals[j]), int(vals.size)


def occlusion_note(condition: str, *, n_patients: int, n_identical: int, n_max_tied: int,
                   n_distinct: int, tied_risk: float, delta_auroc: float,
                   delta_lo: float, delta_hi: float) -> str:
    """The caveat that has to travel with each occlusion row, derived from that row.

    Written as a function rather than a literal because two of these notes are the
    difference between an honest leakage audit and a circular one:

    * ``keep_border_only`` reads as the headline leakage control - "a model shown only the
      text band discriminates at chance" - and it is **not one**. The band is already
      zeroed by the published preprocessing pipeline, so for the large majority of patients
      the border-only input is an all-black image and the model returns one identical
      number. Its AUROC is forced to ~0.5 by degeneracy, so it can neither confirm nor
      refute text leakage. Presented without this note it is circular, and a reviewer who
      notices the circularity before we disclose it will distrust the whole package.
    * ``occlude_border`` is a **pipeline validation**: it proves the perturbation path is
      exact and scoring deterministic, because the crops whose band is already zero must
      come back bit-identical, and they do.

    The informative leakage evidence is ``mask_residual_markers`` and the two widened
    bands, all three of which perturb pixels that actually carry signal.
    """
    ci = f"Δ AUROC {delta_auroc:+.4f} (95% CI {delta_lo:+.4f}, {delta_hi:+.4f})"
    crosses = bool(delta_lo <= 0.0 <= delta_hi)
    if condition == "baseline":
        return ("Locally recomputed unperturbed reference. Every Δ in this table is against "
                "this row, formed inside one process on one device, never against a "
                "published per-patient risk.")
    if condition == "keep_border_only":
        return (f"DEGENERATE INPUT - NOT A TEXT-LEAKAGE TEST. The 31-px band is already "
                f"zeroed by the published preprocessing pipeline, so for most patients the "
                f"border-only input is an all-black image: {n_max_tied} of {n_patients} "
                f"patients receive the identical risk {tied_risk:.10f} and only {n_distinct} "
                f"distinct risks exist across the split. The resulting AUROC is the AUROC of "
                f"a near-constant predictor, forced to about 0.5 by construction; it is "
                f"evidence that the input was destroyed, not that the model ignores "
                f"burned-in text. Read it as a degenerate-input / pipeline check. The "
                f"informative leakage evidence is mask_residual_markers and the widened "
                f"bands (border_62px, border_93px).")
    if condition == "occlude_border":
        return (f"PIPELINE VALIDATION, NOT A LEAKAGE TEST. The band the pipeline already "
                f"blanks is re-blanked, so every crop whose band is exactly zero must return "
                f"a bit-identical risk - {n_identical} of {n_patients} patients do, which is "
                f"what makes this row a proof that the perturbation path is exact and that "
                f"MPS scoring is deterministic within a session. It bounds only what an "
                f"already-blank band could contribute; it does not test burned-in text.")
    if condition == "mask_residual_markers":
        return (f"THE INFORMATIVE LEAKAGE TEST. Every residual saturated marker-like blob in "
                f"the whole test set is blanked while the anatomy is left intact, so this is "
                f"a perturbation of pixels that actually carry signal. {ci}: removing the "
                f"residual markers costs about one thousandth of AUROC, and the interval "
                f"excludes zero, so residual burned-in markers contribute a little and their "
                f"contribution is bounded at that size. Blanking uses crop_qa's looser "
                f"criterion, so it is an upper bound on what markers could contribute.")
    if condition in ("border_62px", "border_93px"):
        return (f"MASKING SENSITIVITY. The band is widened well beyond the published 31 px, "
                f"which removes peri-articular anatomy as well as any residual text, so {ci} "
                f"is an UPPER BOUND on what the widened band could have carried, not a "
                f"text-leakage estimate."
                + (" The interval includes zero." if crosses else
                   " The interval excludes zero."))
    if condition in ANATOMIC_CONDITIONS:
        base = ("ANATOMIC OCCLUSION. " + ci + ". ")
        if crosses:
            return (base + "The interval includes zero, so this row does not establish a "
                    "compartment ranking. Interpret the direction and size of the mean risk "
                    "shift, not the ordering of Δ AUROC across regions.")
        return (base + "The interval excludes zero, but the anatomic conditions were not "
                "adjusted for multiplicity and the fill value moves the estimate (compare "
                "the zero-fill and mean-fill variants of this condition), so still report "
                "the risk shift beside it rather than a compartment ranking.")
    return ci


def stage_occlusion(ctx: InterpContext, arm: str = PRIMARY_ARM, *, n_boot: int = 2000,
                    ops=None) -> pd.DataFrame:
    """B3 + B4 + B6. Every condition scored locally against a locally computed baseline."""
    runner = ArmRunner(ctx, arm)
    ops = list(ops if ops is not None else default_conditions())
    base = runner.score(IDENTITY)
    ctx.log.info("%s BASELINE (local fp32): n=%d events=%d AUROC@5y=%.4f mean risk=%.4f "
                 "[%.0f s]", arm, base.pids.size, int(base.event.sum()),
                 ipcw_auroc(base, ctx), float(base.risk.mean()), base.seconds)
    rp = published_risk(ctx, arm, base.pids)
    np.savez(ctx.out_dir / f"predictions_{arm}_baseline.npz", empi_anon=base.pids,
             hazards=base.hazards, risk=base.risk, rank=base.rank,
             risk_published=rp, time=base.time, event=base.event)
    prov = pd.DataFrame({"arm": [arm],
                         "n_patients": [int(base.pids.size)],
                         "risk_source_for_case_selection": ["published npz (Colab CUDA fp16 autocast)"],
                         "risk_source_for_all_deltas": ["local fp32, recomputed in-process"],
                         "mean_risk_published": [float(rp.mean())],
                         "mean_risk_local": [float(base.risk.mean())],
                         "max_abs_risk_diff": [float(np.abs(rp - base.risk).max())],
                         "median_abs_risk_diff": [float(np.median(np.abs(rp - base.risk)))],
                         "spearman_published_vs_local":
                             [float(pd.Series(rp).corr(pd.Series(base.risk), method="spearman"))],
                         "auroc_5y_published_table": [np.nan],
                         "auroc_5y_local": [ipcw_auroc(base, ctx)]})
    try:
        pm = pd.read_csv(ctx.cfg.path("outputs/tables/test_metrics.csv"))
        prov.loc[0, "auroc_5y_published_table"] = float(
            pm.loc[pm["arm"] == arm, "auc_1825"].iloc[0])
    except Exception as exc:                                       # noqa: BLE001
        ctx.log.warning("could not read the published metric table: %s", exc)
    prov.to_csv(ctx.table_dir / "interp_risk_provenance.csv", index=False)
    ctx.log.info("risk provenance: published vs local max |diff| %.3e, spearman %.6f; "
                 "AUROC local %.6f vs published table %.6f",
                 float(prov.loc[0, "max_abs_risk_diff"]),
                 float(prov.loc[0, "spearman_published_vs_local"]),
                 float(prov.loc[0, "auroc_5y_local"]),
                 float(prov.loc[0, "auroc_5y_published_table"]))
    b_tied, b_val, b_distinct = risk_degeneracy(base.risk)
    rows = [{"arm": arm, "condition": "baseline", "description": "unperturbed, local fp32",
             "n_patients": int(base.pids.size), "n_events": int(base.event.sum()),
             "auroc_5y": ipcw_auroc(base, ctx), "delta_auroc": 0.0,
             "delta_auroc_lo": 0.0, "delta_auroc_hi": 0.0, "p_boot": np.nan,
             "mean_risk_5y": float(base.risk.mean()),
             "mean_delta_risk_5y": 0.0, "mean_abs_delta_risk_5y": 0.0,
             "sd_delta_risk_5y": 0.0, "spearman_vs_baseline": 1.0,
             "n_identical_to_baseline": int(base.pids.size),
             "n_max_tied_risk": b_tied, "n_distinct_risk": b_distinct,
             "note": occlusion_note("baseline", n_patients=int(base.pids.size),
                                    n_identical=int(base.pids.size), n_max_tied=b_tied,
                                    n_distinct=b_distinct, tied_risk=b_val, delta_auroc=0.0,
                                    delta_lo=0.0, delta_hi=0.0),
             "seconds": base.seconds}]
    for op in ops:
        p = runner.score(op)
        assert np.array_equal(p.pids, base.pids)
        d = paired_auroc_delta(base, p, ctx, n_boot=n_boot)
        dr = p.risk - base.risk
        rho = float(pd.Series(p.risk).corr(pd.Series(base.risk), method="spearman"))
        n_tied, tied_val, n_distinct = risk_degeneracy(p.risk)
        n_same = int((p.risk == base.risk).sum())
        rows.append({"arm": arm, "condition": op.name, "description": op.describe(),
                     "n_patients": int(p.pids.size), "n_events": int(p.event.sum()),
                     "auroc_5y": d["auc_condition"], "delta_auroc": d["delta_auc"],
                     "delta_auroc_lo": d["delta_lo"], "delta_auroc_hi": d["delta_hi"],
                     "p_boot": d["p_boot"], "mean_risk_5y": float(p.risk.mean()),
                     "mean_delta_risk_5y": float(dr.mean()),
                     "mean_abs_delta_risk_5y": float(np.abs(dr).mean()),
                     "sd_delta_risk_5y": float(dr.std(ddof=1)),
                     "spearman_vs_baseline": rho,
                     "n_identical_to_baseline": n_same, "n_max_tied_risk": n_tied,
                     "n_distinct_risk": n_distinct,
                     "note": occlusion_note(op.name, n_patients=int(p.pids.size),
                                            n_identical=n_same, n_max_tied=n_tied,
                                            n_distinct=n_distinct, tied_risk=tied_val,
                                            delta_auroc=d["delta_auc"],
                                            delta_lo=d["delta_lo"], delta_hi=d["delta_hi"]),
                     "seconds": p.seconds})
        ctx.log.info("  %-24s AUROC %.4f (delta %+.4f [%+.4f, %+.4f]) mean risk %.4f "
                     "(delta %+.4f) rho %.3f [%.0f s]", op.name, d["auc_condition"],
                     d["delta_auc"], d["delta_lo"], d["delta_hi"], float(p.risk.mean()),
                     float(dr.mean()), rho, p.seconds)
        np.savez(ctx.out_dir / f"predictions_{arm}_{op.name}.npz", empi_anon=p.pids,
                 risk=p.risk, rank=p.rank, time=p.time, event=p.event)
    df = pd.DataFrame(rows)
    df.to_csv(ctx.table_dir / "interp_occlusion.csv", index=False)
    ctx.log.info("wrote %s", ctx.table_dir / "interp_occlusion.csv")
    return df


def default_conditions() -> list[PixelOp]:
    """The pre-specified perturbations: protocol section 22 (b), (c) and (e)."""
    return [
        # B3 - occlusion by anatomic region
        PixelOp("occlude_medial", zero=("medial",)),
        PixelOp("occlude_lateral", zero=("lateral",)),
        PixelOp("occlude_patellofemoral", zero=("patellofemoral",)),
        # occlude_border is NOT a leakage test; it is an internal consistency check. The
        # band is already zero on 88.1% of test crops, so those crops MUST come back
        # bit-identical, and 677 of 734 patients do.
        PixelOp("occlude_border", zero=("border",), frontal_only=False),
        # B4 - negative controls. "background only" IS "occlude_joint": the same model,
        # the same weights, everything inside the joint region destroyed.
        PixelOp("occlude_joint", zero=("joint",)),
        PixelOp("keep_joint_only", keep=("joint",)),
        # THE SAME FACT, CARRIED THROUGH TO ITS CONSEQUENCE. Because the band is already
        # zero on 88.1% of crops, "keep only the border" hands the model an ALL-BLACK image
        # for most patients: 676 of 734 receive one identical risk. Its AUROC of 0.497 is a
        # near-constant predictor's AUROC, forced to ~0.5 by construction, and it is
        # therefore NOT evidence that the model ignores burned-in text. Kept because the
        # protocol pre-specified it and because a degenerate control is still a control -
        # but every consumer gets the caveat through occlusion_note().
        PixelOp("keep_border_only", keep=("border",)),
        # B6 - masking sensitivity
        PixelOp("border_62px", border_px=62),
        PixelOp("border_93px", border_px=93),
        PixelOp("mask_residual_markers", mask_residual_markers=True, frontal_only=False),
        # fill-value sensitivity for the two headline occlusions
        PixelOp("occlude_joint_meanfill", zero=("joint",), mean_fill=True),
        PixelOp("keep_joint_only_meanfill", keep=("joint",), mean_fill=True),
    ]


def _stratify(pred: Prediction, klg: pd.Series, risk_published: np.ndarray,
              horizon: float = HORIZON_DAYS, n_per_cell: int = 8,
              seed: int = 20250720) -> pd.DataFrame:
    """A TP/FP/TN/FN sample at the cohort's own 5-year incidence threshold, KL-matched.

    **The cell assignment uses the PUBLISHED risk**, not the locally recomputed one. The
    two differ by up to ~8e-03 per patient because the published run was Colab CUDA under
    float16 autocast and this one is local float32, which is enough to move a patient
    across the operating point. Selecting on the local risk would put example cases in
    Figure 2 that the published risk table calls something else. Both risks are carried
    into the output table so the discrepancy is auditable.

    Cases are patients with the event by the horizon; controls are patients event-free
    past it; patients censored before the horizon are not classifiable and are excluded.
    The operating point is the risk quantile equal to 1 - observed case fraction, so the
    predicted-positive count matches the observed case count and all four cells exist.
    """
    t, e = np.asarray(pred.time, float), np.asarray(pred.event, int)
    y = np.where((t <= horizon) & (e == 1), 1, np.where(t > horizon, 0, -1))
    usable = y >= 0
    rp = np.asarray(risk_published, dtype=float)
    thr = float(np.quantile(rp[usable], 1.0 - float((y[usable] == 1).mean())))
    hi = rp >= thr
    cell = np.where(usable & (y == 1) & hi, "TP",
                    np.where(usable & (y == 1) & ~hi, "FN",
                             np.where(usable & (y == 0) & hi, "FP",
                                      np.where(usable & (y == 0) & ~hi, "TN", ""))))
    cell_local = np.where(usable & (y == 1) & (pred.risk >= thr), "TP",
                          np.where(usable & (y == 1), "FN",
                                   np.where(usable & (pred.risk >= thr), "FP",
                                            np.where(usable, "TN", ""))))
    d = pd.DataFrame({"empi_anon": pred.pids, "risk_published": rp,
                      "risk_local": pred.risk, "cell": cell,
                      "cell_from_local_risk": cell_local,
                      "cell_agrees": cell == cell_local,
                      "risk_abs_diff": np.abs(rp - pred.risk),
                      "event": pred.event, "time": pred.time})
    d["klg_contra"] = d["empi_anon"].map(klg)
    d = d[d["cell"] != ""].copy()
    rng = np.random.default_rng(int(seed))
    # KL-matched: draw the SAME KL-grade profile in each of the four cells wherever the
    # cell holds a patient at that grade, so a cell-to-cell difference in the maps cannot
    # be a difference in radiographic severity.
    grades = sorted(d["klg_contra"].dropna().unique())
    picks = []
    for g in grades:
        sub = d[d["klg_contra"] == g]
        if sub["cell"].nunique() < 4:
            continue
        k = min(int(np.ceil(n_per_cell / max(1, len(grades) // 2))),
                int(sub["cell"].value_counts().min()))
        for c, part in sub.groupby("cell"):
            take = part.sample(n=min(k, len(part)), random_state=int(rng.integers(1 << 30)))
            picks.append(take)
    out = pd.concat(picks) if picks else d.head(0)
    # top up any short cell from the whole pool so all four cells are represented
    for c in ("TP", "FP", "TN", "FN"):
        have = int((out["cell"] == c).sum())
        if have < n_per_cell:
            pool = d[(d["cell"] == c) & (~d["empi_anon"].isin(out["empi_anon"]))]
            if len(pool):
                out = pd.concat([out, pool.sample(n=min(n_per_cell - have, len(pool)),
                                                  random_state=int(rng.integers(1 << 30)))])
    return out.reset_index(drop=True).assign(threshold=thr)


def stage_attribution(ctx: InterpContext, arm: str = PRIMARY_ARM, *, n_cam: int | None = None,
                      n_ig: int = 120, ig_steps: int = 32, n_panel: int = 8,
                      batch_size: int = 4, seed: int = 20250720) -> pd.DataFrame:
    """B1 + B2 + B5. Grad-CAM and integrated gradients, and where their mass falls."""
    import torch

    runner = ArmRunner(ctx, arm)
    ds = runner.dataset(IDENTITY)
    base = runner.score(IDENTITY)
    feat = pd.read_parquet(ctx.contracts.features_pq, columns=["empi_anon", "klg_contra"])
    klg = feat.set_index(feat["empi_anon"].astype(str))["klg_contra"]
    rp = published_risk(ctx, arm, base.pids)
    strat = _stratify(base, klg, rp, n_per_cell=max(n_panel, 8), seed=seed)
    strat.to_csv(ctx.table_dir / "interp_attribution_sample.csv", index=False)
    ctx.log.info("stratified sample: %s | operating threshold %.4f (on the PUBLISHED risk)",
                 strat["cell"].value_counts().to_dict(), float(strat["threshold"].iloc[0]))
    ctx.log.info("published vs local risk: max |diff| %.3e over %d patients; the cell "
                 "assignment agrees for %d/%d SELECTED cases",
                 float(np.abs(rp - base.risk).max()), base.pids.size,
                 int(strat["cell_agrees"].sum()), len(strat))

    masks = runner.masks
    order = {p: i for i, p in enumerate(ds.pids)}
    n_cam = len(ds) if n_cam is None else min(int(n_cam), len(ds))
    rng = np.random.default_rng(int(seed))
    panel_rows = {order[p] for p in strat["empi_anon"].astype(str) if p in order}
    if n_cam >= len(ds):
        cam_rows = list(range(len(ds)))
    else:                       # the panel cases are never left out of the CAM pass
        pool = [i for i in range(len(ds)) if i not in panel_rows]
        extra = rng.choice(pool, size=max(0, n_cam - len(panel_rows)), replace=False)
        cam_rows = sorted(panel_rows | {int(i) for i in extra})
    # Integrated gradients is ~2 x ig_steps more expensive than Grad-CAM, so it runs on a
    # subsample: the whole stratified TP/FP/TN/FN panel first, then a random top-up drawn
    # from the Grad-CAM rows so the two methods are compared on overlapping images.
    must = sorted({order[p] for p in strat["empi_anon"].astype(str) if p in order})
    pool = [i for i in cam_rows if i not in set(must)]
    top_up = max(0, int(n_ig) - len(must))
    if top_up and pool:
        must += [int(i) for i in rng.choice(pool, size=min(top_up, len(pool)), replace=False)]
    ig_pick = sorted(set(must))

    rows = []
    cam_store: dict[tuple[int, int], np.ndarray] = {}
    cam_native: dict[tuple[int, int], np.ndarray] = {}
    sub = torch.utils.data.Subset(ds, cam_rows)
    t0 = time.time()
    for batch in _iter_batches(sub, batch_size):
        acc: dict[tuple[int, int], np.ndarray] = {}
        nat: dict[tuple[int, int], np.ndarray] = {}
        for net in runner.nets:
            sb, se, cam = gradcam(net, batch, ctx.device)
            for r in range(cam.shape[0]):
                up = upsample_cam(cam[r], ctx.settings.out_size)
                s = float(up.sum())
                key = (int(sb[r]), int(se[r]))
                acc[key] = acc.get(key, 0.0) + (up / s if s > 0 else up)
                sn = float(cam[r].sum())
                nat[key] = nat.get(key, 0.0) + (cam[r] / sn if sn > 0 else cam[r])
        idx = batch["idx"].numpy()
        vid = batch["view_id"].numpy()
        for (bi, ei), a in acc.items():
            a = a / len(runner.nets)
            f = attribution_fractions(a, masks)
            di = int(idx[bi])
            cam_store[(di, ei)] = a.astype(np.float32)
            cam_native[(di, ei)] = (nat[(bi, ei)] / len(runner.nets)).astype(np.float32)
            rows.append({"arm": arm, "method": "gradcam", "empi_anon": ds.pids[di],
                         "slot": ei, "view": ctx.settings.views[int(vid[bi, ei])],
                         **{f"frac_{k}": v for k, v in f.items()}})
    ctx.log.info("Grad-CAM: %d images x %d seeds in %.0f s", len(cam_rows),
                 len(runner.nets), time.time() - t0)

    t0 = time.time()
    sub_ig = torch.utils.data.Subset(ds, ig_pick)
    ig_store: dict[tuple[int, int], np.ndarray] = {}
    for batch in _iter_batches(sub_ig, max(1, batch_size // 2)):
        acc = {}
        for net in runner.nets:
            sb, se, ig = integrated_gradients(net, batch, ctx.device, steps=ig_steps)
            for r in range(ig.shape[0]):
                s = float(np.abs(ig[r]).sum())
                key = (int(sb[r]), int(se[r]))
                acc[key] = acc.get(key, 0.0) + (ig[r] / s if s > 0 else ig[r])
        idx = batch["idx"].numpy()
        vid = batch["view_id"].numpy()
        for (bi, ei), a in acc.items():
            a = a / len(runner.nets)
            f = attribution_fractions(a, masks)
            di = int(idx[bi])
            ig_store[(di, ei)] = a.astype(np.float32)
            rows.append({"arm": arm, "method": "integrated_gradients", "empi_anon": ds.pids[di],
                         "slot": ei, "view": ctx.settings.views[int(vid[bi, ei])],
                         **{f"frac_{k}": v for k, v in f.items()}})
    ctx.log.info("integrated gradients: %d images x %d seeds x %d steps in %.0f s",
                 len(ig_pick), len(runner.nets), ig_steps, time.time() - t0)

    df = pd.DataFrame(rows)
    df.to_csv(ctx.table_dir / "interp_attribution_per_image.csv", index=False)
    area = region_area_fractions(ctx.settings.out_size, ctx.settings.border_px)
    summ = []
    for (method, view), g in df.groupby(["method", "view"]):
        for reg in ("joint", "border", "peripheral", "medial", "lateral", "patellofemoral"):
            v = g[f"frac_{reg}"].dropna()
            # n_images_nonzero and max exist for one row in particular: integrated
            # gradients on the border band. IG is (input - baseline) x gradient with a zero
            # baseline, so a pixel that is exactly zero contributes exactly zero - but the
            # band is NOT exactly zero on 11.9% of test crops (marker masking runs after
            # border masking and can write the ring median back into it), so the row's mean
            # is "essentially zero", not "exactly zero". These two columns are what let a
            # reader tell the two statements apart instead of trusting the rounding.
            summ.append({"arm": arm, "method": method, "view": view, "region": reg,
                         "n_images": int(v.size), "area_fraction": area[reg],
                         "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                         "median": float(v.median()),
                         "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75)),
                         "max": float(v.max()) if v.size else np.nan,
                         "n_images_nonzero": int((v > 0).sum()),
                         "enrichment_vs_area": float(v.mean() / area[reg])})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(ctx.table_dir / "interp_attribution_summary.csv", index=False)
    # Archived at the encoder's NATIVE 16x16 resolution, which is all the information a
    # Grad-CAM map has: the 512x512 version is a deterministic bilinear upsample of it
    # (:func:`upsample_cam`), and storing 858 of those costs 695 MB against 0.9 MB here.
    # Every consumer upsamples; region fractions are scale-invariant, so nothing is lost.
    keys = sorted(cam_store)
    np.savez_compressed(
        ctx.out_dir / f"attribution_maps_{arm}.npz",
        empi_anon=np.asarray([ds.pids[k[0]] for k in keys], dtype=object).astype("U"),
        ds_row=np.asarray([k[0] for k in keys], dtype=np.int64),
        slot=np.asarray([k[1] for k in keys], dtype=np.int64),
        image_row=np.asarray([ds.elems[k[0]][k[1]][0] for k in keys], dtype=np.int64),
        out_size=np.asarray(int(ctx.settings.out_size)),
        gradcam_native=(np.stack([cam_native[k] for k in keys]).astype(np.float32)
                        if keys else np.zeros((0, 1, 1), np.float32)))
    _render_panels(ctx, ds, strat, cam_store, ig_store, order, arm, n_panel=n_panel)
    ctx.log.info("wrote %s and %s", ctx.table_dir / "interp_attribution_per_image.csv",
                 ctx.table_dir / "interp_attribution_summary.csv")
    return sdf


def stage_sanity(ctx: InterpContext, arm: str = PRIMARY_ARM, *, n_images: int = 60,
                 batch_size: int = 4, rand_seed: int = 1234) -> pd.DataFrame:
    """B5. Adebayo-style parameter-randomisation sanity check.

    An attribution method that survives the destruction of the learned function is not
    explaining the model; it is an edge detector. Reported three ways:

    * rank correlation of the map against the TRAINED map, on the same images;
    * the share of mass that still lands in the joint region;
    * the discrimination the corrupted network actually achieves.

    Discrimination is read off the **horizon-free rank score** (total cumulative hazard),
    not the 5-year risk. A randomly initialised head has no ``base_hazard`` bias, so its
    interval hazards sit near 0.5 and the 5-year risk saturates at 1.0 for every patient -
    a scale on which AUROC is meaningless. The rank score never saturates.

    Two rows are reference rows rather than randomisations: ``none_seed0`` is the trained
    seed-0 network, which sets the ceiling for map similarity (the stored map is the
    five-seed mean, so even an untouched member does not reach 1.0), and it also shows what
    one seed achieves against the ensemble.

    **Two denominators, named separately, because they are not the same number.** The map
    columns (``spearman_*``, ``frac_joint_mean``) are computed on the ``n_images_cam``
    sampled crops - 60 by default. The discrimination columns (``auroc_5y_*``,
    ``mean_risk_5y``) are computed on the arm's **whole** scoring roster,
    ``n_patients_auroc`` = 734 for m2_frontal, because ``ArmRunner.score`` scores every
    patient. A single ``n_images`` column at the head of this table read as 60 over the
    whole row and understated the AUROC denominator twelvefold.
    """
    import torch

    runner = ArmRunner(ctx, arm)
    ds = runner.dataset(IDENTITY)
    masks = runner.masks
    rows_pick = sorted(int(i) for i in np.random.default_rng(20250720).choice(
        len(ds), size=min(int(n_images), len(ds)), replace=False))
    sub = torch.utils.data.Subset(ds, rows_pick)
    out = []

    # the trained five-seed reference maps, computed here so this stage is self-contained
    stored: dict[tuple[int, int], np.ndarray] = {}
    for batch in _iter_batches(sub, batch_size):
        acc: dict[tuple[int, int], np.ndarray] = {}
        for net in runner.nets:
            sb, se, cam = gradcam(net, batch, ctx.device)
            for r in range(cam.shape[0]):
                up = upsample_cam(cam[r], ctx.settings.out_size)
                s = float(up.sum())
                acc[(int(sb[r]), int(se[r]))] = acc.get((int(sb[r]), int(se[r])), 0.0) + \
                    (up / s if s > 0 else up)
        idx = batch["idx"].numpy()
        for (bi, ei), a in acc.items():
            stored[(int(idx[bi]), ei)] = (a / len(runner.nets)).astype(np.float32)

    def _one(tag: str, net, is_reference: bool = False):
        sims, sims_in, fr = [], [], []
        for batch in _iter_batches(sub, batch_size):
            sb, se, cam = gradcam(net, batch, ctx.device)
            idx = batch["idx"].numpy()
            for r in range(cam.shape[0]):
                up = upsample_cam(cam[r], ctx.settings.out_size)
                s = float(up.sum())
                up = up / s if s > 0 else up
                key = (int(idx[int(sb[r])]), int(se[r]))
                if key in stored:
                    a, b = stored[key].ravel(), up.ravel()
                    if a.std() > 0 and b.std() > 0:
                        sims.append(float(pd.Series(a).corr(pd.Series(b), method="spearman")))
                    # Restricted to the pixels that actually hold an image. Any two maps
                    # agree that the blanked border and the black collimation margin are
                    # cold, which inflates a whole-image correlation for maps that share
                    # nothing else. This is the number to quote.
                    img = np.asarray(ctx.images[int(ds.elems[key[0]][key[1]][0])])
                    on = (img > 0).ravel()
                    if on.sum() > 100 and a[on].std() > 0 and b[on].std() > 0:
                        sims_in.append(float(pd.Series(a[on]).corr(pd.Series(b[on]),
                                                                   method="spearman")))
                fr.append(attribution_fractions(up, masks)["joint"])
        pr = runner.score(IDENTITY, nets=[net], condition=f"sanity_{tag}")
        out.append({"arm": arm, "randomisation": tag, "is_reference": bool(is_reference),
                    "n_images_cam": len(rows_pick), "n_patients_auroc": int(pr.pids.size),
                    "spearman_to_trained_cam_mean": float(np.mean(sims)) if sims else np.nan,
                    "spearman_to_trained_cam_sd":
                        float(np.std(sims, ddof=1)) if len(sims) > 1 else np.nan,
                    "spearman_within_imaged_pixels_mean":
                        float(np.mean(sims_in)) if sims_in else np.nan,
                    "spearman_within_imaged_pixels_sd":
                        float(np.std(sims_in, ddof=1)) if len(sims_in) > 1 else np.nan,
                    "frac_joint_mean": float(np.mean(fr)) if fr else np.nan,
                    "auroc_5y_rank_score": ipcw_auroc(pr, ctx, score="rank"),
                    "auroc_5y_risk_scale": ipcw_auroc(pr, ctx, score="risk"),
                    "mean_risk_5y": float(pr.risk.mean())})
        ctx.log.info("  sanity %-20s spearman to trained CAM %+.3f (imaged px %+.3f) | "
                     "frac in joint %.3f | AUROC (rank score) %.4f", tag,
                     out[-1]["spearman_to_trained_cam_mean"],
                     out[-1]["spearman_within_imaged_pixels_mean"],
                     out[-1]["frac_joint_mean"], out[-1]["auroc_5y_rank_score"])

    ens = runner.score(IDENTITY, condition="ensemble_reference")
    out.append({"arm": arm, "randomisation": "none_five_seed_ensemble", "is_reference": True,
                "n_images_cam": len(rows_pick), "n_patients_auroc": int(ens.pids.size),
                "spearman_to_trained_cam_mean": np.nan,
                "spearman_to_trained_cam_sd": np.nan,
                "spearman_within_imaged_pixels_mean": np.nan,
                "spearman_within_imaged_pixels_sd": np.nan, "frac_joint_mean": np.nan,
                "auroc_5y_rank_score": ipcw_auroc(ens, ctx, score="rank"),
                "auroc_5y_risk_scale": ipcw_auroc(ens, ctx, score="risk"),
                "mean_risk_5y": float(ens.risk.mean())})
    _one("none_seed0", runner.nets[0], is_reference=True)
    for scope in ("denseblock4", "denseblock3", "denseblock1", "head", "all"):
        net = randomized_net(ctx, runner, scope=scope, seed=rand_seed)
        _one(scope, net)
        del net
    df = pd.DataFrame(out)
    df.to_csv(ctx.table_dir / "interp_sanity_checks.csv", index=False)
    ctx.log.info("wrote %s", ctx.table_dir / "interp_sanity_checks.csv")
    return df


def _render_panels(ctx, ds, strat, cam_store, ig_store, order, arm, n_panel: int = 8):
    """Figure-ready TP/FP/TN/FN attribution panels (B1). PNG, no matplotlib dependency."""
    from PIL import Image
    from scipy import ndimage as ndi

    def heat(a: np.ndarray, smooth: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(rgb, alpha). Magnitude is scaled to its own 99.5th percentile so one hot pixel
        cannot flatten the map, and low-attribution pixels are left transparent so the
        radiograph stays readable underneath. ``smooth`` is DISPLAY ONLY: integrated
        gradients are pixel-sparse and unreadable unsmoothed, and every quantitative number
        in this module is computed on the raw, unsmoothed map."""
        m = np.abs(np.asarray(a, dtype=np.float64))
        if smooth:
            m = ndi.uniform_filter(m, size=int(smooth))
        m = m / max(float(np.percentile(m, 99.5)), 1e-12)
        m = np.clip(m, 0, 1)
        r = np.clip(1.6 * m - 0.35, 0, 1)
        g = np.clip(1.9 * m - 1.0, 0, 1)
        b2 = np.clip(1.0 - 1.9 * m, 0, 1)
        return np.stack([r, g, b2], -1), (m ** 0.7) * 0.72

    imgs = ctx.images
    panel_dir = ctx.out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows_out, manifest = [], []
    for cell in ("TP", "FP", "TN", "FN"):
        sub = strat[strat["cell"] == cell].head(n_panel)
        for _, r in sub.iterrows():
            di = order.get(str(r["empi_anon"]))
            if di is None:
                continue
            key = next((k for k in cam_store if k[0] == di), None)
            if key is None:
                continue
            row = ds.elems[di][key[1]][0]
            base = np.asarray(imgs[row]).astype(np.float32) / 255.0
            gray = np.stack([base] * 3, -1)

            def blend(a, smooth=0):
                rgb, alpha = heat(a, smooth)
                al = alpha[..., None]
                return np.clip(gray * (1 - al) + rgb * al, 0, 1)

            tiles = [gray, blend(cam_store[key])]
            if key in ig_store:
                tiles.append(blend(ig_store[key], smooth=9))
            strip = np.concatenate(tiles, axis=1)
            name = f"{arm}_{cell}_{r['empi_anon']}_klg{r['klg_contra']}.png"
            Image.fromarray((strip * 255).astype(np.uint8)).save(panel_dir / name)
            manifest.append({"arm": arm, "cell": cell, "empi_anon": r["empi_anon"],
                             "klg_contra": r["klg_contra"],
                             "risk_published": float(r["risk_published"]),
                             "risk_local": float(r["risk_local"]),
                             "cell_from_local_risk": r["cell_from_local_risk"],
                             "event": int(r["event"]), "time_days": float(r["time"]),
                             "panels": "image|gradcam" + ("|integrated_gradients"
                                                          if key in ig_store else ""),
                             "file": str((panel_dir / name).resolve())})
            rows_out.append(name)
    pd.DataFrame(manifest).to_csv(ctx.table_dir / "interp_panel_manifest.csv", index=False)
    ctx.log.info("wrote %d attribution panels to %s", len(rows_out), panel_dir)


def stage_regions(ctx: InterpContext) -> pd.DataFrame:
    """Region geometry, its validation against a model-free joint-line detector, and the
    residual-marker / residual-border audit of the test crops. No model is involved."""
    from PIL import Image

    masks = region_masks(ctx.settings.out_size, ctx.settings.border_px)
    area = region_area_fractions(ctx.settings.out_size, ctx.settings.border_px)
    frontal = ctx.index[ctx.index["view"] == "frontal"]["row"].to_numpy()
    jr = np.zeros(frontal.size)
    cc = np.zeros(frontal.size)
    cw = np.zeros(frontal.size)
    for i, r in enumerate(frontal):
        a = np.asarray(ctx.images[int(r)])
        jr[i] = detect_joint_row(a)
        cc[i], cw[i] = limb_landmarks(a, ctx.settings.border_px)
    r0, r1, c0b, c1b = JOINT_BOX
    inside = float(((jr >= r0) & (jr < r1)).mean())
    inside_col = float(((cc >= c0b) & (cc < c1b)).mean())
    covered = (np.minimum(cc + cw / 2, c1b) - np.maximum(cc - cw / 2, c0b)).clip(0) / \
        np.maximum(cw, 1)
    inside_both = float((((jr >= r0) & (jr < r1)) & ((cc >= c0b) & (cc < c1b))).mean())

    band = masks["border"]
    nz, mx = [], []
    for i in range(ctx.images.shape[0]):
        a = np.asarray(ctx.images[i])
        v = a[band]
        nz.append(int((v > 0).sum())); mx.append(int(v.max()))
    nz = np.asarray(nz); mx = np.asarray(mx)
    by_view = (pd.DataFrame({"view": ctx.index["view"].to_numpy(), "nz": nz})
               .groupby("view")["nz"].apply(lambda s: float((s > 0).mean())))

    from src.crop_qa import residual_marker_scan
    from src.preprocess_images import PreprocessParams
    params = PreprocessParams.from_config(ctx.cfg)
    scan = residual_marker_scan([np.asarray(ctx.images[i]) for i in range(ctx.images.shape[0])],
                                list(ctx.index["view"]), params)

    rows = [{"item": "joint_region_box", "value": "rows {}-{} cols {}-{}".format(*JOINT_BOX),
             "note": "the central half of the 512 px crop in each dimension"},
            {"item": "joint_region_area_fraction", "value": f"{area['joint']:.6f}", "note": ""},
            {"item": "border_band_px", "value": str(ctx.settings.border_px),
             "note": "config preprocess.mask_border_frac 0.06 x 512"},
            {"item": "border_band_area_fraction", "value": f"{area['border']:.6f}", "note": ""},
            {"item": "peripheral_area_fraction", "value": f"{area['peripheral']:.6f}", "note": ""},
            {"item": "medial_area_fraction", "value": f"{area['medial']:.6f}",
             "note": "image LEFT half of the joint box; crops read as LEFT knees"},
            {"item": "lateral_area_fraction", "value": f"{area['lateral']:.6f}", "note": ""},
            {"item": "patellofemoral_area_fraction", "value": f"{area['patellofemoral']:.6f}",
             "note": "patellar projection zone, rows {}-{} cols {}-{}".format(*PF_BOX)},
            {"item": "n_frontal_crops_scanned", "value": str(frontal.size), "note": ""},
            {"item": "joint_line_inside_box_fraction", "value": f"{inside:.4f}",
             "note": "model-free matched-filter detector, frontal crops"},
            {"item": "joint_line_row_median", "value": f"{float(np.median(jr)):.1f}", "note": ""},
            {"item": "joint_line_row_iqr", "value":
             f"{float(np.percentile(jr,25)):.0f}-{float(np.percentile(jr,75)):.0f}", "note": ""},
            {"item": "limb_centre_inside_box_fraction", "value": f"{inside_col:.4f}",
             "note": "Otsu limb silhouette centre column inside the box columns"},
            {"item": "joint_line_AND_limb_centre_inside_box_fraction",
             "value": f"{inside_both:.4f}",
             "note": "both landmarks inside; the honest joint coverage of the FIXED box"},
            {"item": "mean_limb_width_covered_by_box", "value": f"{float(covered.mean()):.4f}",
             "note": "mean share of the limb's own width falling inside the box columns"},
            {"item": "limb_centre_col_median", "value": f"{float(np.median(cc)):.1f}",
             "note": f"IQR {float(np.percentile(cc,25)):.0f}-{float(np.percentile(cc,75)):.0f}; "
                     f"image centre is 256"},
            {"item": "limb_width_median", "value": f"{float(np.median(cw)):.1f}", "note": ""},
            {"item": "n_crops_with_nonzero_border_band", "value": str(int((nz > 0).sum())),
             "note": f"of {nz.size} test crops = {float((nz>0).mean())*100:.1f}%"},
            {"item": "border_band_nonzero_pixel_share", "value":
             f"{float(nz.sum())/float(nz.size*band.sum()):.6f}",
             "note": "share of all border-band pixels that are not exactly 0"},
            {"item": "border_band_max_value", "value": str(int(mx.max())),
             "note": "uint8; the marker masker fills with a ring median capped at 60"},
            {"item": "residual_marker_crops_pct", "value": f"{scan['pct']:.1f}",
             "note": f"upper bound, saturated bone edges share the signature; n={scan['total']}"},
            {"item": "residual_marker_by_view_pct", "value":
             json.dumps({k: round(v, 1) for k, v in scan["by_view"].items()}), "note": ""}]
    for v, f in by_view.items():
        rows.append({"item": f"nonzero_border_band_pct_{v}", "value": f"{f*100:.1f}", "note": ""})
    df = pd.DataFrame(rows)
    df.to_csv(ctx.table_dir / "interp_regions.csv", index=False)
    pd.DataFrame({"row": frontal, "joint_row": jr, "limb_centre_col": cc,
                  "limb_width": cw}).to_csv(ctx.out_dir / "joint_line_frontal.csv", index=False)

    mean_frontal = np.zeros((ctx.settings.out_size, ctx.settings.out_size))
    for r in frontal:
        mean_frontal += np.asarray(ctx.images[int(r)], dtype=float)
    mean_frontal /= max(frontal.size, 1)
    np.save(ctx.out_dir / "mean_frontal_crop.npy", mean_frontal)

    def _overlay(base_u8: np.ndarray, path: Path):
        rgb = np.stack([base_u8] * 3, -1).astype(np.float32)
        for reg, col, alpha in (("border", (200, 40, 40), 0.45),
                                ("medial", (40, 170, 90), 0.28),
                                ("lateral", (60, 110, 220), 0.28),
                                ("patellofemoral", (230, 190, 40), 0.25)):
            m = masks[reg]
            rgb[m] = rgb[m] * (1 - alpha) + np.array(col, dtype=np.float32) * alpha
        r0, r1, c0, c1 = JOINT_BOX
        rgb[r0:r0 + 3, c0:c1] = rgb[r1 - 3:r1, c0:c1] = (255, 255, 255)
        rgb[r0:r1, c0:c0 + 3] = rgb[r0:r1, c1 - 3:c1] = (255, 255, 255)
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)

    # A representative crop, not the cohort mean: averaging 858 differently zoomed knees
    # produces a featureless blob, which is useless as a figure panel. "Representative"
    # means BOTH landmarks near the crop centre - picking on the joint row alone selects
    # crops that are vertically typical and horizontally off, which misrepresents the box.
    rep = int(frontal[int(np.argmin(np.hypot((jr - 256) / 72.0, (cc - 256) / 61.0)))])
    _overlay(np.asarray(ctx.images[rep]), ctx.out_dir / "regions_overlay.png")
    _overlay((mean_frontal / max(mean_frontal.max(), 1e-9) * 255).astype(np.uint8),
             ctx.out_dir / "regions_overlay_mean.png")
    ctx.log.info("region overlay drawn on test crop row %d (representative) and on the "
                 "cohort mean", rep)
    ctx.log.info("wrote %s", ctx.table_dir / "interp_regions.csv")
    return df


# =========================================================================== #
# 8. CLI                                                                       #
# =========================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", default="outputs/interpretability")
    ap.add_argument("--table-dir", default="outputs/tables")
    ap.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    ap.add_argument("--stage", default="all",
                    choices=["all", "regions", "attention", "views", "attribution",
                             "occlusion", "sanity", "registered"])
    ap.add_argument("--arm", default=PRIMARY_ARM)
    ap.add_argument("--n-cam", type=int, default=None, help="Grad-CAM sample (default: all)")
    ap.add_argument("--n-ig", type=int, default=120)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    ctx = build_context(args.config, Path(args.shard_dir), Path(args.ckpt_dir),
                        Path(args.cache_dir), args.device, Path(args.out_dir),
                        Path(args.table_dir))
    ran = []
    if args.stage in ("all", "regions"):
        stage_regions(ctx); ran.append("regions")
    if args.stage in ("all", "attention"):
        stage_attention(ctx); ran.append("attention")
    if args.stage in ("all", "views"):
        stage_view_ablation(ctx, n_boot=args.n_boot); ran.append("views")
    if args.stage in ("all", "occlusion"):
        stage_occlusion(ctx, args.arm, n_boot=args.n_boot); ran.append("occlusion")
    if args.stage in ("all", "attribution"):
        stage_attribution(ctx, args.arm, n_cam=args.n_cam, n_ig=args.n_ig,
                          ig_steps=args.ig_steps); ran.append("attribution")
    if args.stage in ("all", "registered"):
        stage_registered(ctx, args.arm); ran.append("registered")
    if args.stage in ("all", "sanity"):
        stage_sanity(ctx, args.arm); ran.append("sanity")
    (ctx.out_dir / "run_record.json").write_text(json.dumps({
        "module": MODULE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stages": ran, "arm": args.arm, "device": str(ctx.device), "amp": False,
        "contract_hash": ctx.settings.contract_hash(),
        "joint_box": list(JOINT_BOX), "pf_box": list(PF_BOX),
        "border_px": int(ctx.settings.border_px), "horizon_days": HORIZON_DAYS,
        "baseline_policy": "every baseline recomputed locally in-process; no published "
                           "npz is ever an arm of a comparison",
        "case_selection_policy": "TP/FP/TN/FN cells are assigned from the PUBLISHED "
                                 "derived-data/cohort/test_hazards_{arm}.npz risk (the "
                                 "manuscript's number), then frozen; every delta is "
                                 "computed locally on that frozen list",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
