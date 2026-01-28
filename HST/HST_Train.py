#!/usr/bin/env python3
"""HST_Train.py

Reproducible, script-based version of the CRNet HST ACS/WFC training + evaluation workflow.

This script is meant to replace a notebook that:
  1) trains a CRNet UNet mask model on deepCR.ACS-WFC,
  2) runs qualitative examples and saves diagnostic figures,
  3) runs ROC evaluation on the HST test set, saves per-filter results,
  4) plots ROC curves to PDF,
  5) saves visualizations of the first Conv2d layer filters and their 16x16 DFT magnitudes.

Typical usage
-------------
# Train + evaluate (recommended)
python HST_Train.py \
  --base_dir /scratch/srinadb/CRNet/CRNet/deepCR.ACS-WFC \
  --out_dir  ACS-WFC-BN \
  --model_name ACS-WFC-BN \
  --device GPU \
  --epochs 50 \
  --seed 1

# Evaluate an already-trained model
python HST_Train.py \
  --base_dir /scratch/srinadb/CRNet/CRNet/deepCR.ACS-WFC \
  --out_dir  ACS-WFC-BN \
  --model_name ACS-WFC-BN \
  --device GPU \
  --skip_train \
  --mask_path ./ACS-WFC-BN/2026-01-11_ACS-WFC-BN_epoch50.pth

Reproducibility notes
---------------------
* True bitwise determinism on GPU is not guaranteed for all kernels/versions.
  If you need maximum determinism, use:
    --deterministic
  and consider running on CPU.

* CRNet.training.train internally chooses num_workers based on CUDA availability.
  Multiple dataloader workers can reduce reproducibility unless worker seeds
  are controlled. If you need strict determinism, consider editing CRNet/training.py
  to accept a fixed num_workers and pass a worker_init_fn + torch.Generator.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import logging
import os
import platform
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Headless backend for clusters/servers
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Stronger deterministic mode (may raise errors for some ops)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            # Recommended for some CUDA matmul determinism
            # Needs to be set before some CUDA libraries are initialized to be fully effective.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    except Exception:
        # Torch not installed or not importable at this point
        pass


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_run_config(out_dir: Path, cfg: Dict) -> None:
    out_path = out_dir / "run_config.json"
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def get_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_available"] = str(torch.cuda.is_available())
        versions["cuda_version"] = str(getattr(torch.version, "cuda", None))
        versions["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception:
        versions["torch"] = "<not available>"

    return versions


# ------------------------------
# CRNet utility wrappers
# ------------------------------

def plot_detection_diagnostics(
    img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    cleaned: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Call whichever plotting helper exists in your CRNet.util."""

    from CRNet import util as U

    if hasattr(U, "plot_cr_detection_diagnostics"):
        # User's notebook import
        U.plot_cr_detection_diagnostics(
            img,
            gt_mask,
            pred_mask,
            cleaned,
            model_name=model_name,
            save_path=str(save_path),
        )
        return

    if hasattr(U, "plotCRDetection"):
        # Function in the util.py you pasted
        U.plotCRDetection(
            img,
            gt_mask,
            pred_mask,
            cleaned,
            model_name=model_name,
            save_path=str(save_path),
        )
        return

    raise AttributeError(
        "Neither util.plot_cr_detection_diagnostics nor util.plotCRDetection exists. "
        "Please add one of them or update HST_Train.py accordingly."
    )


def save_first_conv_visualizations(crnet, out_dir: Path, model_name: str) -> None:
    """Save first conv filters and DFT magnitudes using whichever helper exists."""

    from CRNet import util as U

    filters_path = out_dir / f"First_ConvFilters_{model_name}.png"
    dft_path = out_dir / f"First_ConvFiltersdft16_{model_name}.png"

    if hasattr(U, "save_first_conv_filters_and_dft16_from_model"):
        U.save_first_conv_filters_and_dft16_from_model(
            crnet,
            save_path_filters=str(filters_path),
            save_path_dft=str(dft_path),
        )
        return

    if hasattr(U, "plotFirstConvFilters"):
        # Function in the util.py you pasted
        U.plotFirstConvFilters(
            crnet,
            save_path_filters=str(filters_path),
            save_path_dft=str(dft_path),
        )
        return

    raise AttributeError(
        "Neither util.save_first_conv_filters_and_dft16_from_model nor util.plotFirstConvFilters exists. "
        "Please add one of them or update HST_Train.py accordingly."
    )


def maybe_save_up8_skip_activation_grid(crnet, out_dir: Path, model_name: str) -> None:
    """Optional: call your custom activation-grid saver if it exists."""

    from CRNet import util as U

    if hasattr(U, "save_up8_skip_activation_grid"):
        out_path = out_dir / f"Up8_SkipActivations_{model_name}.png"
        U.save_up8_skip_activation_grid(crnet, save_path=str(out_path))


# ------------------------------
# Training / evaluation steps
# ------------------------------

def run_training(
    base_dir: Path,
    out_dir: Path,
    model_name: str,
    epochs: int,
    epoch_phase0: Optional[int],
    batch_size: int,
    lr: float,
    aug_sky: Tuple[float, float],
    hidden: int,
    model_kwargs: Dict,
) -> Path:
    """Train a CRNet model and return the saved .pth path."""

    from CRNet import train

    train_dirs_path = base_dir / "train_dirs.npy"
    if not train_dirs_path.exists():
        raise FileNotFoundError(f"Missing {train_dirs_path}")

    train_dirs = np.load(str(train_dirs_path), allow_pickle=True)

    logging.info("Training dirs loaded: %d", len(train_dirs))
    logging.info("Training model_name=%s out_dir=%s", model_name, str(out_dir))

    trainer = train(
        train_dirs,
        aug_sky=aug_sky,
        model_kwargs=model_kwargs,
        name=model_name,
        hidden=hidden,
        epoch=int(epochs),
        epoch_phase0=epoch_phase0,
        batch_size=int(batch_size),
        lr=float(lr),
        directory=str(out_dir) + os.sep,
        verbose=True,
        use_tqdm=True,
    )

    try:
        trainer.train()
        trainer.plot_loss()

        # Save final state
        fn_stem = trainer.save()  # returns filename stem without extension

        # Save validation loss history
        np.save(out_dir / "val_loss.npy", np.asarray(trainer.validation_loss, dtype=np.float32))

        mask_path = out_dir / f"{fn_stem}.pth"
        logging.info("Saved trained model: %s", str(mask_path))
        return mask_path

    finally:
        try:
            del trainer
        except Exception:
            pass
        gc.collect()


def load_crnet(
    mask_path: Path,
    device: str,
    hidden: int,
    model_kwargs: Dict,
    scale: float = 1.0,
    norm: bool = True,
):
    from CRNet import CRNet

    return CRNet(
        mask=str(mask_path),
        device=device,
        hidden=hidden,
        model_kwargs=model_kwargs,
        scale=scale,
        norm=norm,
    )


def run_qualitative_examples(
    crnet,
    base_dir: Path,
    out_dir: Path,
    model_name: str,
    threshold: float,
    fill_method: str,
    fill_size: int,
) -> None:

    examples = [
        # ---------- F435W ----------
        ("f435w", "9694", "6", "1_210.npy", "EX", "F435W"),
        ("f435w", "10120", "3", "1_200.npy", "GC", "F435W"),
        ("f435w", "10342", "3", "1_209.npy", "GAL", "F435W"),
        # ---------- F606W ----------
        ("f606w", "12438", "1", "1_134.npy", "EX", "F606W"),
        ("f606w", "11586", "5", "1_100.npy", "GC", "F606W"),
        ("f606w", "10407", "3", "1_156.npy", "GAL", "F606W"),
        # ---------- F814W ----------
        ("f814w", "10092", "1", "1_210.npy", "EX", "F814W"),
        ("f814w", "12602", "1", "1_129.npy", "GC", "F814W"),
        ("f814w", "13804", "6", "1_200.npy", "GAL", "F814W"),
    ]

    for flt, prop, visit, fname, field, filt_tag in examples:
        npy_path = base_dir / "data" / "npy_test" / flt / prop / visit / fname
        if not npy_path.exists():
            logging.warning("Example file missing: %s (skipping)", str(npy_path))
            continue

        data = np.load(str(npy_path))
        img = data[0]
        gt = data[1]

        pred_mask, cleaned = crnet.clean(
            img,
            threshold=float(threshold),
            inpaint=True,
            binary=True,
            fill_method=fill_method,
            fill_size=int(fill_size),
        )

        save_path = out_dir / f"{filt_tag}_{field}_{model_name}.png"
        plot_detection_diagnostics(img, gt, pred_mask, cleaned, model_name=model_name, save_path=save_path)

        logging.info("Saved example diagnostic: %s", str(save_path))


def run_roc_testset(
    crnet,
    base_dir: Path,
    out_dir: Path,
    model_name: str,
    dilate: bool,
    rad: int,
    stride: int = 1,
) -> None:
    """Run ROC evaluation and save results to .npy like the notebook."""

    from CRNet.evaluate import roc

    test_dirs_path = base_dir / "test_dirs.npy"
    if not test_dirs_path.exists():
        raise FileNotFoundError(f"Missing {test_dirs_path}")

    test_dirs = np.load(str(test_dirs_path), allow_pickle=True)

    # Map proposal_visit -> field type
    field_type = {
        "10120_3": "GC",
        "11340_11": "GC",
        "9694_6": "EX",
        "10342_3": "GAL",
        "10407_3": "GAL",
        "13364_95": "GAL",
        "11586_5": "GC",
        "12438_1": "EX",
        "10092_1": "EX",
        "12602_1": "GC",
        "12058_6": "GAL",
        "13804_6": "GAL",
    }

    f814_dirs = {"GC": [], "EX": [], "GAL": []}
    f606_dirs = {"GC": [], "EX": [], "GAL": []}
    f435_dirs = {"GC": [], "EX": [], "GAL": []}

    # Group test directories by filter and field type
    for d in test_dirs:
        d = str(d)
        parts = d.split("/")
        # Expect: .../npy_test/<filter>/<proposal>/<visit>/...
        if len(parts) < 5:
            continue
        _filter = parts[-4]
        key = f"{parts[-3]}_{parts[-2]}"
        f_type = field_type.get(key)
        if f_type is None:
            logging.warning("Unknown field key %s from dir %s (skipping)", key, d)
            continue

        if _filter == "f435w":
            f435_dirs[f_type].append(d)
        elif _filter == "f606w":
            f606_dirs[f_type].append(d)
        elif _filter == "f814w":
            f814_dirs[f_type].append(d)

    def eval_and_save(filter_tag: str, grouped: Dict[str, List[str]]):
        logging.info("Testing model on %s testset", filter_tag)
        for f_type, dirs in grouped.items():
            if not dirs:
                logging.warning("No dirs for %s %s", filter_tag, f_type)
                continue

            dirs_eval = dirs[:: max(1, int(stride))]
            tpr_fpr, tpr_fpr_dilate = roc(crnet, dirs_eval, dilate=bool(dilate), rad=int(rad))

            tpr, fpr = tpr_fpr
            tpr_d, fpr_d = tpr_fpr_dilate

            out_path = out_dir / f"{filter_tag}_{f_type}_{model_name}.npy"
            np.save(str(out_path), [[tpr, fpr], [tpr_d, fpr_d]])
            logging.info("Saved ROC arrays: %s", str(out_path))

    eval_and_save("F435W", f435_dirs)
    eval_and_save("F606W", f606_dirs)
    eval_and_save("F814W", f814_dirs)


def plot_roc_curves(out_dir: Path, model_name: str) -> None:
    """Recreate the notebook ROC plotting to PDF."""

    try:
        import seaborn as sns

        sns.set_theme(style="ticks", context="talk")
    except Exception:
        pass

    def _plotter(filter_name: str) -> None:
        fig, axs = plt.subplots(1, 3, figsize=(26, 6), sharey=True, sharex=True)
        fig.suptitle(f"{filter_name} Test set", y=0.97, fontsize=20)

        # Load saved arrays
        gc_arr = np.load(str(out_dir / f"{filter_name}_GC_{model_name}.npy"), allow_pickle=True)
        ex_arr = np.load(str(out_dir / f"{filter_name}_EX_{model_name}.npy"), allow_pickle=True)
        gal_arr = np.load(str(out_dir / f"{filter_name}_GAL_{model_name}.npy"), allow_pickle=True)

        tf_global = [gc_arr[0], ex_arr[0], gal_arr[0]]
        tf_global_d = [gc_arr[1], ex_arr[1], gal_arr[1]]

        titles = ["Globular Clusters", "Extragalactic Fields", "Resolved Galaxies"]

        for i in range(3):
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("FPR [%]")
            axs[i].grid()
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(20, 100)

            t_global, f_global = tf_global[i]
            t_global_d, f_global_d = tf_global_d[i]

            # Convert to numpy arrays and sort for interpolation
            f_global = np.asarray(f_global)
            t_global = np.asarray(t_global)
            si = np.argsort(f_global)
            f_global_s = f_global[si]
            t_global_s = t_global[si]

            f_global_d = np.asarray(f_global_d)
            t_global_d = np.asarray(t_global_d)
            si_d = np.argsort(f_global_d)
            f_global_d_s = f_global_d[si_d]
            t_global_d_s = t_global_d[si_d]

            # TPR at fixed FPR
            tpr_no_dilate_05 = np.interp(0.5, f_global_s, t_global_s)
            tpr_dilate_05 = np.interp(0.5, f_global_d_s, t_global_d_s)
            tpr_no_dilate_005 = np.interp(0.05, f_global_s, t_global_s)
            tpr_dilate_005 = np.interp(0.05, f_global_d_s, t_global_d_s)

            label_no_dilate = (
                f"{filter_name} w/o dilate\n"
                f"(TPR@0.5={tpr_no_dilate_05:.1f}%, TPR@0.05={tpr_no_dilate_005:.1f}%)"
            )
            label_dilate = (
                f"{filter_name} with dilate\n"
                f"(TPR@0.5={tpr_dilate_05:.1f}%, TPR@0.05={tpr_dilate_005:.1f}%)"
            )

            axs[i].plot(f_global, t_global, label=label_no_dilate, ls="-.", c="k")
            axs[i].plot(f_global_d, t_global_d, label=label_dilate, ls="-", c="r")

            axs[i].legend(frameon=False, loc="lower right")

        axs[0].set_ylabel("TPR [%]")

        out_pdf = out_dir / f"{filter_name}_ROC_{model_name}.pdf"
        fig.savefig(str(out_pdf), format="pdf", bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved ROC plot: %s", str(out_pdf))

    _plotter("F435W")
    _plotter("F606W")
    _plotter("F814W")


# ------------------------------
# CLI
# ------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train + evaluate CRNet on HST ACS/WFC deepCR dataset")

    p.add_argument("--base_dir", type=str, required=True, help="Path to deepCR.ACS-WFC root (contains train_dirs.npy/test_dirs.npy)")
    p.add_argument("--out_dir", type=str, default="ACS-WFC-BN", help="Output directory")
    p.add_argument("--model_name", type=str, default="ACS-WFC-BN", help="Model name prefix")

    p.add_argument("--device", type=str, default="GPU", choices=["CPU", "GPU"], help="Inference device for CRNet")

    # Training
    p.add_argument("--skip_train", action="store_true", help="Skip training and use --mask_path")
    p.add_argument("--mask_path", type=str, default=None, help="Existing .pth to load if --skip_train")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epoch_phase0", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.005)

    p.add_argument("--aug_sky_min", type=float, default=-0.9)
    p.add_argument("--aug_sky_max", type=float, default=3.0)

    # Model
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--act_norm", type=str, default="batch", choices=["none", "batch", "group", "instance"])
    p.add_argument("--att", action="store_true")
    p.add_argument("--deeper", action="store_true")

    # Reproducibility
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic", action="store_true", help="Enable torch deterministic algorithms (may be slower / raise errors)")

    # Qualitative examples
    p.add_argument("--skip_examples", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fill_method", type=str, default="maskfill", choices=["maskfill", "median"])
    p.add_argument("--fill_size", type=int, default=5)

    # Test / ROC
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--skip_plot_roc", action="store_true")
    p.add_argument("--dilate", action="store_true", help="Use dilation when computing ROC")
    p.add_argument("--rad", type=int, default=1, help="Disk radius for dilation")
    p.add_argument("--stride", type=int, default=1, help="Subsample test dirs by this stride")

    # Filters/diagnostics
    p.add_argument("--skip_filters", action="store_true", help="Skip saving first-conv filter visualizations")
    p.add_argument("--save_up8_skip", action="store_true", help="If your util has save_up8_skip_activation_grid, call it")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())

    # Logging
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_path))],
    )

    seed_everything(int(args.seed), deterministic=bool(args.deterministic))

    cfg = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "model_name": args.model_name,
        "device": args.device,
        "epochs": args.epochs,
        "epoch_phase0": args.epoch_phase0,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "aug_sky": [args.aug_sky_min, args.aug_sky_max],
        "hidden": args.hidden,
        "act_norm": args.act_norm,
        "att": bool(args.att),
        "deeper": bool(args.deeper),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "threshold": args.threshold,
        "fill_method": args.fill_method,
        "fill_size": args.fill_size,
        "dilate": bool(args.dilate),
        "rad": args.rad,
        "stride": args.stride,
        "versions": get_versions(),
    }
    save_run_config(out_dir, cfg)

    # Build model_kwargs consistently for both trainer + CRNet loader
    model_kwargs = {
        "hidden": int(args.hidden),
        "act_norm": str(args.act_norm),
        "att": bool(args.att),
        "deeper": bool(args.deeper),
    }

    # Train or load
    if args.skip_train:
        if not args.mask_path:
            raise ValueError("--skip_train requires --mask_path")
        mask_path = Path(args.mask_path).expanduser().resolve()
        if not mask_path.exists():
            raise FileNotFoundError(str(mask_path))
    else:
        mask_path = run_training(
            base_dir=base_dir,
            out_dir=out_dir,
            model_name=args.model_name,
            epochs=int(args.epochs),
            epoch_phase0=args.epoch_phase0,
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            aug_sky=(float(args.aug_sky_min), float(args.aug_sky_max)),
            hidden=int(args.hidden),
            model_kwargs=model_kwargs,
        )

    # Instantiate model
    crnet = load_crnet(
        mask_path=mask_path,
        device=str(args.device),
        hidden=int(args.hidden),
        model_kwargs=model_kwargs,
        scale=1.0,
        norm=True,
    )

    # Qualitative examples
    if not args.skip_examples:
        run_qualitative_examples(
            crnet,
            base_dir=base_dir,
            out_dir=out_dir,
            model_name=args.model_name,
            threshold=float(args.threshold),
            fill_method=str(args.fill_method),
            fill_size=int(args.fill_size),
        )

    # Optional extra diagnostics
    if args.save_up8_skip:
        maybe_save_up8_skip_activation_grid(crnet, out_dir=out_dir, model_name=args.model_name)

    # ROC evaluation
    if not args.skip_test:
        run_roc_testset(
            crnet,
            base_dir=base_dir,
            out_dir=out_dir,
            model_name=args.model_name,
            dilate=True,  # always compute both in roc(); it returns both when dilate=True
            rad=int(args.rad),
            stride=int(args.stride),
        )

    if not args.skip_plot_roc:
        plot_roc_curves(out_dir=out_dir, model_name=args.model_name)

    # First conv filters
    if not args.skip_filters:
        save_first_conv_visualizations(crnet, out_dir=out_dir, model_name=args.model_name)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
