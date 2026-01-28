
"""
CRNet model service.

Loads the CRNet PyTorch model once and provides inference helpers.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Any

import numpy as np
import torch

# Import via package root for maximum compatibility across CRNet versions.
from CRNet import CRNet


_MODEL = None
_MODEL_LOCK = threading.Lock()


def _auto_model_kwargs_from_path(model_path: str) -> dict:
    """
    Infer UNet configuration flags from weight filename conventions.

    Expected patterns in filename (case-insensitive):
      - "att" -> attention UNet (att=True)
      - "gn"  -> group norm (act_norm="group")
      - "bn"  -> batch norm (act_norm="batch")

    If no hints are found, we default to CRNet defaults (batch norm, no attention).
    """
    p = os.path.basename(model_path).lower()
    mk = {}
    if "att" in p:
        mk["att"] = True
    if "gn" in p:
        mk["act_norm"] = "group"
    elif "bn" in p:
        mk["act_norm"] = "batch"
    return mk


def get_model() -> CRNet:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            model_path = os.environ.get(
                "CRNET_MODEL_PATH",
                # Default to the bundled ACS/WFC batch-norm weights (from CRNet1).
                os.path.join(os.path.dirname(__file__), "..", "weights", "ACS-WFC-BN-16.pth"),
            )
            model_path = os.path.abspath(model_path)

            device = os.environ.get("CRNET_DEVICE", "CPU").upper().strip()
            hidden = int(os.environ.get("CRNET_HIDDEN", "32"))

            model_kwargs = _auto_model_kwargs_from_path(model_path)

            # Optional explicit overrides (wins over filename auto-detection)
            env_att = os.environ.get("CRNET_ATT")
            if env_att is not None:
                v = env_att.strip().lower()
                if v in {"1", "true", "yes", "y"}:
                    model_kwargs["att"] = True
                elif v in {"0", "false", "no", "n"}:
                    model_kwargs["att"] = False

            env_act = os.environ.get("CRNET_ACT_NORM")
            if env_act:
                model_kwargs["act_norm"] = env_act.strip().lower()

            env_deeper = os.environ.get("CRNET_DEEPER")
            if env_deeper is not None:
                v = env_deeper.strip().lower()
                if v in {"1", "true", "yes", "y"}:
                    model_kwargs["deeper"] = True
                elif v in {"0", "false", "no", "n"}:
                    model_kwargs["deeper"] = False


            # Torch perf knobs (safe defaults for CPU inference)
            try:
                torch.set_num_threads(int(os.environ.get("CRNET_TORCH_THREADS", "1")))
            except Exception:
                pass

            _MODEL = CRNet(
                mask=model_path,
                device=device,
                hidden=hidden,
                model_kwargs=model_kwargs,
            )
        return _MODEL


def infer_prob_mask(roi: np.ndarray) -> np.ndarray:
    """
    Returns a float32 probability mask in [0,1] with same HxW as roi.
    """
    mdl = get_model()

    # The CRNet wrapper expects 2D float32-ish arrays.
    roi_f = np.asarray(roi, dtype=np.float32)

    prob = mdl.clean(
        roi_f,
        threshold=0.5,
        inpaint=False,
        binary=False,
        segment=False,
        n_jobs=1,
    )
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.clip(prob, 0.0, 1.0)
    return prob
