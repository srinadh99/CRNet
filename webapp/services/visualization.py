
"""
Visualization utilities for astronomy images.

Supports:
- zscale (DS9-like interval), with linear mapping
- logstretch
- asinhstretch

The goal is stable, dependency-light rendering to 8-bit PNG for web display.
If astropy is available, we use astropy.visualization.ZScaleInterval for zscale limits.
Otherwise we fall back to percentile-based limits.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from PIL import Image

VisMode = Literal["zscale", "logstretch", "asinhstretch", "minmax"]


@dataclass(frozen=True)
class RenderResult:
    png_bytes: bytes
    width: int
    height: int
    # Mapping from display pixels to original pixels (orig = disp * scale + offset; here offset=0)
    scale_x: float
    scale_y: float
    vmin: float
    vmax: float


def _finite_view(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")
    return finite


def zscale_limits(data: np.ndarray) -> Tuple[float, float]:
    finite = _finite_view(data)
    # Try astropy's ZScaleInterval for better astronomy-style scaling.
    try:
        from astropy.visualization import ZScaleInterval  # type: ignore
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(finite)
    except Exception:
        # Fallback: robust percentiles (not identical to ZScale but close enough for display)
        vmin, vmax = np.percentile(finite, [0.5, 99.5]).astype(float)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Last-resort: min/max on finite pixels
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def minmax_limits(data: np.ndarray) -> Tuple[float, float]:
    finite = _finite_view(data)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def apply_stretch(norm01: np.ndarray, mode: VisMode, *, log_a: float = 1000.0, asinh_a: float = 10.0) -> np.ndarray:
    """
    norm01 must be in [0,1]. Returns stretched values also in [0,1].
    """
    if mode == "zscale" or mode == "minmax":
        return norm01
    if mode == "logstretch":
        a = float(log_a)
        # log1p mapping (stable at 0)
        return np.log1p(a * norm01) / np.log1p(a)
    if mode == "asinhstretch":
        a = float(asinh_a)
        return np.arcsinh(a * norm01) / np.arcsinh(a)
    raise ValueError(f"Unknown visualization mode: {mode}")


def _normalize(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    norm = (data - vmin) / denom
    return np.clip(norm, 0.0, 1.0)


def render_png(
    data: np.ndarray,
    *,
    mode: VisMode = "zscale",
    vmin: float | None = None,
    vmax: float | None = None,
    max_size: int | None = None,
) -> RenderResult:
    """
    Render a 2D array into an 8-bit PNG (grayscale). Optionally downsample to fit max_size.

    Downsampling strategy:
      - If max_size is set and max(H, W) > max_size: use a simple stride step (nearest sampling).
        This keeps mapping from display->original easy: orig ≈ disp * step.

    Returns mapping scale_x/scale_y such that:
      x_orig ≈ x_disp * scale_x
      y_orig ≈ y_disp * scale_y
    """
    if data.ndim != 2:
        raise ValueError("render_png expects a 2D array")

    # Downsample (for full-image display)
    h, w = data.shape
    step = 1
    if max_size is not None and max(h, w) > int(max_size) and int(max_size) > 0:
        step = int(math.ceil(max(h, w) / int(max_size)))
        data_view = data[::step, ::step]
    else:
        data_view = data

    # Limits
    if vmin is None or vmax is None:
        if mode == "minmax":
            vmin2, vmax2 = minmax_limits(data_view)
        else:
            vmin2, vmax2 = zscale_limits(data_view)
    else:
        vmin2, vmax2 = float(vmin), float(vmax)

    norm = _normalize(data_view.astype(np.float32, copy=False), vmin2, vmax2)
    stretched = apply_stretch(norm, mode)
    img8 = (stretched * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(img8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    hh, ww = img8.shape
    return RenderResult(
        png_bytes=png_bytes,
        width=int(ww),
        height=int(hh),
        scale_x=float(step),
        scale_y=float(step),
        vmin=float(vmin2),
        vmax=float(vmax2),
    )


def png_bytes_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def mask_to_png_bytes(mask01: np.ndarray) -> bytes:
    """
    Convert a binary mask (0/1) into an 8-bit PNG (0=black, 1=white).
    """
    if mask01.ndim != 2:
        raise ValueError("mask_to_png_bytes expects 2D")
    img8 = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
