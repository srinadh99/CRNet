
from __future__ import annotations

import base64
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from .services.crnet_service import infer_prob_mask, get_model
from .services.fits_io import load_fits_from_bytes
from .services.visualization import render_png, png_bytes_to_b64, VisMode


FULL_MAX_SIZE = 900  # max dimension for the full-image display (pixels)
MAX_IMAGES = 5       # keep last N uploaded images in memory
MAX_ROI_SIZE = 1024  # safety bound


class ImageStore:
    def __init__(self, max_items: int = MAX_IMAGES):
        self.max_items = int(max_items)
        self._store: "OrderedDict[str, dict]" = OrderedDict()

    def put(self, data: np.ndarray, meta: dict) -> str:
        image_id = uuid.uuid4().hex
        self._store[image_id] = {"data": data, "meta": meta}
        self._store.move_to_end(image_id)
        self._evict()
        return image_id

    def get(self, image_id: str) -> dict | None:
        item = self._store.get(image_id)
        if item is not None:
            self._store.move_to_end(image_id)
        return item

    def _evict(self) -> None:
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


store = ImageStore()


def _finite_stats(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _parse_vis(value: str | None) -> VisMode:
    if not value:
        return "zscale"
    value = value.strip().lower()
    if value in {"zscale", "logstretch", "asinhstretch", "minmax"}:
        return value  # type: ignore
    return "zscale"


app = Flask(__name__)


@app.route("/")
def index():
    # Trigger model load early (optional)
    try:
        get_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {e}"
    return render_template("index.html", model_status=model_status)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "fits" not in request.files:
        return jsonify({"ok": False, "error": "No file field named 'fits' found."}), 400

    f = request.files["fits"]
    if not f.filename.lower().endswith(".fits"):
        return jsonify({"ok": False, "error": "Please upload a .fits file."}), 400

    fits_bytes = f.read()
    try:
        data2d, meta = load_fits_from_bytes(fits_bytes)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to read FITS: {e}"}), 400

    image_id = store.put(data2d, meta)

    # Default render
    vis = _parse_vis(request.form.get("vis") or request.args.get("vis") or "zscale")
    rr = render_png(data2d, mode=vis, max_size=FULL_MAX_SIZE)

    payload = {
        "ok": True,
        "image_id": image_id,
        "orig_shape": [int(data2d.shape[1]), int(data2d.shape[0])],  # [W, H]
        "display_shape": [rr.width, rr.height],
        "scale_x": rr.scale_x,
        "scale_y": rr.scale_y,
        "vis": vis,
        "vmin": rr.vmin,
        "vmax": rr.vmax,
        "full_png_b64": png_bytes_to_b64(rr.png_bytes),
        "stats": _finite_stats(data2d),
    }
    return jsonify(payload)


@app.route("/api/render_full", methods=["POST"])
def api_render_full():
    body = request.get_json(force=True, silent=True) or {}
    image_id = body.get("image_id")
    vis = _parse_vis(body.get("vis"))

    item = store.get(str(image_id))
    if item is None:
        return jsonify({"ok": False, "error": "Unknown image_id (upload again)."}), 400

    data2d = item["data"]

    rr = render_png(data2d, mode=vis, max_size=FULL_MAX_SIZE)

    return jsonify(
        {
            "ok": True,
            "image_id": image_id,
            "display_shape": [rr.width, rr.height],
            "scale_x": rr.scale_x,
            "scale_y": rr.scale_y,
            "vis": vis,
            "vmin": rr.vmin,
            "vmax": rr.vmax,
            "full_png_b64": png_bytes_to_b64(rr.png_bytes),
        }
    )


@app.route("/api/roi_data", methods=["POST"])
def api_roi_data():
    body = request.get_json(force=True, silent=True) or {}
    image_id = str(body.get("image_id") or "")
    vis = _parse_vis(body.get("roi_vis") or body.get("vis") or "zscale")

    try:
        x = int(body.get("x", 0))
        y = int(body.get("y", 0))
        size = int(body.get("size", 256))
    except Exception:
        return jsonify({"ok": False, "error": "x, y, and size must be integers."}), 400

    item = store.get(image_id)
    if item is None:
        return jsonify({"ok": False, "error": "Unknown image_id (upload again)."}), 400

    img = item["data"]
    h, w = img.shape

    size = int(max(1, min(size, MAX_ROI_SIZE, h, w)))
    x = int(max(0, min(x, w - size)))
    y = int(max(0, min(y, h - size)))

    roi = img[y : y + size, x : x + size]

    # ROI render
    rr = render_png(roi, mode=vis, max_size=None)

    # ROI stats
    stats = _finite_stats(roi)

    # Raw ROI values for cursor intensity readout on the front-end.
    # We transmit float32 in row-major order (C-contiguous) as base64.
    roi_f32 = np.ascontiguousarray(roi.astype(np.float32, copy=False))
    roi_f32_b64 = base64.b64encode(roi_f32.tobytes()).decode("ascii")

    # CRNet inference
    t0 = time.perf_counter()
    prob = infer_prob_mask(roi)  # float32 [0,1]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    prob_u8 = (prob * 255.0 + 0.5).astype(np.uint8)
    prob_b64 = base64.b64encode(prob_u8.tobytes()).decode("ascii")

    return jsonify(
        {
            "ok": True,
            "image_id": image_id,
            "x": x,
            "y": y,
            "size": size,
            "roi_vis": vis,
            "roi_png_b64": png_bytes_to_b64(rr.png_bytes),
            "roi_vmin": rr.vmin,
            "roi_vmax": rr.vmax,
            "roi_stats": stats,
            "roi_f32_b64": roi_f32_b64,
            "mask_prob_u8_b64": prob_b64,
            "infer_ms": dt_ms,
        }
    )


@app.route("/api/roi_render", methods=["POST"])
def api_roi_render():
    """
    Re-render ROI image only (no model inference) for visualization mode changes.
    """
    body = request.get_json(force=True, silent=True) or {}
    image_id = str(body.get("image_id") or "")
    vis = _parse_vis(body.get("roi_vis") or body.get("vis") or "zscale")

    try:
        x = int(body.get("x", 0))
        y = int(body.get("y", 0))
        size = int(body.get("size", 256))
    except Exception:
        return jsonify({"ok": False, "error": "x, y, and size must be integers."}), 400

    item = store.get(image_id)
    if item is None:
        return jsonify({"ok": False, "error": "Unknown image_id (upload again)."}), 400

    img = item["data"]
    h, w = img.shape

    size = int(max(1, min(size, MAX_ROI_SIZE, h, w)))
    x = int(max(0, min(x, w - size)))
    y = int(max(0, min(y, h - size)))

    roi = img[y : y + size, x : x + size]

    rr = render_png(roi, mode=vis, max_size=None)
    stats = _finite_stats(roi)

    return jsonify(
        {
            "ok": True,
            "image_id": image_id,
            "x": x,
            "y": y,
            "size": size,
            "roi_vis": vis,
            "roi_png_b64": png_bytes_to_b64(rr.png_bytes),
            "roi_vmin": rr.vmin,
            "roi_vmax": rr.vmax,
            "roi_stats": stats,
        }
    )
@app.route("/api/download_mask_fits", methods=["POST"])
def api_download_mask_fits():
    """Generate and return an edited ROI mask as a FITS file.

    The mask comes from the client (after threshold/dilation/brush edits),
    so we accept a base64-encoded uint8 array of length size*size.

    Output FITS contains a 2D uint8 image (0/1) with a few ROI metadata keywords.
    """
    body = request.get_json(force=True, silent=True) or {}
    image_id = str(body.get("image_id") or "")

    try:
        x = int(body.get("x", 0))
        y = int(body.get("y", 0))
        size = int(body.get("size", 256))
    except Exception:
        return jsonify({"ok": False, "error": "x, y, and size must be integers."}), 400

    mask_b64 = body.get("mask_u8_b64")
    if not mask_b64 or not isinstance(mask_b64, str):
        return jsonify({"ok": False, "error": "Missing mask_u8_b64."}), 400

    item = store.get(image_id)
    if item is None:
        return jsonify({"ok": False, "error": "Unknown image_id (upload again)."}), 400

    # Safety clamp
    img = item["data"]
    h, w = img.shape
    size = int(max(1, min(size, MAX_ROI_SIZE, h, w)))
    x = int(max(0, min(x, w - size)))
    y = int(max(0, min(y, h - size)))

    try:
        mask_bytes = base64.b64decode(mask_b64.encode("ascii"), validate=False)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid mask_u8_b64: {e}"}), 400

    expected = size * size
    if mask.size != expected:
        return jsonify(
            {
                "ok": False,
                "error": f"Mask has {mask.size} bytes but expected {expected} (size={size}).",
            }
        ), 400

    mask = mask.reshape((size, size)).astype(np.uint8)
    # Normalize to strictly 0/1 for clarity
    mask = (mask > 0).astype(np.uint8)

    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:
        return jsonify({"ok": False, "error": "astropy is required to write FITS."}), 500

    hdu = fits.PrimaryHDU(mask)
    hdr = hdu.header
    hdr["ROIX0"] = (x, "ROI top-left X in original image")
    hdr["ROIY0"] = (y, "ROI top-left Y in original image")
    hdr["ROISIZE"] = (size, "ROI size (square)")
    hdr["COMMENT"] = "CRNet edited ROI mask (0/1) exported from webapp"

    import io

    bio = io.BytesIO()
    hdu.writeto(bio, overwrite=True)
    bio.seek(0)

    pad = lambda n: str(int(n)).zfill(4)
    fname = f"CRNet_mask_x{pad(x)}_y{pad(y)}_s{size}.fits"

    return send_file(
        bio,
        as_attachment=True,
        download_name=fname,
        mimetype="application/fits",
        max_age=0,
    )


if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
