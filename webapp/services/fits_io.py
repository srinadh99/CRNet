
"""
FITS loading utilities.

We use astropy.io.fits when available.
"""

from __future__ import annotations

import io
from typing import Tuple, Any

import numpy as np


def load_fits_from_bytes(fits_bytes: bytes) -> Tuple[np.ndarray, dict]:
    """
    Returns (data2d, meta_dict).

    - Attempts to find the first HDU that contains image data.
    - If the first image is >2D, it slices leading dimensions until 2D.
    """
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "astropy is required to read FITS files. Install it with `pip install astropy`."
        ) from e

    with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
        header0 = dict(hdul[0].header) if len(hdul) else {}
        data2d = None
        header = None

        for hdu in hdul:
            if getattr(hdu, "data", None) is None:
                continue
            arr = np.asarray(hdu.data)
            if arr.size == 0:
                continue

            # slice down to 2D if needed
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim == 2:
                data2d = arr
                header = dict(getattr(hdu, "header", {}))
                break

        if data2d is None:
            raise ValueError("No 2D image found in FITS file.")

    # Promote to float32 for stable inference and visualization
    data2d = np.asarray(data2d, dtype=np.float32)

    meta = {
        "primary_header": header0,
        "image_header": header or {},
        "shape": tuple(int(x) for x in data2d.shape),
        "dtype": str(data2d.dtype),
    }
    return data2d, meta
