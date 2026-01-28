""" Main module to instantiate CRNet models and use them """

from __future__ import annotations

from os import path, mkdir
import math
import shutil
import secrets

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import from_numpy
from joblib import Parallel, delayed
from joblib import dump, load
from joblib import wrap_non_picklable_objects

from .unet import UNet, WrappedModel
from .util import fillMask

__all__ = ["CRNet"]


class CRNet:
    """
    CRNet wrapper that: (1) predicts CR mask with a UNet and (2) (optionally) inpaints using util.fillMask 

    Notes:
      - For n_jobs>1 (joblib multiprocessing), this implementation supports CPU only.
        (Torch models on GPU are not safely shareable across joblib worker processes.)
    """

    def __init__(
        self,
        mask: str = "ACS-WFC",
        device: str = "GPU",
        hidden: int | None = None,
        model_kwargs: dict | None = None,
        scale: float = 1.0,
        norm: bool = True,
    ):
        """
        Parameters
        ----------
        mask : str
            File path to a saved mask model (.pth). 
        device : str
            One of 'CPU' or 'GPU'
        hidden : int
            UNet hidden channels (used only if you later want to create an uninitialized model)
        model_kwargs : dict, optional
            Extra kwargs passed to UNet(...). If None, defaults are used.
        scale : float
            Multiplicative scale used when pre-scaling inputs (img/scale).
        norm : bool
            If True, standardize image by mean/std before inference and un-standardize after inpainting.
        """
        device = device.upper().strip()
        if device not in {"CPU", "GPU"}:
            raise ValueError("device must be 'CPU' or 'GPU'")

        if device == "GPU" and not torch.cuda.is_available():
            raise AssertionError("No CUDA device detected!")

        self.device = torch.device("cuda" if device == "GPU" else "cpu")
        self.is_cuda = (self.device.type == "cuda")
        self.wrapper = nn.DataParallel if self.is_cuda else WrappedModel

        # Model loading
        if not (isinstance(mask, str) and mask.lower().endswith(".pth") and path.exists(mask)):
            raise FileNotFoundError(
                "mask must be a valid path to a .pth file."
            )
        self.mask_path = mask
                
        # Build UNet architecture (must match the saved weights!)
        mk = dict(n_channels=1, n_classes=1, hidden=hidden, act_norm="batch", att=False, deeper=False)
        if model_kwargs:
            mk.update(model_kwargs)
        net = UNet(**mk)

        self.maskNet = self.wrapper(net).to(self.device)

        state = torch.load(self.mask_path, map_location=self.device)
        self.maskNet.load_state_dict(state)
        self.maskNet.eval()
        for p in self.maskNet.parameters():
            p.requires_grad = False

        # preprocessing config
        self.scale = float(scale)
        self.norm = bool(norm)
        self.mean = None
        self.std = None

    # Public API
        
    def clean(
        self,
        img0: np.ndarray,
        threshold: float = 0.5,
        inpaint: bool = False,
        binary: bool = True,
        segment: bool = True,
        patch: int = 1024,
        n_jobs: int = 1,
        fill_method: str = "median",
        fill_size: int = 5,
    ):
        """
        Identify cosmic rays in an input image, and (optionally) inpaint using fillMask.

        Parameters
        ----------
        img0 : np.ndarray
            2D input image.
        threshold : float
            Threshold applied to probabilistic mask to generate binary mask.
        inpaint : bool
            If True, also return clean inpainted image.
        binary : bool
            If True return binary CR mask. If False return probabilistic mask.
        segment : bool
            If True, segment image into patch x patch tiles for memory control.
        patch : int
            Patch size when segment=True.
        n_jobs : int
            Number of jobs for parallel segmentation (CPU only).
        fill_method : str
            Passed to util.fillMask (e.g., "median" or "maskfill").
        fill_size : int
            Passed to util.fillMask size argument.

        Returns
        -------
        mask  or (mask, img_clean)
        """
        if img0.ndim != 2:
            raise ValueError("img0 must be a 2D array")

        if self.is_cuda and n_jobs != 1:
            raise ValueError("n_jobs>1 is supported only on CPU (set device='CPU' or n_jobs=1).")

        # Pre-scale + optional standardization
        img = img0.astype(np.float32) / self.scale
        img = img.copy()

        if self.norm:
            self.mean = float(img.mean())
            self.std = float(img.std()) if float(img.std()) > 0 else 1.0
            img = (img - self.mean) / self.std

        if (not segment) and n_jobs == 1:
            out = self.clean_(img, threshold=threshold, inpaint=inpaint, binary=binary,
                              fill_method=fill_method, fill_size=fill_size)
        else:
            if n_jobs == 1:
                out = self.clean_large(img, threshold=threshold, inpaint=inpaint, binary=binary,
                                       patch=patch, fill_method=fill_method, fill_size=fill_size)
            else:
                out = self.clean_large_parallel(img, threshold=threshold, inpaint=inpaint, binary=binary,
                                                patch=patch, n_jobs=n_jobs, fill_method=fill_method, fill_size=fill_size)

        return out

    # Core inference on a tile
    
    def clean_(
        self,
        img0: np.ndarray,
        threshold: float = 0.5,
        inpaint: bool = True,
        binary: bool = True,
        fill_method: str = "median",
        fill_size: int = 5,
    ):
        """
        Run inference on a single image (or a tile). Handles padding to /4.

        Returns
        -------
        mask  or (mask, img_clean)
        """
        # pad to be divisible by 4
        shape = img0.shape
        pad_x = (-shape[0]) % 4
        pad_y = (-shape[1]) % 4
        if pad_x or pad_y:
            img0p = np.pad(img0, ((pad_x, 0), (pad_y, 0)), mode="constant")
        else:
            img0p = img0

        x = from_numpy(img0p).to(self.device, dtype=torch.float32).view(1, 1, *img0p.shape)

        with torch.no_grad():
            mask_prob = self.maskNet(x)  # (1,1,H,W) sigmoid already in UNet forward
        mask_prob_np = mask_prob.detach().cpu().squeeze(0).squeeze(0).numpy()

        if binary:
            mask_out = (mask_prob_np > threshold).astype(np.uint8)
        else:
            mask_out = mask_prob_np.astype(np.float32)

        # Unpad mask to original size
        mask_out = mask_out[pad_x:, pad_y:]

        if not inpaint:
            return mask_out

        # Always inpaint via fillMask 
        bin_mask = (mask_prob_np > threshold).astype(np.uint8)[pad_x:, pad_y:]
        img_orig = img0p[pad_x:, pad_y:]  # standardized (if norm=True) and pre-scaled

        img_filled = fillMask(img_orig, bin_mask, method=fill_method, size=fill_size)

        # Undo standardization + rescale back
        if self.norm:
            img_filled = img_filled * self.std + self.mean

        img_filled = img_filled * self.scale
        return mask_out, img_filled

    # Large image helpers
    
    def clean_large(
        self,
        img0: np.ndarray,
        threshold: float = 0.5,
        inpaint: bool = True,
        binary: bool = True,
        patch: int = 256,
        fill_method: str = "median",
        fill_size: int = 5,
    ):
        """
        Segment image into tiles and run clean_ sequentially.
        """
        im_shape = img0.shape
        hh = int(math.ceil(im_shape[0] / patch))
        ww = int(math.ceil(im_shape[1] / patch))

        mask = np.zeros(im_shape, dtype=np.uint8 if binary else np.float32)
        img1 = np.zeros(im_shape, dtype=np.float32) if inpaint else None

        for i in range(hh):
            for j in range(ww):
                y0 = i * patch
                y1 = min((i + 1) * patch, im_shape[0])
                x0 = j * patch
                x1 = min((j + 1) * patch, im_shape[1])

                tile = img0[y0:y1, x0:x1]
                if inpaint:
                    m, c = self.clean_(tile, threshold=threshold, inpaint=True, binary=binary,
                                       fill_method=fill_method, fill_size=fill_size)
                    mask[y0:y1, x0:x1] = m
                    img1[y0:y1, x0:x1] = c
                else:
                    m = self.clean_(tile, threshold=threshold, inpaint=False, binary=binary,
                                    fill_method=fill_method, fill_size=fill_size)
                    mask[y0:y1, x0:x1] = m

        return (mask, img1) if inpaint else mask

    def clean_large_parallel(
        self,
        img0: np.ndarray,
        threshold: float = 0.5,
        inpaint: bool = True,
        binary: bool = True,
        patch: int = 256,
        n_jobs: int = -1,
        fill_method: str = "median",
        fill_size: int = 5,
    ):
        """
        Parallel tiling for CPU only.
        """
        if self.is_cuda:
            raise ValueError("clean_large_parallel is CPU-only. Use n_jobs=1 or device='CPU'.")

        folder = "./joblib_memmap_" + secrets.token_hex(3)
        try:
            mkdir(folder)
        except FileExistsError:
            folder = "./joblib_memmap_" + secrets.token_hex(3)
            mkdir(folder)

        im_shape = img0.shape
        img0_dtype = img0.dtype
        hh = int(math.ceil(im_shape[0] / patch))
        ww = int(math.ceil(im_shape[1] / patch))

        img0_filename_memmap = path.join(folder, "img0_memmap")
        dump(img0, img0_filename_memmap)
        img0_mm = load(img0_filename_memmap, mmap_mode="r")

        if inpaint:
            img1_filename_memmap = path.join(folder, "img1_memmap")
            img1 = np.memmap(img1_filename_memmap, dtype=np.float32, shape=im_shape, mode="w+")
        else:
            img1 = None

        mask_filename_memmap = path.join(folder, "mask_memmap")
        mask = np.memmap(
            mask_filename_memmap,
            dtype=np.int8 if binary else img0_dtype,
            shape=im_shape,
            mode="w+",
        )

        @wrap_non_picklable_objects
        def fill_values(i, j, img0_mem, img1_mem, mask_mem):
            y0 = i * patch
            y1 = min((i + 1) * patch, im_shape[0])
            x0 = j * patch
            x1 = min((j + 1) * patch, im_shape[1])

            tile = img0_mem[y0:y1, x0:x1]

            if inpaint:
                m, c = self.clean_(
                    tile,
                    threshold=threshold,
                    inpaint=True,
                    binary=binary,
                    fill_method=fill_method,
                    fill_size=fill_size,
                )
                mask_mem[y0:y1, x0:x1] = m
                img1_mem[y0:y1, x0:x1] = c
            else:
                m = self.clean_(
                    tile,
                    threshold=threshold,
                    inpaint=False,
                    binary=binary,
                    fill_method=fill_method,
                    fill_size=fill_size,
                )
                mask_mem[y0:y1, x0:x1] = m

        Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fill_values)(i, j, img0_mm, img1, mask) for i in range(hh) for j in range(ww)
        )

        mask_out = np.array(mask)
        img1_out = np.array(img1) if inpaint else None

        try:
            shutil.rmtree(folder)
        except Exception:
            print("Could not clean-up automatically.")

        return (mask_out, img1_out) if inpaint else mask_out

    # Standalone inpaint helper (optional public API)
    
    def inpaint(self, img0: np.ndarray, mask: np.ndarray, method: str = "median", size: int = 5):
        """
        Inpaint img0 under mask using util.fillMask only (no learned inpaint model).

        Parameters
        ----------
        img0 : np.ndarray
        mask : np.ndarray (0/1)
        method : str
        size : int

        Returns
        -------
        np.ndarray
        """
        img = img0.astype(np.float32) / self.scale
        if self.norm:
            mu = float(img.mean())
            sd = float(img.std()) if float(img.std()) > 0 else 1.0
            img = (img - mu) / sd

        filled = fillMask(img, mask.astype(np.uint8), method=method, size=size)

        if self.norm:
            filled = filled * sd + mu

        return filled * self.scale
