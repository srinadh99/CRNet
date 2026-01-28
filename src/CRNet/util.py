import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch, LogStretch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def fillMask(img, mask, method="median", size=5):
    
    """
    Inpaint CR masked pixels.
    img, mask are 2D numpy arrays.    
    method: - "median"   : local 5x5 median filter samping
            - "maskfill" : maskfill-based inpainting  (https://iopscience.iop.org/article/10.1088/1538-3873/ad2866)
            - GitHub: (https://github.com/dokkum/maskfill/tree/main)
            - (A Robust and Simple Method for Filling in Masked Data in Astronomical Images by Pieter van Dokkum and Imad Pasha).
    """
    
    if method == "maskfill":
        try:
            from maskfill import maskfill
        except ImportError as e:
            raise ImportError("Install maskfill with: pip install maskfill") from e

        smooth, raw = maskfill(img, mask.astype(bool), size=size)
        return smooth if smooth is not None else raw

    # ---- median filter sampling ----    
    out = img.copy()
    H, W = img.shape
    global_med = np.median(img)
    valid = img * (1 - mask)
    
    ys, xs = np.nonzero(mask)    
    for y, x in zip(ys, xs):
        y0, y1 = max(y-2, 0), min(y+3, H)
        x0, x1 = max(x-2, 0), min(x+3, W)

        patch = valid[y0:y1, x0:x1]
        vals = patch[patch != 0]
        out[y, x] = np.median(vals) if vals.size else global_med

    return out

def maskMetric(PD, GT):
    """
    Compute metrics - TP, TN, FP, FN.
    """
    
    if len(PD.shape) == 2:
        PD = PD.reshape(1, *PD.shape)
    if len(GT.shape) == 2:
        GT = GT.reshape(1, *GT.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(GT.shape[0]):
        P = GT[i].sum()
        TP += (PD[i][GT[i] == 1] == 1).sum()
        TN += (PD[i][GT[i] == 0] == 0).sum()
        FP += (PD[i][GT[i] == 0] == 1).sum()
        FN += (PD[i][GT[i] == 1] == 0).sum()
    return np.array([TP, TN, FP, FN])

def plotCRDetection(img, gt_mask, pred_mask, cleaned, boundary_lw: float = 3.0,
    titles=("Image", "Ground truth", "Prediction", "Cleaned image"),
    model_name: str | None = None, save_path: str | None = None):
    
    """
    Visualize CR detection results in 4 panels.
    """

    for name, arr in [("img", img), ("gt_mask", gt_mask), ("pred_mask", pred_mask), ("cleaned", cleaned)]:
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")

    if img.shape != gt_mask.shape or img.shape != pred_mask.shape or img.shape != cleaned.shape:
        raise ValueError(f"All inputs must have the same shape. Got: "
                         f"img={img.shape}, gt={gt_mask.shape}, pred={pred_mask.shape}, cleaned={cleaned.shape}")

    # Ensure masks are 0/1
    gt = (gt_mask > 0).astype(np.uint8)
    pr = (pred_mask > 0).astype(np.uint8)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

    # --- error map for pred mask ---
    fp = (pr == 1) & (gt == 0)
    fn = (pr == 0) & (gt == 1)
    err = fp | fn

    base = (1 - pr).astype(float)
    pred_rgb = np.dstack([base, base, base])
    pred_rgb[err] = np.array([1.0, 0.0, 0.0], dtype=float)

    # --- plot ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

    axes[0].imshow(img, norm=norm, cmap="gray", origin="lower")
    axes[0].set_title(titles[0])

    axes[1].imshow(gt, vmin=0, vmax=1, cmap="gray_r", origin="lower", interpolation="nearest")
    axes[1].set_title(titles[1])

    # model name appended ONLY here
    pred_title = titles[2]
    if model_name is not None:
        pred_title = f"{pred_title} ({model_name})"

    axes[2].imshow(pred_rgb, vmin=0, vmax=1, cmap='gray_r', origin="lower", interpolation="nearest")
    axes[2].set_title(pred_title)

    axes[3].imshow(cleaned, norm=norm, cmap="gray", origin="lower")
    axes[3].set_title(titles[3])

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(boundary_lw)
            spine.set_edgecolor("black")

    # optional save
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes

def plotFirstConvFilters(model: nn.Module,
    save_path_filters: str = "first_conv_filters.png",
    save_path_dft: str = "first_conv_dft16.png",
    max_filters: int | None = 32,
    normalize_filters: bool = True,
    dft_log: bool = True,
    cols: int = 8,
    spine_lw: float = 1.2,
):
    """
    Save spatial filters and 16x16 DFT magnitudes from the first Conv2d layer
    of an already-loaded model.
    """

    net = model

    # If this is your CRNet wrapper, extract the actual torch model
    if hasattr(net, "maskNet") and isinstance(net.maskNet, nn.Module):
        net = net.maskNet

    # Handle DataParallel
    if isinstance(net, nn.DataParallel):
        net = net.module

    # Handle nested `.module`
    if hasattr(net, "module") and isinstance(getattr(net, "module"), nn.Module):
        net = net.module
    
    # Find first Conv2d
    first_conv = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if first_conv is None:
        raise RuntimeError("No nn.Conv2d layer found in the model.")

    w = first_conv.weight.detach().cpu().numpy()  # (out_ch, in_ch, kH, kW)
    out_ch, in_ch, kH, kW = w.shape

    filt2d = w[:, 0] if in_ch == 1 else w.mean(axis=1)

    n = out_ch if max_filters is None else min(out_ch, int(max_filters))
    rows = int(np.ceil(n / cols))

    # Helpers
    def norm01(a):
        mn, mx = float(a.min()), float(a.max())
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)

    def draw_box(ax):
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(spine_lw)
            
    def save_grid(images, out_path, vmin=None, vmax=None, show=True):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), squeeze=False)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            draw_box(ax)

            if i >= n:
                ax.imshow(np.zeros_like(images[0]), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(images[i], cmap="gray", vmin=vmin, vmax=vmax)

        fig.tight_layout()

        if show:
            plt.show()          

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
            

    # (1) Spatial filters
    spatial_imgs = []
    for i in range(n):
        f = filt2d[i].astype(np.float32)
        if normalize_filters:
            f = norm01(f)
        spatial_imgs.append(f)

    save_grid(
        spatial_imgs,
        save_path_filters,
        vmin=0 if normalize_filters else None,
        vmax=1 if normalize_filters else None,
    )

    # (2) 16x16 DFT magnitudes
    dft_imgs = []
    for i in range(n):
        f = filt2d[i].astype(np.float32)

        f16 = np.zeros((16, 16), dtype=np.float32)
        y0 = (16 - kH) // 2
        x0 = (16 - kW) // 2
        f16[y0:y0+kH, x0:x0+kW] = f

        F = np.fft.fftshift(np.fft.fft2(f16))
        mag = np.abs(F)
        if dft_log:
            mag = np.log1p(mag)
        dft_imgs.append(norm01(mag))

    save_grid(dft_imgs, save_path_dft, vmin=0, vmax=1)

    return save_path_filters, save_path_dft

