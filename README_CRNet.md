# CRNet

Deep learning framework for **Cosmic Ray (CR) detection** in astronomical images.

CRNet provides a small, Python-first API to:

- **predict a CR mask** from a 2D image using a U-Net model (PyTorch)
- **optionally “clean” CR pixels** via classical inpainting (`median` filter or optional `maskfill`)
- **train new models** on paired image/mask datasets
- **evaluate models** with ROC curves and compare against **L.A.Cosmic** (`astroscrappy`)

> Note: CRNet currently *learns the mask*. The “cleaned image” is produced by **inpainting masked pixels**
> (not by a learned model).

---

## Install (editable)

From the repo root (the folder that contains `pyproject.toml`):

### 1) Create an environment

```bash
conda create -n crnet python=3.10
conda activate crnet
```

### 2) Install PyTorch

Install a PyTorch build that matches your system (CPU or CUDA).  
(How you install PyTorch depends on your platform/CUDA version.)

### 3) Install CRNet

```bash
pip install -U pip
pip install -e .
```

---

## Pretrained weights included in this repo

This repo ships example trained weights you can use immediately. You pass the **path to a `.pth` file**
when creating a `CRNet` model.

Common locations:

- `HST/learned_models/`
  - `HST/learned_models/ACS-WFC-BN-16.pth`
  - `HST/learned_models/ACS-WFC-GN-16.pth`
  - `HST/learned_models/ACS-WFC-AttBN-16.pth`
  - `HST/learned_models/ACS-WFC-AttGN-16.pth`

### Model kwargs must match the weights

When loading weights, the UNet architecture settings must match what the weights expect.
In CRNet this is controlled by `model_kwargs`.

For the included HST/ACS-WFC weights in this repo, the following mappings work:

- **BN models** (`...-BN...pth`): `act_norm="batch"`
- **GN models** (`...-GN...pth`): `act_norm="group"`
- **Att models** (`...Att...pth`): `att=True`
- These weights use `hidden=32` and `deeper=False`

Example:

```python
model_kwargs = {"hidden": 32, "act_norm": "batch", "att": False, "deeper": False}
# or, for attention + groupnorm:
# model_kwargs = {"hidden": 32, "act_norm": "group", "att": True, "deeper": False}
```

---

## Quickstart: detect CRs in a FITS image

CRNet expects a **2D numpy array**.

```python
import numpy as np
from astropy.io import fits
from CRNet import CRNet

# Load a 2D image (pick the right extension for your FITS file)
img = fits.getdata("jdba2sooq_flc.fits").astype(np.float32)

crnet = CRNet(
    mask="webapp3/weights/ACS-WFC-BN.pth",
    device="GPU",  # "GPU" or "CPU"
    model_kwargs={"hidden": 32, "act_norm": "batch", "att": False, "deeper": False},
)

# Predict a binary mask + inpaint (“clean”) the image
mask, cleaned = crnet.clean(
    img,
    threshold=0.5,
    inpaint=True,
    segment=True,   # tile large images
    patch=1024,     # tile size (pixels)
    n_jobs=1,       # CPU-only parallelism (use >1 only with device="CPU")
)

# mask is 0/1 (uint8), cleaned is float image with masked pixels filled
print(mask.shape, cleaned.shape)
```

### Probabilistic mask (no thresholding)

```python
prob = crnet.clean(img, binary=False, inpaint=False)  # float32 in [0, 1]
```

---

## Python API

The package exports the most common entry points at the top level:

```python
from CRNet import CRNet, train, roc, roc_lacosmic
from CRNet import plotCRDetection, plotFirstConvFilters
```

### `CRNet.CRNet` — inference wrapper

```python
from CRNet import CRNet
crnet = CRNet(mask="path/to/model.pth", device="GPU", model_kwargs={...})
```

**Constructor parameters (most used):**
- `mask` (str): path to a `.pth` weights file
- `device` (str): `"GPU"` or `"CPU"`
- `model_kwargs` (dict): UNet settings (must match weights), e.g.
  `{"hidden": 32, "act_norm": "batch", "att": False, "deeper": False}`
- `scale` (float): input pre-scale factor (image is divided by `scale` internally)
- `norm` (bool): if `True`, standardizes by mean/std before inference (recommended)

#### `clean(img0, ...)`

Detect CRs in `img0` and optionally inpaint masked pixels.

```python
mask = crnet.clean(img0, inpaint=False)
mask, cleaned = crnet.clean(img0, inpaint=True)
```

Key options:
- `threshold` (float): probability threshold for binary masks (default `0.5`)
- `binary` (bool): `True` returns 0/1 mask, `False` returns probabilities
- `segment` (bool): tile large images to control memory
- `patch` (int): tile size (pixels) when `segment=True`
- `n_jobs` (int): parallel tiling **CPU only** (`-1` uses all cores)
- `fill_method` (str): `"median"` (default) or `"maskfill"`
- `fill_size` (int): median window size / inpainting size (default `5`)

#### `inpaint(img0, mask, ...)`

Just inpaint pixels under an existing mask (no neural net step):

```python
cleaned = crnet.inpaint(img0, mask, method="median", size=5)
```

---

## Training a model

Training is provided via the `CRNet.training.train` class (exported as `CRNet.train`).

### Data formats supported

You can train from either:

1) **In-memory arrays**
- `image`: `(N, H, W)` CR-affected images
- `mask`:  `(N, H, W)` ground-truth CR masks (0/1)
- `ignore` (optional): `(N, H, W)` pixels to ignore in loss/metrics
- `sky` (optional): `(N,)` background levels (used only if you want sky augmentation)

2) **A list of `.npy` paths**
Each `.npy` file should contain an array shaped `(2, H, W)` or `(3, H, W)`:
- `[0]` = image
- `[1]` = mask
- `[2]` = ignore (optional)

If a `sky.npy` is present in the same directory, training can use it for background augmentation.

### Example training run

```python
import numpy as np
from CRNet.training import train

# toy example shapes
images = np.random.rand(100, 64, 64).astype("float32")
masks  = (np.random.rand(100, 64, 64) > 0.98).astype("float32")

trainer = train(
    image=images,
    mask=masks,
    name="my_crnet",
    hidden=32,
    model_kwargs={"act_norm": "batch", "att": False, "deeper": False},
    epoch=50,
    batch_size=16,
    lr=1e-3,
    directory="./outputs/",
)

trainer.train()
trainer.save()   # writes ./outputs/my_crnet.pth
```

Then load the trained model for inference:

```python
from CRNet import CRNet
crnet = CRNet(mask="./outputs/my_crnet.pth", device="GPU",
              model_kwargs={"hidden": 32, "act_norm": "batch", "att": False, "deeper": False})
```

---

## Evaluation (ROC curves)

### CRNet ROC

```python
import numpy as np
from CRNet.evaluate import roc

tpr, fpr = roc(
    model=crnet,
    image=test_images,
    mask=test_masks,
    ignore=test_ignore,                # optional
    thresholds=np.linspace(0.001, 0.999, 500),
)
```

### L.A.Cosmic baseline (astroscrappy)

```python
from CRNet.evaluate import roc_lacosmic

sigclip = np.linspace(1, 10, 50)      # example grid
(tpr, fpr), (tpr_d, fpr_d) = roc_lacosmic(sigclip, image=test_images, mask=test_masks, dilate=True, rad=1)
```

---

## Visualization utilities

```python
from CRNet.util import plotCRDetection, plotFirstConvFilters

plotCRDetection(img, gt_mask, pred_mask, cleaned, save_path="example.png")
plotFirstConvFilters(crnet, save_path_filters="filters.png", save_path_dft="filters_dft.png")
```

---

## Optional: Flask web UI

This repo also contains an optional Flask-based web UI in `webapp/`.

The core CRNet library does **not** require Flask.  
To run the web app, install:

```bash
python -m webapp.app
```

---

## License

MIT License (see `pyproject.toml`).
