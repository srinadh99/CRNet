# CRNet Flask Web Application

This folder adds a **Flask-based web UI** on top of the CRNet package so you can:

- Upload a **`.fits`** image
- View the **full image** (top-right) with **ZScale / LogStretch / AsinhStretch**
- Drag a **fixed-size ROI** on the full image
- See the **ROI image** (bottom-left) with its own visualization settings + ROI min/max/mean/std
- See the **predicted cosmic ray (CR) mask** for that ROI (bottom-right)
- Post-process the mask with:
  - **Threshold**
  - **Dilation**
- Manually edit the ROI mask using a **brush** (sizes **1 / 3 / 5**) to paint pixels to **1** or erase to **0**
- Download the edited ROI mask as a PNG

---

## 1) Create an environment and install dependencies

From the **repo root** (the folder that contains `pyproject.toml`):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```

Install CRNet (editable install) + its scientific dependencies:

```bash
pip install -U pip
pip install -e .
```

Install the webapp runtime dependencies:

```bash
pip install -r webapp/requirements.txt
```

> **Note:** `astropy` is required for FITS loading and (optionally) zscale limits.

---

## 2) Choose your model weights (optional)

By default the app uses:

- `webapp/weights/ACS-WFC-BN-16.pth`

You can override with an environment variable:

```bash
export CRNET_MODEL_PATH="path/to/your_model.pth"
export CRNET_DEVICE="CPU"   # or GPU (requires CUDA build of PyTorch)
```

A template is provided in:

- `webapp/.env.example`

If your `.pth` filename doesn’t contain `BN/GN/Att` hints, you can explicitly set:

- `CRNET_ATT=true|false`
- `CRNET_ACT_NORM=batch|group|none`
- `CRNET_DEEPER=true|false`

---

## 3) Run the app

From the repo root:

```bash
python -m webapp.app
```

Then open:

- `http://127.0.0.1:5000`

---

## 4) How the UI maps to your requirements

1. **Top-right shows entire image**  
   - Konva canvas stage in the “Full image” panel.

2. **Top-left visualization features (zscale/log/asinh/minmax)**  
   - “Full image visualization” dropdown.

3. **Fixed-size ROI interactively chosen on full image**  
   - Green ROI box is draggable; click centers it.

4. **Chosen area appears bottom-left**  
   - “ROI image” panel updates when ROI changes.

5. **Predicted CR mask appears bottom-right**  
   - “CR mask” panel shows the binary mask in an editable canvas.

6. **ROI visualization + min/max shown**  
   - ROI visualization dropdown + ROI stats in the top-left panel.

7. **Mask dilation + threshold**  
   - Sliders + **Apply** button (rebuilds mask from model probability).

8. **Brush (1,3,5) paint or erase**  
   - “Brush mode” and “Brush size” controls; drag in the mask canvas.

---

## Notes / Design choices

- The backend returns a **probability mask** from CRNet (0–1), quantized to **uint8 (0–255)** for compact transfer.
- Threshold + dilation are applied **client-side** for interactivity.
- Changing threshold/dilation **resets manual edits** (the UI makes this explicit).

---

## Troubleshooting

### “astropy is required to read FITS”
Install astropy (it should already be included if you installed CRNet via `pip install -e .`):

```bash
pip install astropy
```

### GPU
If you set `CRNET_DEVICE=GPU`, you need:
- a CUDA-capable GPU
- a CUDA build of PyTorch that matches your CUDA version

For many users, CPU inference is simplest.

---

## Files you may want to customize

- `webapp/app.py` — Flask routes and in-memory image store
- `webapp/services/crnet_service.py` — model load + inference
- `webapp/services/visualization.py` — zscale/log/asinh rendering to PNG
- `webapp/static/js/app.js` — ROI interaction, threshold/dilation, brush editing
- `webapp/templates/index.html` — layout and controls
- `webapp/static/css/style.css` — styling and fixed-size square viewers
