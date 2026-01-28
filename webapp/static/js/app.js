// CRNet Web App front-end logic
// - Upload FITS
// - Display full image with ROI rectangle (fixed size) on Konva canvas
// - Fetch ROI image + model probability mask
// - Apply threshold + dilation to produce editable binary mask
// - Brush tool to edit mask pixels (1/3/5) paint or erase

const state = {
  imageId: null,

  origW: 0,
  origH: 0,
  dispW: 0,
  dispH: 0,
  scaleX: 1, // orig pixels per display pixel
  scaleY: 1,

  fullVis: "zscale",
  roiVis: "zscale",

  roiSize: 256, // in ORIGINAL pixels (square)
  roiX: 0,      // in ORIGINAL pixels (top-left)
  roiY: 0,

  probU8: null,     // Uint8Array length roiSize*roiSize
  maskU8: null,     // Uint8Array current editable binary mask (0/1)

  // Raw ROI pixel values (float32) for cursor intensity readout.
  // Length = roiSize * roiSize, row-major.
  roiF32: null,

  threshold: 0.5,
  dilation: 0,

  brushMode: "paint",
  brushSize: 1,

  inferMs: null,
};

// DOM
const el = {
  statusBadge: document.getElementById("statusBadge"),

  fitsFile: document.getElementById("fitsFile"),

  fullVis: document.getElementById("fullVis"),
  roiSize: document.getElementById("roiSize"),

  fullShape: document.getElementById("fullShape"),
  fullMinMax: document.getElementById("fullMinMax"),
  fullMeanStd: document.getElementById("fullMeanStd"),

  fullMinMaxHeader: document.getElementById("fullMinMaxHeader"),

  roiXY: document.getElementById("roiXY"),
  roiMinMax: document.getElementById("roiMinMax"),
  roiMinMaxInline: document.getElementById("roiMinMaxInline"),
  roiMeanStdInline: document.getElementById("roiMeanStdInline"),
  roiCursorXY: document.getElementById("roiCursorXY"),
  roiMeanStd: document.getElementById("roiMeanStd"),
  inferMs: document.getElementById("inferMs"),

  fullStageContainer: document.getElementById("fullStageContainer"),

  roiVis: document.getElementById("roiVis"),
  roiMinVal: document.getElementById("roiMinVal"),
  roiMaxVal: document.getElementById("roiMaxVal"),
  roiImage: document.getElementById("roiImage"),
  roiViewer: document.getElementById("roiViewer"),
  roiXYOverlay: document.getElementById("roiXYOverlay"),

  maskThreshold: document.getElementById("maskThreshold"),
  maskThresholdVal: document.getElementById("maskThresholdVal"),
  maskDilation: document.getElementById("maskDilation"),
  maskDilationVal: document.getElementById("maskDilationVal"),
  maskCursorXY: document.getElementById("maskCursorXY"),
  maskXYOverlay: document.getElementById("maskXYOverlay"),

  btnApplyMask: document.getElementById("btnApplyMask"),
  btnResetMask: document.getElementById("btnResetMask"),

  btnDownloadMask: document.getElementById("btnDownloadMask"),
  btnDownloadMaskFits: document.getElementById("btnDownloadMaskFits"),

  brushMode: document.getElementById("brushMode"),
  brushSize: document.getElementById("brushSize"),
  maskCanvas: document.getElementById("maskCanvas"),
  maskViewer: document.getElementById("maskViewer"),
};

// -------- Mouse wheel zoom (ROI image + mask) --------
// - ROI image: wheel zoom, drag pan, dblclick reset
// - Mask: wheel zoom, shift+drag pan, dblclick reset
const viewer = {
  roi: { scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0, min: 0.5, max: 12 },
  mask: { scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0, min: 0.5, max: 12 },
};

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function applyTransform(targetEl, v) {
  targetEl.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`;
}

function resetView(targetEl, v) {
  v.scale = 1;
  v.tx = 0;
  v.ty = 0;
  applyTransform(targetEl, v);
}

function installWheelZoom(viewportEl, targetEl, v) {
  if (!viewportEl || !targetEl) return;

  // Initial
  applyTransform(targetEl, v);

  viewportEl.addEventListener(
    "wheel",
    (e) => {
      // Zoom around the cursor
      e.preventDefault();

      const rect = viewportEl.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;

      const oldScale = v.scale;
      const zoomBase = e.ctrlKey ? 1.02 : 1.08;
      const newScale = clamp(
        oldScale * (e.deltaY < 0 ? zoomBase : 1 / zoomBase),
        v.min,
        v.max
      );

      // Keep mouse point stable
      const mx = (px - v.tx) / oldScale;
      const my = (py - v.ty) / oldScale;

      v.scale = newScale;
      v.tx = px - mx * newScale;
      v.ty = py - my * newScale;

      applyTransform(targetEl, v);
    },
    { passive: false }
  );

  viewportEl.addEventListener("dblclick", () => resetView(targetEl, v));
}

function installPan(viewportEl, targetEl, v, { requireShift = false } = {}) {
  if (!viewportEl || !targetEl) return;

  viewportEl.addEventListener(
    "mousedown",
    (e) => {
      if (e.button !== 0) return; // left only
      if (requireShift && !e.shiftKey) return;

      v.dragging = true;
      v.lastX = e.clientX;
      v.lastY = e.clientY;

      // Stop other handlers (e.g., brush painting) when panning the mask
      if (requireShift) {
        e.preventDefault();
        e.stopPropagation();
      }
    },
    // Capture so we can intercept before canvas brush handlers when needed
    { capture: true }
  );

  viewportEl.addEventListener(
    "mousemove",
    (e) => {
      if (!v.dragging) return;
      if (requireShift && !e.shiftKey) return;

      const dx = e.clientX - v.lastX;
      const dy = e.clientY - v.lastY;
      v.lastX = e.clientX;
      v.lastY = e.clientY;

      v.tx += dx;
      v.ty += dy;
      applyTransform(targetEl, v);

      if (requireShift) {
        e.preventDefault();
        e.stopPropagation();
      }
    },
    { capture: true }
  );

  window.addEventListener("mouseup", () => {
    v.dragging = false;
  });
  viewportEl.addEventListener("mouseleave", () => {
    v.dragging = false;
  });
}

// -------- Cursor coordinate readout (ROI image + mask) --------
// Shows ROI-local pixel indices (x,y) as the user moves the cursor.
function getLocalXYFromEvent(evt, targetEl, size) {
  if (!targetEl) return null;
  if (!Number.isFinite(size) || size <= 0) return null;

  const rect = targetEl.getBoundingClientRect();
  const px = evt.clientX;
  const py = evt.clientY;

  if (px < rect.left || px >= rect.right || py < rect.top || py >= rect.bottom) return null;

  const x = Math.floor(((px - rect.left) / rect.width) * size);
  const y = Math.floor(((py - rect.top) / rect.height) * size);
  if (x < 0 || y < 0 || x >= size || y >= size) return null;
  return { x, y };
}

function getRoiValueAt(x, y) {
  if (!state.roiF32) return null;
  const size = state.roiSize | 0;
  const idx = (y * size + x) | 0;
  if (idx < 0 || idx >= state.roiF32.length) return null;
  const v = state.roiF32[idx];
  return Number.isFinite(v) ? v : null;
}

function getMaskValueAt(x, y) {
  if (!state.maskU8) return null;
  const size = state.roiSize | 0;
  const idx = (y * size + x) | 0;
  if (idx < 0 || idx >= state.maskU8.length) return null;
  return state.maskU8[idx];
}

function installCursorReadout(viewportEl, targetEl, outEl, overlayEl, getValueAt, formatValue) {
  if (!viewportEl || !targetEl || !outEl) return;

  // Tooltip-style overlay that follows the cursor.
  // We keep outEl updated (hidden in DOM) for debugging or future UI use.
  viewportEl.addEventListener("mousemove", (evt) => {
    const p = getLocalXYFromEvent(evt, targetEl, state.roiSize);
    outEl.textContent = p ? `${p.x}, ${p.y}` : "—";

    if (!overlayEl) return;

    if (!p) {
      overlayEl.style.display = "none";
      return;
    }

    const v = (getValueAt ? getValueAt(p.x, p.y) : null);
    const vStr = (v === null || v === undefined)
      ? "—"
      : (formatValue ? formatValue(v) : String(v));

    // Render coords + value (I=) in the tooltip.
    overlayEl.innerHTML = `<span class="xycoord">(x,y)=(${p.x}, ${p.y})</span> <span class="ival">I=${vStr}</span>`;
    overlayEl.style.display = "block";

    // Position near cursor within the viewport bounds.
    const vr = viewportEl.getBoundingClientRect();
    const localX = evt.clientX - vr.left;
    const localY = evt.clientY - vr.top;

    // Slight offset so the label doesn't sit directly under the cursor.
    const offset = 12;
    let left = localX + offset;
    let top = localY + offset;

    // Clamp so the label stays inside the viewer.
    // (overlayEl must be display:block to have non-zero dimensions)
    const ow = overlayEl.offsetWidth || 0;
    const oh = overlayEl.offsetHeight || 0;
    const pad = 4;
    const maxLeft = viewportEl.clientWidth - ow - pad;
    const maxTop = viewportEl.clientHeight - oh - pad;

    left = clamp(left, pad, Math.max(pad, maxLeft));
    top = clamp(top, pad, Math.max(pad, maxTop));

    overlayEl.style.left = `${left}px`;
    overlayEl.style.top = `${top}px`;
  });

  viewportEl.addEventListener("mouseleave", () => {
    outEl.textContent = "—";
    if (overlayEl) overlayEl.style.display = "none";
  });
}

function setStatus(text, kind="secondary") {
  el.statusBadge.className = `badge text-bg-${kind}`;
  el.statusBadge.textContent = text;
}

function fmtNum(x) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return "—";
  // scientific for big/small
  const ax = Math.abs(x);
  if ((ax > 0 && ax < 1e-3) || ax >= 1e6) return x.toExponential(3);
  return x.toFixed(3);
}

function b64ToUint8Array(b64) {
  const binStr = atob(b64);
  const len = binStr.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binStr.charCodeAt(i);
  return bytes;
}

function b64ToFloat32Array(b64) {
  // Decode base64 float32 (little-endian) into a Float32Array.
  const u8 = b64ToUint8Array(b64);
  const n = Math.floor(u8.byteLength / 4);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = dv.getFloat32(i * 4, true);
  }
  return out;
}

function uint8ArrayToB64(bytes) {
  // Chunked conversion to avoid call-stack limits for large ROIs.
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, i + chunk);
    // `apply` accepts typed arrays as array-like in modern browsers.
    binary += String.fromCharCode.apply(null, sub);
  }
  return btoa(binary);
}

function pngB64ToDataUrl(pngB64) {
  return `data:image/png;base64,${pngB64}`;
}

// --- Mask UI helpers (threshold/dilation live in the header now) ---
function readThresholdFromUI({ snap = false } = {}) {
  // snap=false  -> don't rewrite the input while the user is typing
  // snap=true   -> clamp + format the input (e.g., on blur/change)
  let v = parseFloat(el.maskThreshold.value);
  if (!Number.isFinite(v)) v = 0.5;

  // Clamp for internal state
  v = Math.max(0, Math.min(1, v));
  state.threshold = v;
  el.maskThresholdVal.textContent = v.toFixed(2);

  if (snap) {
    el.maskThreshold.value = v.toFixed(2);
  }
}

function readDilationFromUI() {
  let r = parseInt(el.maskDilation.value, 10);
  if (!Number.isFinite(r)) r = 0;
  r = Math.max(0, Math.min(7, r));
  state.dilation = r;
  el.maskDilation.value = String(r);
  el.maskDilationVal.textContent = String(r);
}

// -------- Konva full image + ROI rectangle --------
let stage = null;
let layer = null;
let konvaImg = null;
let roiRect = null;

function initKonvaStage() {
  // Clear container
  el.fullStageContainer.innerHTML = "";

  stage = new Konva.Stage({
    container: "fullStageContainer",
    width: state.dispW,
    height: state.dispH,
  });

  layer = new Konva.Layer();
  stage.add(layer);

  konvaImg = new Konva.Image({
    x: 0,
    y: 0,
    width: state.dispW,
    height: state.dispH,
  });
  layer.add(konvaImg);

  roiRect = new Konva.Rect({
    x: 0,
    y: 0,
    width: roiDisplaySize(),
    height: roiDisplaySizeY(),
    stroke: "lime",
    strokeWidth: 2,
    draggable: true,
  });

  roiRect.dragBoundFunc(function(pos) {
    // Keep ROI within image bounds (display coords)
    const w = roiRect.width();
    const h = roiRect.height();
    let x = pos.x;
    let y = pos.y;
    x = Math.max(0, Math.min(x, state.dispW - w));
    y = Math.max(0, Math.min(y, state.dispH - h));
    return {x, y};
  });

  roiRect.on("dragend", () => {
    updateRoiOrigFromRect();
    requestRoiData();
  });

  layer.add(roiRect);
  layer.draw();

  // Click to center ROI
  stage.on("click", (evt) => {
    // ignore clicks that start drag on ROI
    const target = evt.target;
    if (target === roiRect) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const w = roiRect.width();
    const h = roiRect.height();
    const x = Math.max(0, Math.min(pointer.x - w / 2, state.dispW - w));
    const y = Math.max(0, Math.min(pointer.y - h / 2, state.dispH - h));

    roiRect.position({x, y});
    layer.draw();

    updateRoiOrigFromRect();
    requestRoiData();
  });
}

function roiDisplaySize() {
  return state.roiSize / state.scaleX;
}
function roiDisplaySizeY() {
  return state.roiSize / state.scaleY;
}

function updateRoiRectSize() {
  if (!roiRect) return;
  roiRect.width(roiDisplaySize());
  roiRect.height(roiDisplaySizeY());

  // Ensure inside bounds
  const w = roiRect.width();
  const h = roiRect.height();
  let x = roiRect.x();
  let y = roiRect.y();
  x = Math.max(0, Math.min(x, state.dispW - w));
  y = Math.max(0, Math.min(y, state.dispH - h));
  roiRect.position({x, y});
  layer.draw();

  updateRoiOrigFromRect();
}

function updateRoiOrigFromRect() {
  const xDisp = roiRect.x();
  const yDisp = roiRect.y();
  state.roiX = Math.round(xDisp * state.scaleX);
  state.roiY = Math.round(yDisp * state.scaleY);

  el.roiXY.textContent = `${state.roiX}, ${state.roiY} (size=${state.roiSize})`;
}

// -------- Mask rendering + processing --------
const maskCtx = el.maskCanvas.getContext("2d", { willReadFrequently: true });

function setMaskCanvasSize(size) {
  el.maskCanvas.width = size;
  el.maskCanvas.height = size;
}

function renderMaskCanvas() {
  if (!state.maskU8) return;
  const size = state.roiSize;
  setMaskCanvasSize(size);
  const imgData = maskCtx.createImageData(size, size);
  const data = imgData.data;
  // white=1, black=0
  for (let i = 0; i < state.maskU8.length; i++) {
    const v = state.maskU8[i] ? 255 : 0;
    const j = i * 4;
    data[j] = v;
    data[j+1] = v;
    data[j+2] = v;
    data[j+3] = 255;
  }
  maskCtx.putImageData(imgData, 0, 0);
}

function applyThresholdAndDilationFromProb() {
  if (!state.probU8) return;

  const size = state.roiSize;
  const out = new Uint8Array(state.probU8.length);

  const thr = Math.max(0, Math.min(1, state.threshold));
  const thrU8 = Math.round(thr * 255);

  for (let i = 0; i < out.length; i++) {
    out[i] = (state.probU8[i] >= thrU8) ? 1 : 0;
  }

  const r = state.dilation | 0;
  state.maskU8 = (r > 0) ? dilateBinary(out, size, size, r) : out;

  renderMaskCanvas();
}

function buildDiskOffsets(r) {
  const off = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx*dx + dy*dy <= r*r) off.push([dx, dy]);
    }
  }
  return off;
}

function dilateBinary(maskU8, w, h, r) {
  const out = new Uint8Array(maskU8.length);
  const offsets = buildDiskOffsets(r);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      // if any neighbor is 1 => 1
      for (const [dx, dy] of offsets) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        if (maskU8[yy*w + xx]) { v = 1; break; }
      }
      out[y*w + x] = v;
    }
  }
  return out;
}

// -------- Brush editing --------
let painting = false;
let needsRender = false;

function scheduleMaskRender() {
  if (needsRender) return;
  needsRender = true;
  requestAnimationFrame(() => {
    needsRender = false;
    renderMaskCanvas();
  });
}

function paintAtEvent(evt) {
  if (!state.maskU8) return;
  const rect = el.maskCanvas.getBoundingClientRect();
  const size = state.roiSize;

  const x = Math.floor((evt.clientX - rect.left) / rect.width * size);
  const y = Math.floor((evt.clientY - rect.top) / rect.height * size);
  if (x < 0 || y < 0 || x >= size || y >= size) return;

  const bs = state.brushSize | 0;
  const half = Math.floor(bs / 2);
  const val = (state.brushMode === "paint") ? 1 : 0;

  for (let yy = y - half; yy <= y + half; yy++) {
    if (yy < 0 || yy >= size) continue;
    for (let xx = x - half; xx <= x + half; xx++) {
      if (xx < 0 || xx >= size) continue;
      state.maskU8[yy*size + xx] = val;
    }
  }
  scheduleMaskRender();
}

function installBrushHandlers() {
  el.maskCanvas.addEventListener("mousedown", (evt) => {
    if (!state.maskU8) return;
    // Shift+drag is reserved for panning the mask viewer.
    if (evt.shiftKey) return;
    painting = true;
    paintAtEvent(evt);
  });
  window.addEventListener("mouseup", () => painting = false);
  el.maskCanvas.addEventListener("mouseleave", () => painting = false);
  el.maskCanvas.addEventListener("mousemove", (evt) => {
    if (!painting) return;
    if (evt.shiftKey) return;
    paintAtEvent(evt);
  });
}

// -------- API calls --------
async function uploadFits(file) {
  const fd = new FormData();
  fd.append("fits", file);
  fd.append("vis", state.fullVis);

  setStatus("Uploading…", "warning");

  const resp = await fetch("/api/upload", {
    method: "POST",
    body: fd,
  });
  const js = await resp.json();
  if (!js.ok) {
    setStatus("Upload error", "danger");
    alert(js.error || "Upload failed");
    return;
  }

  state.imageId = js.image_id;
  state.origW = js.orig_shape[0];
  state.origH = js.orig_shape[1];
  state.dispW = js.display_shape[0];
  state.dispH = js.display_shape[1];
  state.scaleX = js.scale_x;
  state.scaleY = js.scale_y;

  // Update stats UI
  el.fullShape.textContent = `${state.origW} × ${state.origH}`;
  el.fullMinMax.textContent = `${fmtNum(js.stats.min)} / ${fmtNum(js.stats.max)}`;
  if (el.fullMinMaxHeader) el.fullMinMaxHeader.textContent = `${fmtNum(js.stats.min)} / ${fmtNum(js.stats.max)}`;
  el.fullMeanStd.textContent = `${fmtNum(js.stats.mean)} / ${fmtNum(js.stats.std)}`;

  setStatus("Loaded", "success");

  // Build stage, set image, create ROI
  initKonvaStage();
  await setFullImagePng(js.full_png_b64);

  // Default ROI: centered
  state.roiSize = parseInt(el.roiSize.value, 10);
  updateRoiRectSize();

  const roiW = roiRect.width();
  const roiH = roiRect.height();
  roiRect.position({
    x: Math.max(0, (state.dispW - roiW) / 2),
    y: Math.max(0, (state.dispH - roiH) / 2),
  });
  layer.draw();
  updateRoiOrigFromRect();

  // Fetch ROI data + inference
  await requestRoiData();
}

async function setFullImagePng(pngB64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      konvaImg.image(img);
      layer.draw();
      resolve();
    };
    img.src = pngB64ToDataUrl(pngB64);
  });
}

async function requestRenderFull() {
  if (!state.imageId) return;
  setStatus("Rendering…", "warning");

  const resp = await fetch("/api/render_full", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ image_id: state.imageId, vis: state.fullVis }),
  });
  const js = await resp.json();
  if (!js.ok) {
    setStatus("Render error", "danger");
    alert(js.error || "Render failed");
    return;
  }

  // Update display geometry if needed
  state.dispW = js.display_shape[0];
  state.dispH = js.display_shape[1];
  state.scaleX = js.scale_x;
  state.scaleY = js.scale_y;

  initKonvaStage();
  await setFullImagePng(js.full_png_b64);

  // Re-apply ROI size and keep ROI in same ORIGINAL location if possible
  updateRoiRectSize();
  roiRect.position({
    x: state.roiX / state.scaleX,
    y: state.roiY / state.scaleY,
  });
  layer.draw();
  updateRoiOrigFromRect();

  await requestRoiData();
  setStatus("Loaded", "success");
}

async function requestRoiData() {
  if (!state.imageId) return;

  setStatus("Infer ROI…", "warning");

  const payload = {
    image_id: state.imageId,
    x: state.roiX,
    y: state.roiY,
    size: state.roiSize,
    roi_vis: state.roiVis,
  };

  const resp = await fetch("/api/roi_data", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const js = await resp.json();
  if (!js.ok) {
    setStatus("ROI error", "danger");
    alert(js.error || "ROI request failed");
    return;
  }

  // Server may clamp ROI to image bounds; update state/rect accordingly
  state.roiX = js.x;
  state.roiY = js.y;
  state.roiSize = js.size;
  el.roiSize.value = String(state.roiSize);

  // Move ROI rect to match clamped coords
  if (roiRect) {
    roiRect.width(roiDisplaySize());
    roiRect.height(roiDisplaySizeY());
    roiRect.position({ x: state.roiX / state.scaleX, y: state.roiY / state.scaleY });
    layer.draw();
  }

  // Update ROI image
  el.roiImage.src = pngB64ToDataUrl(js.roi_png_b64);

  // Raw ROI values (float32) for cursor intensity readout.
  // This is independent of visualization mode.
  if (js.roi_f32_b64) {
    try {
      state.roiF32 = b64ToFloat32Array(js.roi_f32_b64);
    } catch (e) {
      console.warn("Failed to decode roi_f32_b64", e);
      state.roiF32 = null;
    }
  } else {
    state.roiF32 = null;
  }

  // Update ROI stats
  const st = js.roi_stats || {};
  el.roiXY.textContent = `${state.roiX}, ${state.roiY} (size=${state.roiSize})`;
  el.roiMinMax.textContent = `${fmtNum(st.min)} / ${fmtNum(st.max)}`;
  el.roiMeanStd.textContent = `${fmtNum(st.mean)} / ${fmtNum(st.std)}`;
  el.roiMinMaxInline.textContent = `${fmtNum(st.min)} / ${fmtNum(st.max)}`;
  if (el.roiMinVal) el.roiMinVal.value = fmtNum(st.min);
  if (el.roiMaxVal) el.roiMaxVal.value = fmtNum(st.max);
  el.roiMeanStdInline.textContent = `${fmtNum(st.mean)} / ${fmtNum(st.std)}`;

  state.inferMs = js.infer_ms;
  el.inferMs.textContent = `${fmtNum(state.inferMs)} ms`;

  // Probability mask
  state.probU8 = b64ToUint8Array(js.mask_prob_u8_b64);

  // Read threshold/dilation from UI (keeps labels in sync)
  readThresholdFromUI({snap:true});
  readDilationFromUI();

  applyThresholdAndDilationFromProb();
  setStatus("Loaded", "success");
}

async function requestRoiRenderOnly() {
  if (!state.imageId) return;

  const payload = {
    image_id: state.imageId,
    x: state.roiX,
    y: state.roiY,
    size: state.roiSize,
    roi_vis: state.roiVis,
  };

  setStatus("Rendering ROI…", "warning");

  const resp = await fetch("/api/roi_render", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const js = await resp.json();
  if (!js.ok) {
    setStatus("ROI render error", "danger");
    alert(js.error || "ROI render failed");
    return;
  }

  // Update ROI image and stats (no mask changes)
  el.roiImage.src = pngB64ToDataUrl(js.roi_png_b64);

  const st = js.roi_stats || {};
  el.roiMinMax.textContent = `${fmtNum(st.min)} / ${fmtNum(st.max)}`;
  el.roiMeanStd.textContent = `${fmtNum(st.mean)} / ${fmtNum(st.std)}`;
  el.roiMinMaxInline.textContent = `${fmtNum(st.min)} / ${fmtNum(st.max)}`;
  el.roiMeanStdInline.textContent = `${fmtNum(st.mean)} / ${fmtNum(st.std)}`;

  setStatus("Loaded", "success");
}

// -------- UI events --------
el.fitsFile.addEventListener("change", async (evt) => {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".fits")) {
    alert("Please choose a .fits file.");
    return;
  }
  await uploadFits(file);
});

el.fullVis.addEventListener("change", async () => {
  state.fullVis = el.fullVis.value;
  if (!state.imageId) return;
  await requestRenderFull();
});

el.roiSize.addEventListener("change", async () => {
  state.roiSize = parseInt(el.roiSize.value, 10);
  updateRoiRectSize();
  updateRoiOrigFromRect();
  await requestRoiData();
});

el.roiVis.addEventListener("change", async () => {
  state.roiVis = el.roiVis.value;
  await requestRoiRenderOnly();
});

// Mask controls
el.maskThreshold.addEventListener("input", () => readThresholdFromUI({snap:false}));
el.maskThreshold.addEventListener("change", () => readThresholdFromUI({snap:true}));
readThresholdFromUI({snap:true});

el.maskDilation.addEventListener("change", readDilationFromUI);
readDilationFromUI();

el.btnApplyMask.addEventListener("click", () => {
  if (!state.probU8) return;
  applyThresholdAndDilationFromProb();
});

el.btnResetMask.addEventListener("click", () => {
  if (!state.probU8) return;
  el.maskThreshold.value = "0.50";
  el.maskDilation.value = "0";
  readThresholdFromUI({snap:true});
  readDilationFromUI();
  applyThresholdAndDilationFromProb();
});


el.btnDownloadMask.addEventListener("click", () => {
  if (!state.maskU8) return;
  // Use the canvas as-is (binary mask). Download with ROI coordinates in filename.
  const a = document.createElement("a");
  const pad = (n) => String(n).padStart(4, "0");
  const fname = `CRNet_mask_x${pad(state.roiX)}_y${pad(state.roiY)}_s${state.roiSize}.png`;
  a.download = fname;
  a.href = el.maskCanvas.toDataURL("image/png");
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

el.btnDownloadMaskFits.addEventListener("click", async () => {
  if (!state.maskU8 || !state.imageId) return;

  const pad = (n) => String(n).padStart(4, "0");
  const fname = `CRNet_mask_x${pad(state.roiX)}_y${pad(state.roiY)}_s${state.roiSize}.fits`;

  try {
    const payload = {
      image_id: state.imageId,
      x: state.roiX,
      y: state.roiY,
      size: state.roiSize,
      mask_u8_b64: uint8ArrayToB64(state.maskU8),
    };

    const resp = await fetch("/api/download_mask_fits", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const js = await resp.json().catch(() => null);
      alert((js && js.error) ? js.error : "Failed to generate FITS file.");
      return;
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
    alert("Failed to download FITS mask.");
  }
});

// Brush controls
el.brushMode.addEventListener("change", () => {
  state.brushMode = el.brushMode.value;
});
el.brushSize.addEventListener("change", () => {
  state.brushSize = parseInt(el.brushSize.value, 10);
});

installBrushHandlers();

// Mouse wheel zoom (ROI + mask)
installWheelZoom(el.roiViewer, el.roiImage, viewer.roi);
installPan(el.roiViewer, el.roiImage, viewer.roi, { requireShift: false });

installWheelZoom(el.maskViewer, el.maskCanvas, viewer.mask);
installPan(el.maskViewer, el.maskCanvas, viewer.mask, { requireShift: true });

// Cursor readouts (ROI + mask)
installCursorReadout(el.roiViewer, el.roiImage, el.roiCursorXY, el.roiXYOverlay, getRoiValueAt, fmtNum);
installCursorReadout(el.maskViewer, el.maskCanvas, el.maskCursorXY, el.maskXYOverlay, getMaskValueAt, (v) => String(v));

setStatus("No file", "secondary");
