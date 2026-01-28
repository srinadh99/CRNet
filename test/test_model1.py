import os
import time
import numpy as np
import pytest

from CRNet.model import CRNet

MODEL_PATH = "/home/srinadb/CRNet/MAE_deepCR/UNet2/CRNet/HST/learned_models/ACS-WFC-BN.pth"
MODEL_KWARGS = {"hidden": 32, "act_norm": "batch", "att": False}


def _writable_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure a NumPy array is contiguous + writable.
    This prevents PyTorch from warning about non-writable tensors when using from_numpy().
    """
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _make_model() -> CRNet:
    if not os.path.isfile(MODEL_PATH):
        pytest.skip(f"Model checkpoint not found: {MODEL_PATH}")
    return CRNet(mask=MODEL_PATH, device="CPU", model_kwargs=MODEL_KWARGS)


def test_CRNet_serial():
    mdl = _make_model()
    in_im = _writable_array(np.ones((299, 299), dtype=np.float32))

    out = mdl.clean(in_im, inpaint=True)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    out = mdl.clean(in_im, inpaint=False)
    assert out.shape == in_im.shape


def test_CRNet_parallel():
    mdl = _make_model()
    in_im = _writable_array(np.ones((299, 299), dtype=np.float32))

    out = mdl.clean(in_im, n_jobs=-1, inpaint=True)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    # Correctness test for parallel on larger image (NO speed assertion; too flaky on HPC/CI)
    if (os.cpu_count() or 1) > 2:
        big_im = _writable_array(np.ones((1024, 1024), dtype=np.float32))
        out = mdl.clean(big_im, inpaint=False, n_jobs=4, patch=256)
        assert out.shape == big_im.shape


def test_seg():
    mdl = _make_model()
    in_im = _writable_array(np.ones((300, 500), dtype=np.float32))

    out = mdl.clean(in_im, segment=True, inpaint=True)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    out = mdl.clean(in_im, inpaint=False, segment=True)
    assert out.shape == in_im.shape


if __name__ == "__main__":
    # Run as a script (pytest will discover tests automatically when running `pytest`)
    test_seg()
    test_CRNet_serial()
    test_CRNet_parallel()
