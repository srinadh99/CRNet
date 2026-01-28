import numpy as np
import pytest

from CRNet import roc as roc_module


class DummyDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class DummyModel:
    def __init__(self, pdt_mask):
        self.pdt_mask = pdt_mask

    def clean(self, image, inpaint=False, binary=False):
        # returns "probability" mask for the current image
        return self.pdt_mask


def test_roc_returns_arrays(monkeypatch):
    # ---- make tiny deterministic data ----
    # One sample, 4x4
    image = np.zeros((1, 4, 4), dtype=float)
    mask = np.zeros((1, 4, 4), dtype=int)
    ignore = np.zeros((1, 4, 4), dtype=int)

    # ground truth has two CR pixels
    mask[0, 1, 1] = 1
    mask[0, 2, 2] = 1

    # model predicts probabilities
    pdt = np.zeros((4, 4), dtype=float)
    pdt[1, 1] = 0.9
    pdt[2, 2] = 0.8
    pdt[0, 0] = 0.7  # false positive at high threshold maybe

    model = DummyModel(pdt)

    # dataset() inside roc() should return an object with __len__/__getitem__
    items = [(image[0], mask[0], ignore[0])]
    monkeypatch.setattr(roc_module, "dataset", lambda *args, **kwargs: DummyDataset(items))

    thresholds = np.array([0.5, 0.85])

    tpr, fpr = roc_module.roc(
        model=model,
        image=image,
        mask=mask,
        ignore=ignore,
        thresholds=thresholds,
        dilate=False,
    )

    assert tpr.shape == thresholds.shape
    assert fpr.shape == thresholds.shape
    assert np.all((tpr >= 0) & (tpr <= 100))
    assert np.all((fpr >= 0) & (fpr <= 100))


def test_roc_monotonic_behavior_basic(monkeypatch):
    # If threshold increases, predicted positives should not increase:
    # so TPR and FPR should be non-increasing (in general).
    image = np.zeros((1, 4, 4), dtype=float)
    mask = np.zeros((1, 4, 4), dtype=int)
    ignore = np.zeros((1, 4, 4), dtype=int)

    mask[0, 1, 1] = 1
    mask[0, 2, 2] = 1

    pdt = np.zeros((4, 4), dtype=float)
    pdt[1, 1] = 0.9
    pdt[2, 2] = 0.6
    pdt[0, 0] = 0.7  # FP

    model = DummyModel(pdt)
    items = [(image[0], mask[0], ignore[0])]
    monkeypatch.setattr(roc_module, "dataset", lambda *args, **kwargs: DummyDataset(items))

    thresholds = np.array([0.1, 0.5, 0.8])
    tpr, fpr = roc_module.roc(model, image, mask=mask, ignore=ignore, thresholds=thresholds, dilate=False)

    # minimal monotonic checks: later points should be <= earlier points
    assert np.all(np.diff(tpr) <= 1e-9)
    assert np.all(np.diff(fpr) <= 1e-9)


def test_roc_dilate_true_returns_two_curves(monkeypatch):
    image = np.zeros((1, 4, 4), dtype=float)
    mask = np.zeros((1, 4, 4), dtype=int)
    ignore = np.zeros((1, 4, 4), dtype=int)

    mask[0, 1, 1] = 1

    pdt = np.zeros((4, 4), dtype=float)
    pdt[1, 1] = 0.9
    model = DummyModel(pdt)

    items = [(image[0], mask[0], ignore[0])]
    monkeypatch.setattr(roc_module, "dataset", lambda *args, **kwargs: DummyDataset(items))

    # make dilation a no-op (keeps test simple and avoids depending on skimage)
    monkeypatch.setattr(roc_module, "dilation", lambda arr, kernel: arr)

    thresholds = np.array([0.5, 0.95])

    (tpr, fpr), (tpr_d, fpr_d) = roc_module.roc(
        model=model,
        image=image,
        mask=mask,
        ignore=ignore,
        thresholds=thresholds,
        dilate=True,
        rad=1,
    )

    assert tpr.shape == thresholds.shape
    assert fpr.shape == thresholds.shape
    assert tpr_d.shape == thresholds.shape
    assert fpr_d.shape == thresholds.shape
