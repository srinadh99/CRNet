import pytest

import numpy as np

from CRNet import util


def test_medmask():

    in_im = np.random.random((256, 256))
    mask = np.random.randint(0,2, size=(256, 256))

    masked = util.fillMask(in_im, mask)

    ok_inds = np.where(mask == 0)

    assert np.all(in_im[ok_inds] == masked[ok_inds])
    
def test_maskMetric_simple():
    GT = np.array([[1, 0],
                   [1, 0]])
    PD = np.array([[1, 1],
                   [0, 0]])

    TP, TN, FP, FN = util.maskMetric(PD, GT)

    assert TP == 1
    assert TN == 1
    assert FP == 1
    assert FN == 1    