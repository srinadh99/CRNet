"""
CRNet: Deep Learning Based Cosmic Ray Detection in Astronomical Images

Apply a learned deep learning model to a 2D numpy array to detect and remove
cosmic rays.

https://github.com/srinadh99/CRNet
"""

from .model import CRNet
from .training import train
from .evaluate import roc, roc_lacosmic
from .util import plotCRDetection, plotFirstConvFilters

__all__ = ["CRNet", "train", "roc", "roc_lacosmic", "plotCRDetection", "plotFirstConvFilters"]
__version__ = "0.1.0"
