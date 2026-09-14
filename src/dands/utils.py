# -*- coding: ascii -*-
import os
import json
import numpy as np
import nibabel as nib


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_tr_and_T(img: nib.spatialimages.SpatialImage):
    """Return (TR, T) from a 4D NIfTI image."""
    zooms = img.header.get_zooms()
    if len(zooms) < 4:
        raise ValueError("NIfTI image missing time zoom (TR)")
    TR = float(zooms[3])
    shape = img.shape
    if len(shape) != 4:
        raise ValueError("Expected 4D fMRI image")
    T = int(shape[3])
    return TR, T


def affine_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if two affines are approximately equal."""
    if a.shape != (4, 4) or b.shape != (4, 4):
        return False
    return np.allclose(a, b, atol=tol, rtol=0)


def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    m = np.nanmean(x, axis=axis, keepdims=True)
    s = np.nanstd(x, axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def save_json(obj, path: str):
    with open(path, 'w', encoding='ascii') as f:
        json.dump(obj, f, indent=2)
