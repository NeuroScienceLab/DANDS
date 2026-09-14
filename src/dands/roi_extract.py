# -*- coding: ascii -*-
from typing import Dict, Tuple, Sequence
import numpy as np
import nibabel as nib


def extract_roi_timeseries(fmri_img: nib.spatialimages.SpatialImage,
                           atlas_img: nib.spatialimages.SpatialImage,
                           roi_ids: Sequence[int]) -> Tuple[np.ndarray, Dict]:
    """
    Compute ROI mean time series for given labels.
    Returns (T x R array, info dict with voxel counts per ROI).
    """
    data = fmri_img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError("Expected 4D fMRI image")
    X, Y, Z, T = data.shape
    atlas = atlas_img.get_fdata().astype(np.int32)
    if atlas.shape != (X, Y, Z):
        raise ValueError("Atlas/fMRI shape mismatch after resampling")
    flat = data.reshape(-1, T)  # (V, T)
    labels = atlas.reshape(-1)
    R = len(roi_ids)
    ts = np.zeros((T, R), dtype=np.float32)
    counts = {}
    for idx, roi in enumerate(roi_ids):
        mask = labels == int(roi)
        n = int(mask.sum())
        counts[int(roi)] = n
        if n == 0:
            ts[:, idx] = np.nan
        else:
            roi_vox = flat[mask]
            ts[:, idx] = roi_vox.mean(axis=0)
    info = {'voxel_counts': counts, 'T': T, 'R': R}
    return ts, info
