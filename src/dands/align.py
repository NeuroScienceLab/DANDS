# -*- coding: ascii -*-
from typing import Dict, Tuple
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from .utils import affine_close


def check_and_resample_atlas_to_fmri(fmri_img: nib.spatialimages.SpatialImage,
                                     atlas_img: nib.spatialimages.SpatialImage) -> Tuple[nib.spatialimages.SpatialImage, Dict]:
    """
    Ensure atlas is aligned to fMRI space. If not, resample atlas (nearest) to fmri_img.
    Returns (atlas_in_fmri_space, info_dict).
    """
    info = {
        'fmri_shape': tuple(fmri_img.shape),
        'atlas_shape': tuple(atlas_img.shape),
        'affine_equal': affine_close(fmri_img.affine, atlas_img.affine),
        'resampled': False,
    }

    if atlas_img.shape[:3] != fmri_img.shape[:3] or not info['affine_equal']:
        # Resample atlas to fmri grid
        target = (fmri_img.shape[:3], fmri_img.affine)
        atlas_res = resample_from_to(atlas_img, target, order=0)  # nearest neighbor for labels
        info['resampled'] = True
        info['atlas_shape_resampled'] = tuple(atlas_res.shape)
        return atlas_res, info
    else:
        return atlas_img, info
