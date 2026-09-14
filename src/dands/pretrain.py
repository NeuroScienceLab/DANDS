# -*- coding: ascii -*-
"""Pretraining: Phase 1 only (train model parameters with W=0) across subjects.

After pretraining, per-subject fine-tuning + Phase 2 inference is done by train.py.
"""
import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .utils import ensure_dir
from .align import check_and_resample_atlas_to_fmri
from .roi_extract import extract_roi_timeseries
from .model import DANDSModel


def ensure_roi_timeseries(subject_nii: str, atlas_nii: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    ts_path = os.path.join(out_dir, 'roi_timeseries.npy')
    if os.path.exists(ts_path):
        return ts_path
    fmri = nib.load(subject_nii)
    atlas = nib.load(atlas_nii)
    atlas_in_fmri, _ = check_and_resample_atlas_to_fmri(fmri, atlas)
    ts, _ = extract_roi_timeseries(fmri, atlas_in_fmri, list(range(1, 361)))
    ts = (ts - ts.mean(0, keepdims=True)) / (ts.std(0, keepdims=True) + 1e-6)
    np.save(ts_path, ts.astype(np.float32))
    return ts_path


def pretrain(data_root: str,
             atlas_path: str,
             result_root: str,
             model_out: str = 'model/pretrain.pt',
             device: str = 'cuda',
             n_e: int = 48, n_i: int = 16,
             dt: float = 1.0,
             epochs: int = 1,
             lr: float = 5e-4,
             lambda_local: float = 1e-5,
             amp: bool = True,
             accum_steps: int = 4):
    """Pretrain model parameters (Phase 1) across all subjects in data_root."""
    ensure_dir(os.path.dirname(model_out))
    subj_dirs = sorted(glob.glob(os.path.join(data_root, '*')))
    R = 360
    model = DANDSModel(n_roi=R, n_e=n_e, n_i=n_i, dt=dt).to(device)
    model.enforce_biophysical_constraints_()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    amp_enabled = amp and str(device).startswith('cuda')
    scaler = GradScaler('cuda', enabled=amp_enabled)
    W_zero = torch.zeros(R, R, device=device)

    for epoch in range(epochs):
        for subj_dir in subj_dirs:
            nii_list = glob.glob(os.path.join(subj_dir, '*.nii*'))
            if not nii_list:
                continue
            subj_nii = nii_list[0]
            ts_path = ensure_roi_timeseries(
                subj_nii, atlas_path,
                os.path.join(result_root, os.path.basename(subj_dir)))
            ts = np.load(ts_path)
            y = torch.from_numpy(ts).to(torch.float32).to(device)
            T = y.shape[0]
            state = model.init_state(batch=1, device=device)
            optimizer.zero_grad(set_to_none=True)
            for t in range(T - 1):
                y_t = y[t:t + 1, :]
                y_tp1 = y[t + 1:t + 2, :]
                with autocast('cuda', enabled=amp_enabled):
                    y_hat, _, state = model.step(y_t, W_zero, state)
                    mse = ((y_hat - y_tp1) ** 2).mean()
                    l2 = (model.W_EE.square().mean() +
                          model.W_EI.square().mean() +
                          model.W_IE.square().mean() +
                          model.W_II.square().mean())
                    loss = mse + lambda_local * l2
                scaler.scale(loss / accum_steps).backward()
                if (t + 1) % accum_steps == 0 or t == T - 2:
                    scaler.step(optimizer)
                    scaler.update()
                    model.enforce_biophysical_constraints_()
                    optimizer.zero_grad(set_to_none=True)
                state = {k: v.detach() for k, v in state.items()}

    torch.save(model.state_dict(), model_out)
    return model_out


def pretrain_from_timeseries(ts_list: list,
                             model_out: str,
                             init_model: str = None,
                             device: str = 'cuda',
                             n_e: int = 48, n_i: int = 16,
                             dt: float = 1.0,
                             epochs: int = 3,
                             lr: float = 5e-4,
                             lambda_local: float = 1e-5,
                             amp: bool = True,
                             accum_steps: int = 4):
    """Pretrain from a list of numpy timeseries arrays (T, R)."""
    ensure_dir(os.path.dirname(model_out))
    R = ts_list[0].shape[1]
    model = DANDSModel(n_roi=R, n_e=n_e, n_i=n_i, dt=dt).to(device)
    if init_model and os.path.exists(init_model):
        try:
            sd = torch.load(init_model, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f'Warning: failed to load init model: {e}')
    model.enforce_biophysical_constraints_()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    amp_enabled = amp and str(device).startswith('cuda')
    scaler = GradScaler('cuda', enabled=amp_enabled)
    W_zero = torch.zeros(R, R, device=device)

    for epoch in range(epochs):
        for ts in ts_list:
            y = torch.from_numpy(ts).to(torch.float32).to(device)
            T = y.shape[0]
            state = model.init_state(batch=1, device=device)
            optimizer.zero_grad(set_to_none=True)
            for t in range(T - 1):
                y_t = y[t:t + 1, :]
                y_tp1 = y[t + 1:t + 2, :]
                with autocast('cuda', enabled=amp_enabled):
                    y_hat, _, state = model.step(y_t, W_zero, state)
                    mse = ((y_hat - y_tp1) ** 2).mean()
                    l2 = (model.W_EE.square().mean() +
                          model.W_EI.square().mean() +
                          model.W_IE.square().mean() +
                          model.W_II.square().mean())
                    loss = mse + lambda_local * l2
                scaler.scale(loss / accum_steps).backward()
                if (t + 1) % accum_steps == 0 or t == T - 2:
                    scaler.step(optimizer)
                    scaler.update()
                    model.enforce_biophysical_constraints_()
                    optimizer.zero_grad(set_to_none=True)
                state = {k: v.detach() for k, v in state.items()}

    torch.save(model.state_dict(), model_out)
    return model_out
