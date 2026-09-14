# -*- coding: ascii -*-
"""Two-phase training for DANDS.

Phase 1: Train model parameters (local weights, time constants, readout) with
          W fixed at zero. Multiple epochs. Backprop through SNN dynamics.
Phase 2: Freeze model. Single inference pass with online W(t) update via
          ThreeFactorPlasticity. Saves W_dynamic.npy trajectory.
"""
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .model import DANDSModel
from .stdp import ThreeFactorPlasticity
from .utils import ensure_dir


def train_subject(timeseries: np.ndarray,
                  out_dir: str,
                  pretrained_path: str = None,
                  device: str = 'cuda',
                  n_e: int = 48,
                  n_i: int = 16,
                  dt: float = 1.0,
                  # Phase 1 params
                  epochs: int = 3,
                  lr: float = 5e-4,
                  accum_steps: int = 4,
                  lambda_local: float = 1e-5,
                  amp: bool = True,
                  # Phase 2 (inference) params
                  eta: float = 0.01,
                  rho: float = 0.0,
                  lam: float = 0.0):
    """Train on one subject and extract dynamic connectivity.

    Returns dict of output file paths.
    """
    ensure_dir(out_dir)
    T, R = timeseries.shape
    y = torch.from_numpy(timeseries).to(torch.float32).to(device)

    model = DANDSModel(n_roi=R, n_e=n_e, n_i=n_i, dt=dt).to(device)
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            sd = torch.load(pretrained_path, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights: {e}")
    model.enforce_biophysical_constraints_()

    # --- Phase 1: Train model parameters (W = 0) ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    amp_enabled = amp and str(device).startswith('cuda')
    scaler = GradScaler('cuda', enabled=amp_enabled)
    W_zero = torch.zeros(R, R, device=device)

    for epoch in range(epochs):
        state = model.init_state(batch=1, device=device)
        optimizer.zero_grad(set_to_none=True)
        for t in range(T - 1):
            y_t = y[t:t + 1, :]
            y_tp1 = y[t + 1:t + 2, :]
            with autocast('cuda', enabled=amp_enabled):
                y_hat, _, state = model.step(y_t, W_zero, state)
                mse = ((y_hat - y_tp1) ** 2).mean()
                # Local weight L2 regularization
                l2 = (model.W_EE.square().mean() + model.W_EI.square().mean() +
                      model.W_IE.square().mean() + model.W_II.square().mean())
                loss = mse + lambda_local * l2
            scaler.scale(loss / accum_steps).backward()
            if (t + 1) % accum_steps == 0 or t == T - 2:
                scaler.step(optimizer)
                scaler.update()
                model.enforce_biophysical_constraints_()
                optimizer.zero_grad(set_to_none=True)
            state = {k: v.detach() for k, v in state.items()}

    # --- Phase 2: Inference pass (frozen model, online W) ---
    model.eval()
    model.requires_grad_(False)
    plasticity = ThreeFactorPlasticity(
        n_roi=R, eta=eta, rho=rho, lam=lam,
        device=torch.device(device))

    # Allocate output arrays
    w_path = os.path.join(out_dir, 'W_dynamic.npy')
    W_mem = np.lib.format.open_memmap(
        w_path, dtype='float32', mode='w+', shape=(T - 1, R, R))

    state = model.init_state(batch=1, device=device)
    denoised = np.zeros((T, R), dtype=np.float32)
    denoised[0] = timeseries[0]

    # Chunked buffer for efficient GPU->CPU transfer
    chunk_size = 50
    buf_W = []
    buf_start = 0

    with torch.no_grad():
        for t in range(T - 1):
            y_t = y[t:t + 1, :]
            y_tp1 = y[t + 1:t + 2, :]
            y_hat, rE_pool, state = model.step(y_t, plasticity.W, state)
            denoised[t + 1] = y_hat.squeeze(0).cpu().numpy()
            # Plasticity update
            error = (y_tp1 - y_hat).squeeze(0)
            plasticity.step(y_t.squeeze(0), error)
            buf_W.append(plasticity.W.cpu().numpy().copy())
            if len(buf_W) >= chunk_size or t == T - 2:
                n = len(buf_W)
                W_mem[buf_start:buf_start + n] = np.array(buf_W)
                W_mem.flush()
                buf_start += n
                buf_W.clear()

    np.save(os.path.join(out_dir, 'denoised_fMRI.npy'), denoised)

    # Save model state for virtual intervention
    model_path = os.path.join(out_dir, 'model_state.pt')
    torch.save(model.state_dict(), model_path)

    # Fingerprint
    fp = {
        'roi_count': int(R),
        'n_e': n_e, 'n_i': n_i, 'dt': dt,
        'alpha_I': float(model.alpha_I.cpu()),
        'phase1_epochs': epochs,
        'phase2_eta': eta, 'phase2_rho': rho, 'phase2_lam': lam,
    }
    with open(os.path.join(out_dir, 'fingerprint.json'), 'w') as f:
        json.dump(fp, f, indent=2)

    return {
        'denoised_path': os.path.join(out_dir, 'denoised_fMRI.npy'),
        'W_dynamic_path': w_path,
        'model_state_path': model_path,
        'fingerprint_path': os.path.join(out_dir, 'fingerprint.json'),
    }
