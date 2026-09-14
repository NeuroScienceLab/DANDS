# -*- coding: ascii -*-
"""Vectorized DANDS model: ALIF E/I microcircuits with diagonal readout.

All R ROIs are computed in parallel via stacked tensors. No per-ROI Python
loops. Cross-ROI prediction flows exclusively through the external W(t) matrix,
enforced by a per-ROI diagonal readout (no nn.Linear(R,R)).

Key design: inter-ROI routing uses the OBSERVED signal y_in (not spike-derived
rE_pool). This makes W directly interpretable as effective connectivity and
ensures the plasticity rule receives high-fidelity pre-synaptic information.
A learnable skip connection handles the per-ROI autoregressive component.

State is a flat dict of 8 tensors, each (B, R, N).
"""
import torch
import torch.nn as nn
from .model_components import SurrogateSpike


class DANDSModel(nn.Module):
    """Vectorized spiking neural network for dynamic effective connectivity.

    Parameters are stacked across R ROIs as native tensors (not ModuleList).
    The step() method returns (y_hat, rE_pool, state_next).
    """

    def __init__(self, n_roi: int = 360, n_e: int = 48, n_i: int = 16,
                 dt: float = 1.0, slope: float = 10.0, sparsity: float = 0.1):
        super().__init__()
        self.n_roi = n_roi
        self.n_e = n_e
        self.n_i = n_i
        self.dt = dt
        self.slope = slope

        # Local recurrent weights: (R, post, pre)
        self.W_EE = nn.Parameter(self._init_block(n_roi, n_e, n_e, sparsity, +1))
        self.W_EI = nn.Parameter(self._init_block(n_roi, n_e, n_i, sparsity, -1))
        self.W_IE = nn.Parameter(self._init_block(n_roi, n_i, n_e, sparsity, +1))
        self.W_II = nn.Parameter(self._init_block(n_roi, n_i, n_i, sparsity, -1))

        # Input projection: (R, N)
        self.W_in_E = nn.Parameter(torch.randn(n_roi, n_e) * 0.1)
        self.W_in_I = nn.Parameter(torch.randn(n_roi, n_i) * 0.1)
        self.gain = nn.Parameter(torch.full((n_roi,), 1.5))
        self.bE = nn.Parameter(torch.full((n_roi,), 0.1))
        self.bI = nn.Parameter(torch.full((n_roi,), 0.1))

        # ALIF neuron parameters per ROI: (R,)
        self.log_tau_m_E = nn.Parameter(torch.full((n_roi,), 1.6094))  # log(5)
        self.log_tau_a_E = nn.Parameter(torch.full((n_roi,), 2.3026))  # log(10)
        self.V_th0_E = nn.Parameter(torch.full((n_roi,), 0.2))
        self.beta_E = nn.Parameter(torch.full((n_roi,), 0.3))
        self.log_tau_m_I = nn.Parameter(torch.full((n_roi,), 1.6094))
        self.log_tau_a_I = nn.Parameter(torch.full((n_roi,), 2.3026))
        self.V_th0_I = nn.Parameter(torch.full((n_roi,), 0.2))
        self.beta_I = nn.Parameter(torch.full((n_roi,), 0.3))

        # Inter-ROI routing: fraction of negative W current sent to I population
        self.alpha_I = nn.Parameter(torch.tensor(0.2))

        # Readout: skip connection + SNN nonlinear correction
        # y_hat = w_skip * y_in + w_out * pooled + b_out
        self.gamma = nn.Parameter(torch.full((n_roi,), 0.5))
        self.w_skip = nn.Parameter(torch.full((n_roi,), 0.5))
        self.w_out = nn.Parameter(torch.ones(n_roi))
        self.b_out = nn.Parameter(torch.zeros(n_roi))

    @staticmethod
    def _init_block(n_roi, n_post, n_pre, sparsity, sign):
        W = torch.zeros(n_roi, n_post, n_pre)
        mask = (torch.rand_like(W) < sparsity).to(W)
        mag = torch.abs(torch.randn_like(W)) * 0.02
        return mask * mag * (1.0 if sign > 0 else -1.0)

    @torch.no_grad()
    def enforce_biophysical_constraints_(self):
        self.W_EE.clamp_(min=0.0)
        self.W_IE.clamp_(min=0.0)
        self.W_EI.clamp_(max=0.0)
        self.W_II.clamp_(max=0.0)
        self.alpha_I.clamp_(0.0, 1.0)

    def init_state(self, batch: int, device):
        R, n_e, n_i = self.n_roi, self.n_e, self.n_i
        z = lambda d: torch.zeros(batch, R, d, device=device)
        return {'vE': z(n_e), 'aE': z(n_e), 'zE': z(n_e), 'rE': z(n_e),
                'vI': z(n_i), 'aI': z(n_i), 'zI': z(n_i), 'rI': z(n_i)}

    def step(self, y_in, W_t, state):
        """Single time step.

        Args:
            y_in: (B, R) observed signal at time t
            W_t: (R, R) inter-ROI connectivity matrix
            state: dict of 8 tensors each (B, R, N)

        Returns:
            y_hat: (B, R) prediction for t+1
            rE_pool: (B, R) mean excitatory firing rate
            state_next: dict of 8 tensors
        """
        R = self.n_roi
        # Inter-ROI current from OBSERVED signal (not spike-derived rate).
        # This makes W directly interpretable as effective connectivity.
        W_pos = torch.clamp(W_t, min=0.0)
        W_neg = torch.clamp(-W_t, min=0.0)
        I_rec_E = y_in @ W_pos.T - y_in @ W_neg.T  # (B, R)
        I_rec_I = self.alpha_I * (y_in @ W_neg.T)

        # Local recurrent currents via einsum: (B,R,N_pre) x (R,N_post,N_pre) -> (B,R,N_post)
        I_loc_E = (torch.einsum('brn,rmn->brm', state['zE'], self.W_EE) +
                   torch.einsum('brn,rmn->brm', state['zI'], self.W_EI))
        I_loc_I = (torch.einsum('brn,rmn->brm', state['zE'], self.W_IE) +
                   torch.einsum('brn,rmn->brm', state['zI'], self.W_II))

        # External input projection
        y_scaled = (y_in * self.gain).unsqueeze(-1)  # (B, R, 1)
        I_ext_E = y_scaled * self.W_in_E.unsqueeze(0)  # (B, R, n_e)
        I_ext_I = y_scaled * self.W_in_I.unsqueeze(0)  # (B, R, n_i)

        # Total synaptic current
        I_E = I_loc_E + I_ext_E + I_rec_E.unsqueeze(-1) + self.bE.view(1, R, 1)
        I_I = I_loc_I + I_ext_I + I_rec_I.unsqueeze(-1) + self.bI.view(1, R, 1)

        # ALIF dynamics - E population
        tau_m_E = torch.exp(self.log_tau_m_E).view(1, R, 1)
        tau_a_E = torch.exp(self.log_tau_a_E).view(1, R, 1)
        alpha_v_E = torch.clamp(1.0 - self.dt / (tau_m_E + 1e-6), 0.0, 1.0)
        alpha_a_E = torch.clamp(1.0 - self.dt / (tau_a_E + 1e-6), 0.0, 1.0)
        vE = alpha_v_E * state['vE'] + I_E
        thE = self.V_th0_E.view(1, R, 1) + self.beta_E.view(1, R, 1) * state['aE']
        zE = SurrogateSpike.apply(vE - thE, self.slope)
        rE = 0.9 * state['rE'] + 0.1 * zE
        vE = vE - zE * thE
        aE = alpha_a_E * state['aE'] + zE

        # ALIF dynamics - I population
        tau_m_I = torch.exp(self.log_tau_m_I).view(1, R, 1)
        tau_a_I = torch.exp(self.log_tau_a_I).view(1, R, 1)
        alpha_v_I = torch.clamp(1.0 - self.dt / (tau_m_I + 1e-6), 0.0, 1.0)
        alpha_a_I = torch.clamp(1.0 - self.dt / (tau_a_I + 1e-6), 0.0, 1.0)
        vI = alpha_v_I * state['vI'] + I_I
        thI = self.V_th0_I.view(1, R, 1) + self.beta_I.view(1, R, 1) * state['aI']
        zI = SurrogateSpike.apply(vI - thI, self.slope)
        rI = 0.9 * state['rI'] + 0.1 * zI
        vI = vI - zI * thI
        aI = alpha_a_I * state['aI'] + zI

        # Diagonal readout with skip connection + direct W pathway
        rE_pool = rE.mean(dim=-1)  # (B, R)
        rI_pool = rI.mean(dim=-1)
        pooled = rE_pool - self.gamma * rI_pool
        # Direct W contribution: y_in @ W.T gives the linear cross-ROI prediction.
        # This ensures W has an unattenuated effect on y_hat, making the
        # plasticity gradient exact (LMS). The SNN pathway adds nonlinear correction.
        y_hat = self.w_skip * y_in + self.w_out * pooled + self.b_out + (y_in @ W_t.T)

        state_next = {'vE': vE, 'aE': aE, 'zE': zE, 'rE': rE,
                      'vI': vI, 'aI': aI, 'zI': zI, 'rI': rI}
        return y_hat, rE_pool, state_next
