# -*- coding: ascii -*-
"""Ablation models for non-spiking comparison experiments.

Three models that replace the SNN in the W-First v2 pipeline:
1. MLPLocalModel: feedforward, no state, no spikes
2. GRULocalModel: stateful, no spikes
3. ALIFNoRecurrenceModel: spikes + adaptation, no E-I recurrence

All share the same interface as DANDSModel:
    step(y_in, W_t, state) -> (y_hat, pooled, state_next)
    init_state(batch, device) -> state_dict
"""
import torch
import torch.nn as nn
from .model_components import SurrogateSpike


class MLPLocalModel(nn.Module):
    """Stateless feedforward MLP. Tests: does any nonlinear f(y) help NLMS?

    Per-ROI: Linear(1,hidden) -> ReLU -> Linear(hidden,hidden) -> ReLU -> Linear(hidden,1)
    Vectorized across R ROIs.
    ~673 params/ROI with hidden=24.
    """

    def __init__(self, n_roi: int = 66, hidden: int = 24):
        super().__init__()
        self.n_roi = n_roi
        self.hidden = hidden
        self.W1 = nn.Parameter(torch.randn(n_roi, hidden, 1) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(n_roi, hidden))
        self.W2 = nn.Parameter(torch.randn(n_roi, hidden, hidden) * 0.05)
        self.b2 = nn.Parameter(torch.zeros(n_roi, hidden))
        self.W3 = nn.Parameter(torch.randn(n_roi, 1, hidden) * 0.05)
        self.b3 = nn.Parameter(torch.zeros(n_roi, 1))

    def init_state(self, batch: int, device):
        return {}

    def step(self, y_in, W_t, state):
        """y_in: (B, R), W_t: ignored, state: ignored."""
        B, R = y_in.shape
        x = y_in.unsqueeze(-1)  # (B, R, 1)
        h = torch.einsum('brn,rmn->brm', x, self.W1) + self.b1.unsqueeze(0)
        h = torch.relu(h)
        h = torch.einsum('brn,rmn->brm', h, self.W2) + self.b2.unsqueeze(0)
        h = torch.relu(h)
        out = torch.einsum('brn,rmn->brm', h, self.W3) + self.b3.unsqueeze(0)
        y_hat = out.squeeze(-1)  # (B, R)
        return y_hat, torch.zeros_like(y_hat), state


class GRULocalModel(nn.Module):
    """Stateful GRU (no spikes). Tests: is temporal memory the key ingredient?

    Per-ROI GRU cell with hidden_size=13, vectorized across R ROIs.
    ~638 params/ROI.
    """

    def __init__(self, n_roi: int = 66, hidden: int = 13):
        super().__init__()
        self.n_roi = n_roi
        self.hidden = hidden
        # GRU gates: z(update), r(reset), n(new)
        # Each gate: W_ih(R, hidden, 1) + W_hh(R, hidden, hidden) + b(R, hidden)
        s = 0.1
        self.Wz_i = nn.Parameter(torch.randn(n_roi, hidden, 1) * s)
        self.Wz_h = nn.Parameter(torch.randn(n_roi, hidden, hidden) * 0.05)
        self.bz = nn.Parameter(torch.zeros(n_roi, hidden))
        self.Wr_i = nn.Parameter(torch.randn(n_roi, hidden, 1) * s)
        self.Wr_h = nn.Parameter(torch.randn(n_roi, hidden, hidden) * 0.05)
        self.br = nn.Parameter(torch.zeros(n_roi, hidden))
        self.Wn_i = nn.Parameter(torch.randn(n_roi, hidden, 1) * s)
        self.Wn_h = nn.Parameter(torch.randn(n_roi, hidden, hidden) * 0.05)
        self.bn = nn.Parameter(torch.zeros(n_roi, hidden))
        # Readout
        self.W_out = nn.Parameter(torch.randn(n_roi, 1, hidden) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(n_roi, 1))

    def init_state(self, batch: int, device):
        return {'h': torch.zeros(batch, self.n_roi, self.hidden, device=device)}

    def step(self, y_in, W_t, state):
        """y_in: (B, R), W_t: ignored."""
        B, R = y_in.shape
        h_prev = state['h']  # (B, R, hidden)
        x = y_in.unsqueeze(-1)  # (B, R, 1)
        # Update gate
        z = torch.sigmoid(
            torch.einsum('brn,rmn->brm', x, self.Wz_i) +
            torch.einsum('brn,rmn->brm', h_prev, self.Wz_h) + self.bz)
        # Reset gate
        r = torch.sigmoid(
            torch.einsum('brn,rmn->brm', x, self.Wr_i) +
            torch.einsum('brn,rmn->brm', h_prev, self.Wr_h) + self.br)
        # New candidate
        n = torch.tanh(
            torch.einsum('brn,rmn->brm', x, self.Wn_i) +
            torch.einsum('brn,rmn->brm', r * h_prev, self.Wn_h) + self.bn)
        # Hidden state update
        h_new = (1 - z) * n + z * h_prev
        # Readout
        out = torch.einsum('brn,rmn->brm', h_new, self.W_out) + self.b_out
        y_hat = out.squeeze(-1)  # (B, R)
        return y_hat, torch.zeros_like(y_hat), {'h': h_new}


class ALIFNoRecurrenceModel(nn.Module):
    """ALIF neurons without recurrent E-I connectivity.
    Tests: are spikes alone sufficient without E-I microcircuit structure?

    Per-ROI: 24 ALIF neurons, feedforward only (no W_EE/EI/IE/II).
    ~629 params/ROI.
    """

    def __init__(self, n_roi: int = 66, n_neurons: int = 24,
                 dt: float = 1.0, slope: float = 10.0):
        super().__init__()
        self.n_roi = n_roi
        self.n_neurons = n_neurons
        self.dt = dt
        self.slope = slope
        # Input projection: (R, n_neurons)
        self.W_in = nn.Parameter(torch.randn(n_roi, n_neurons) * 0.1)
        self.gain = nn.Parameter(torch.full((n_roi,), 1.5))
        self.bias = nn.Parameter(torch.full((n_roi,), 0.1))
        # ALIF parameters per ROI: (R,)
        self.log_tau_m = nn.Parameter(torch.full((n_roi,), 1.6094))  # log(5)
        self.log_tau_a = nn.Parameter(torch.full((n_roi,), 2.3026))  # log(10)
        self.V_th0 = nn.Parameter(torch.full((n_roi,), 0.2))
        self.beta = nn.Parameter(torch.full((n_roi,), 0.3))
        # Readout
        self.w_out = nn.Parameter(torch.ones(n_roi))
        self.b_out = nn.Parameter(torch.zeros(n_roi))

    def init_state(self, batch: int, device):
        R, N = self.n_roi, self.n_neurons
        return {
            'v': torch.zeros(batch, R, N, device=device),
            'a': torch.zeros(batch, R, N, device=device),
            'z': torch.zeros(batch, R, N, device=device),
            'r': torch.zeros(batch, R, N, device=device),
        }

    def step(self, y_in, W_t, state):
        """y_in: (B, R), W_t: ignored."""
        R = self.n_roi
        # Feedforward input only (no recurrence)
        y_scaled = (y_in * self.gain).unsqueeze(-1)  # (B, R, 1)
        I_syn = y_scaled * self.W_in.unsqueeze(0) + self.bias.view(1, R, 1)
        # ALIF dynamics
        tau_m = torch.exp(self.log_tau_m).view(1, R, 1)
        tau_a = torch.exp(self.log_tau_a).view(1, R, 1)
        alpha_v = torch.clamp(1.0 - self.dt / (tau_m + 1e-6), 0.0, 1.0)
        alpha_a = torch.clamp(1.0 - self.dt / (tau_a + 1e-6), 0.0, 1.0)
        v = alpha_v * state['v'] + I_syn
        theta = self.V_th0.view(1, R, 1) + self.beta.view(1, R, 1) * state['a']
        z = SurrogateSpike.apply(v - theta, self.slope)
        r = 0.9 * state['r'] + 0.1 * z
        v = v - z * theta
        a = alpha_a * state['a'] + z
        # Readout: mean firing rate
        r_pool = r.mean(dim=-1)  # (B, R)
        y_hat = self.w_out * r_pool + self.b_out
        state_next = {'v': v, 'a': a, 'z': z, 'r': r}
        return y_hat, r_pool, state_next
