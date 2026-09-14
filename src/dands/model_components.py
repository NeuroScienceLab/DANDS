# -*- coding: ascii -*-
import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope: float):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        k = ctx.slope
        sig = torch.sigmoid(k * x)
        grad = k * sig * (1.0 - sig)
        return grad_output * grad, None


class ALIFGroup(nn.Module):
    def __init__(self, n_neurons: int, dt: float = 1.0, slope: float = 10.0):
        super().__init__()
        self.n = n_neurons
        self.log_tau_m = nn.Parameter(torch.log(torch.tensor(5.0)))
        self.log_tau_a = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.V_th0 = nn.Parameter(torch.tensor(0.2))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.dt = dt
        self.slope = slope

    def init_state(self, batch: int, device) -> dict:
        return {
            'v': torch.zeros(batch, self.n, device=device),
            'a': torch.zeros(batch, self.n, device=device),
            'z': torch.zeros(batch, self.n, device=device),
            'r': torch.zeros(batch, self.n, device=device),
        }

    def forward(self, I_syn: torch.Tensor, state: dict):
        tau_m = torch.exp(self.log_tau_m)
        tau_a = torch.exp(self.log_tau_a)
        alpha_v = torch.clamp(1.0 - self.dt / (tau_m + 1e-6), 0.0, 1.0)
        alpha_a = torch.clamp(1.0 - self.dt / (tau_a + 1e-6), 0.0, 1.0)
        v = alpha_v * state['v'] + I_syn
        theta = self.V_th0 + self.beta * state['a']
        x = v - theta
        z = SurrogateSpike.apply(x, self.slope)
        r = 0.9 * state['r'] + 0.1 * z
        v = v - z * theta
        a = alpha_a * state['a'] + z
        state = {'v': v, 'a': a, 'z': z, 'r': r}
        return z, state


class EIMicroCircuit(nn.Module):
    """DEPRECATED: kept for loading old checkpoints only. Use DANDSModel directly."""
    def __init__(self, n_e: int, n_i: int, sparsity: float = 0.1):
        super().__init__()
        self.n_e = n_e
        self.n_i = n_i
        # Local weights
        self.W_EE = nn.Parameter(self._init_block(n_e, n_e, sparsity, sign=+1))
        self.W_EI = nn.Parameter(self._init_block(n_e, n_i, sparsity, sign=-1))
        self.W_IE = nn.Parameter(self._init_block(n_i, n_e, sparsity, sign=+1))
        self.W_II = nn.Parameter(self._init_block(n_i, n_i, sparsity, sign=-1))
        # External input projection and biases
        self.W_in_E = nn.Parameter(torch.randn(n_e) * 0.1)
        self.W_in_I = nn.Parameter(torch.randn(n_i) * 0.1)
        self.input_gain = nn.Parameter(torch.tensor(1.5))
        self.I_bias_E = nn.Parameter(torch.tensor(0.1))
        self.I_bias_I = nn.Parameter(torch.tensor(0.1))
        # Neuron groups
        self.E = ALIFGroup(n_e)
        self.I = ALIFGroup(n_i)

    @staticmethod
    def _init_block(n_post, n_pre, sparsity: float, sign: int):
        W = torch.zeros(n_post, n_pre)
        mask = (torch.rand_like(W) < sparsity).to(W)
        mag = torch.abs(torch.randn_like(W)) * 0.02
        W = mask * mag * (1.0 if sign > 0 else -1.0)
        return W

    @torch.no_grad()
    def enforce_sign_constraints_(self):
        """Project local E/I weights back to their physiological sign cones."""
        self.W_EE.clamp_(min=0.0)
        self.W_IE.clamp_(min=0.0)
        self.W_EI.clamp_(max=0.0)
        self.W_II.clamp_(max=0.0)

    def init_state(self, batch: int, device):
        return {
            'E': self.E.init_state(batch, device),
            'I': self.I.init_state(batch, device),
        }

    def forward(self, y_in: torch.Tensor,
                I_rec_E: torch.Tensor,
                I_rec_I: torch.Tensor,
                state: dict):
        z_E_prev = state['E']['z']
        z_I_prev = state['I']['z']
        I_loc_E = torch.matmul(z_E_prev, self.W_EE.T) + torch.matmul(z_I_prev, self.W_EI.T)
        I_loc_I = torch.matmul(z_E_prev, self.W_IE.T) + torch.matmul(z_I_prev, self.W_II.T)
        I_ext_E = y_in.unsqueeze(-1) * self.W_in_E * self.input_gain
        I_ext_I = y_in.unsqueeze(-1) * self.W_in_I * self.input_gain
        Irec_E = I_rec_E.unsqueeze(-1)
        Irec_I = I_rec_I.unsqueeze(-1)
        I_E = I_loc_E + I_ext_E + Irec_E + self.I_bias_E
        I_I = I_loc_I + I_ext_I + Irec_I + self.I_bias_I
        z_E, state_E = self.E(I_E, state['E'])
        z_I, state_I = self.I(I_I, state['I'])
        r_E = state_E['r']
        r_I = state_I['r']
        state = {'E': state_E, 'I': state_I}
        return z_E, z_I, r_E, r_I, state

