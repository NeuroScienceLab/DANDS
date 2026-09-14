# -*- coding: ascii -*-
"""Online NLMS-based plasticity for dynamic inter-ROI connectivity W(t).

Since W contributes directly to the model output (y_hat includes y_in @ W.T),
the exact gradient of MSE w.r.t. W[i,j] is: -2 * error_i * y_in_j.
We use NLMS (Normalized LMS) which divides by input power for stable
convergence regardless of signal scale or dimensionality.

The eligibility trace provides temporal credit assignment for delayed effects
(HRF lag in fMRI), extending the effective memory window beyond one time step.

The system remains spike-based: the SNN processes inputs through ALIF neurons
with binary spikes and also receives inter-ROI current from W. The plasticity
rule operates on the observed signal because W acts on observed signals in the
model (y_in @ W.T contributes both to SNN input and directly to output).
"""
import torch


class ThreeFactorPlasticity:
    """Online NLMS plasticity for inter-ROI connectivity W(t).

    Update per time step:
        e_j(t) = lam * e_j(t-1) + y_in_j(t)             [eligibility trace]
        mu = eta / (||e||^2 + eps)                        [NLMS normalization]
        dW_ij(t) = mu * error_i(t) * e_j(t)              [NLMS gradient]
        W(t+1) = (1-rho) * W(t) + dW(t)                  [decay + update]

    The NLMS normalization ensures stable convergence regardless of input
    scale or dimensionality. The eligibility trace extends the effective
    memory window for delayed causal effects (HRF lag).
    """

    def __init__(self, n_roi: int, eta: float = 0.01, rho: float = 0.0,
                 lam: float = 0.0, w_min: float = -1.0, w_max: float = 1.0,
                 device=None):
        self.n = n_roi
        self.eta = eta
        self.rho = rho
        self.lam = lam
        self.w_min = w_min
        self.w_max = w_max
        self.device = device or torch.device('cpu')
        self.W = torch.zeros(n_roi, n_roi, device=self.device)
        self.e = torch.zeros(n_roi, device=self.device)

    def reset(self):
        self.W.zero_()
        self.e.zero_()

    @torch.no_grad()
    def step(self, y_in: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """One plasticity update (NLMS with optional eligibility trace).

        Args:
            y_in: (R,) observed signal per ROI at time t
            error: (R,) prediction error = y_true(t+1) - y_hat(t+1)

        Returns:
            W: (R, R) updated connectivity matrix
        """
        # Eligibility trace: exponential memory of pre-synaptic activity
        self.e = self.lam * self.e + y_in
        # NLMS: normalize by input power for scale-invariant convergence
        norm = self.e @ self.e + 1e-6
        mu = self.eta / norm
        # Gradient update
        dW = mu * torch.outer(error, self.e)
        # Weight decay + update
        self.W = (1.0 - self.rho) * self.W + dW
        self.W.fill_diagonal_(0.0)
        self.W.clamp_(self.w_min, self.w_max)
        return self.W
