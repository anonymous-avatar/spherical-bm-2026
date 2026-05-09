# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
# ]
# ///
"""Normalizing flow on S_N following Rezende et al. 2020.

Architecture: L layers, each composed of
  1. Householder reflection  (rotates the "active axis")
  2. 1-D altitude transform  h: [-1,1] -> [-1,1]
     h(t) = tanh(a · arctanh(t) + b),  a = 1 + alpha
     (identity when alpha=beta=0)

Base distribution: uniform on S_N = {x in R^N : ||x||^2 = N}.

The density on S_N is
    log Q(x) = -log vol(S_N) - sum_l log D_upd_l

where D_upd_l = h'(t_l) * ((1-h(t_l)^2)/(1-t_l^2))^{(N-3)/2}
is the measure density update from the l-th altitude transform.

Key property: at alpha=beta=0 for all layers, the flow is the identity
and Q = uniform on S_N.  L2 regularization on (alpha, beta) pushes
toward the uniform distribution — the correct analog of the SBM prior.

Reference:
    Rezende et al., "Normalizing Flows on Tori and Spheres",
    arXiv:2002.02428 (2020).
"""

import math

import torch
import torch.nn as nn


class SphericalFlow(nn.Module):
    """Normalizing flow on S_N via Householder reflections + altitude transforms."""

    def __init__(self, N: int, n_layers: int = 8):
        super().__init__()
        self.N = N
        self.n_layers = n_layers

        # Altitude parameters (identity at zero)
        self.alpha = nn.Parameter(torch.zeros(n_layers))
        self.beta = nn.Parameter(torch.zeros(n_layers))

        # Householder vectors (random init, normalized at use time)
        v = torch.randn(n_layers, N)
        self.v = nn.Parameter(v)

        # Precompute log volume of S_N
        self._log_vol = self._compute_log_vol(N)

    # ── static helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_log_vol(N: int) -> float:
        """log vol(S_N) where S_N = {x: ||x||^2 = N} (= S^{N-1} scaled)."""
        # vol = N^{(N-1)/2} * 2 pi^{N/2} / Gamma(N/2)
        return (
            (N - 1) / 2 * math.log(N)
            + math.log(2)
            + N / 2 * math.log(math.pi)
            - math.lgamma(N / 2)
        )

    # ── 1-D altitude transform ──────────────────────────────────────

    @staticmethod
    def _altitude_forward(
        t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, N: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h(t) = tanh(a * arctanh(t) + b),  a = 1 + alpha.

        Returns (h, log_density_update) where
            log D_upd = log|a| + (N-1)/2 * [log(1-h^2) - log(1-t^2)].
        """
        EPS = 1e-7
        t = t.clamp(-1 + EPS, 1 - EPS)
        a = 1.0 + alpha

        arctanh_t = 0.5 * (torch.log1p(t) - torch.log1p(-t))
        z = a * arctanh_t + beta
        h = torch.tanh(z)
        h = h.clamp(-1 + EPS, 1 - EPS)

        # log D_upd = log|a| + (N-1)/2 * [log(1-h^2) - log(1-t^2)]
        diff = torch.log1p(-h * h) - torch.log1p(-t * t)
        log_dup = torch.log(a.abs()) + (N - 1) / 2.0 * diff
        return h, log_dup

    @staticmethod
    def _altitude_inverse(
        h: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Inverse: t = tanh((arctanh(h) - b) / a)."""
        EPS = 1e-7
        h = h.clamp(-1 + EPS, 1 - EPS)
        a = 1.0 + alpha
        arctanh_h = 0.5 * (torch.log1p(h) - torch.log1p(-h))
        t = torch.tanh((arctanh_h - beta) / a)
        return t

    # ── full flow ────────────────────────────────────────────────────

    def _reflect(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Householder reflection x -> x - 2 (x.v) v / ||v||^2."""
        v_norm = v / v.norm().clamp(min=1e-8)
        return x - 2.0 * (x @ v_norm).unsqueeze(-1) * v_norm

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Push x (on S_N) through the flow.

        Returns (x_out, sum_log_dup) where
            log Q(x_out) = -log_vol - sum_log_dup.
        """
        sqrt_N = self.N**0.5
        log_dup_total = torch.zeros(
            x.shape[:-1], dtype=x.dtype, device=x.device
        )

        for l in range(self.n_layers):
            x = self._reflect(x, self.v[l])

            t = x[..., -1] / sqrt_N  # altitude in [-1, 1]
            h, log_dup = self._altitude_forward(
                t, self.alpha[l], self.beta[l], self.N
            )
            log_dup_total = log_dup_total + log_dup

            # Reassemble  (out-of-place for autograd)
            scale = ((1.0 - h * h) / (1.0 - t * t).clamp(min=1e-12)).clamp(
                min=0
            ).sqrt()
            x_eq = x[..., :-1] * scale.unsqueeze(-1)  # equatorial
            x_alt = sqrt_N * h  # altitude
            x = torch.cat([x_eq, x_alt.unsqueeze(-1)], dim=-1)

        return x, log_dup_total

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse flow (for sampling): x_target -> x_base."""
        sqrt_N = self.N**0.5

        for l in range(self.n_layers - 1, -1, -1):
            h = x[..., -1] / sqrt_N
            t = self._altitude_inverse(h, self.alpha[l], self.beta[l])

            scale = ((1.0 - t * t) / (1.0 - h * h).clamp(min=1e-12)).clamp(
                min=0
            ).sqrt()
            x_eq = x[..., :-1] * scale.unsqueeze(-1)
            x_alt = sqrt_N * t
            x = torch.cat([x_eq, x_alt.unsqueeze(-1)], dim=-1)

            x = self._reflect(x, self.v[l])  # Householder is self-inverse

        return x

    # ── density and sampling ─────────────────────────────────────────

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log Q(x) on S_N."""
        # Map x back to base, accumulating log density update
        # Equivalent to: z = inverse(x); _, log_dup = forward(z)
        # But more efficient to accumulate during inverse.
        # ... for simplicity, just do forward on the inverse:
        z = self.inverse(x)
        _, log_dup = self.forward(z)
        return -self._log_vol - log_dup

    @torch.no_grad()
    def sample_uniform(self, n: int) -> torch.Tensor:
        """Sample uniformly on S_N."""
        z = torch.randn(n, self.N)
        return z / z.norm(dim=-1, keepdim=True) * self.N**0.5

    def sample_with_log_prob(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x ~ Q and return (x, log Q(x)).

        Uses reparameterized sampling for gradient flow.
        """
        # Uniform base sample (detached — gradient flows only through flow)
        z = torch.randn(n, self.N)
        z = z / z.norm(dim=-1, keepdim=True) * self.N**0.5
        z = z.detach()

        x, log_dup = self.forward(z)
        log_q = -self._log_vol - log_dup
        return x, log_q

    def l2_penalty(self) -> torch.Tensor:
        """L2 on all parameters (altitude + Householder vectors)."""
        return sum(p.pow(2).sum() for p in self.parameters())
