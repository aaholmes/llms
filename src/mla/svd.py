"""Activation-aware SVD primitive for MHA→MLA conversion (Stage B).

Given a weight W (d_out, d_in) and per-layer input covariance C (d_in, d_in),
find low-rank factors A (d_out, r), B (r, d_in) such that A B approximates W
minimizing ‖(AB − W) X‖_F², where C = X Xᵀ.

Standard SVD-LLM formulation:

    C = Rᵀ R                  (Cholesky)
    M = W Rᵀ                  (whitened weight)
    M = U Σ Vᵀ                (SVD)
    truncate to rank r:
    A = U_r Σ_r,  B = V_rᵀ R⁻ᵀ

Reconstruction error in the C-weighted norm equals Σ_{i>r} σ_i² of M — the
sum of squared discarded singular values of W Rᵀ.

All math runs in fp64; the caller is responsible for the final dtype cast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class _Whitening:
    """Whitening factor R such that Rᵀ R ≈ C."""

    R: torch.Tensor              # (d_in, d_in)
    is_triangular: bool          # True from Cholesky; False from eigendecomp fallback


def _ridge_cholesky(
    C: torch.Tensor,
    ridge_lambda: float,
    ridge_max_iters: int,
) -> _Whitening:
    """Cholesky with ridge backoff; eigendecomp fallback on persistent failure.

    The ridge is scaled by tr(C)/d so it's invariant to overall covariance
    magnitude. λ exponentiates by 10× per retry.
    """
    d = C.shape[0]
    diag_mean = torch.diagonal(C).mean()
    eye = torch.eye(d, dtype=C.dtype, device=C.device)

    lam = ridge_lambda
    for attempt in range(ridge_max_iters):
        try:
            R = torch.linalg.cholesky(C + lam * diag_mean * eye, upper=True)
            return _Whitening(R=R, is_triangular=True)
        except Exception:  # noqa: BLE001 — torch raises various error subtypes
            lam *= 10
            if attempt >= 1:
                logger.warning(
                    "Cholesky failed at attempt %d; escalating ridge to λ=%.3e (d=%d)",
                    attempt + 1, lam, d,
                )

    logger.warning(
        "Cholesky failed after %d ridge retries (final λ=%.3e, d=%d); "
        "falling back to eigendecomposition",
        ridge_max_iters, lam, d,
    )
    # Persistent failure: eigendecomp + clamp eigvals at 1e-8.
    # R = sqrt(Λ) Vᵀ satisfies Rᵀ R = V Λ Vᵀ = C.
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = torch.clamp(eigvals, min=1e-8)
    R = (eigvecs * torch.sqrt(eigvals).unsqueeze(0)).T
    return _Whitening(R=R, is_triangular=False)


def activation_aware_factor(
    W: torch.Tensor,
    C: torch.Tensor,
    rank: int,
    *,
    ridge_lambda: float = 1e-6,
    ridge_max_iters: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor W as A B minimizing ‖(AB − W) X‖_F² weighted by C = X Xᵀ.

    Parameters
    ----------
    W : (d_out, d_in)
        Weight matrix to factor.
    C : (d_in, d_in)
        Symmetric PSD activation covariance.
    rank : int
        Target rank, in ``[1, min(W.shape)]``.
    ridge_lambda : float
        Initial ridge factor (scaled by tr(C)/d_in inside).
    ridge_max_iters : int
        Cholesky retry budget; on exhaustion, falls back to eigendecomp.

    Returns
    -------
    A : (d_out, rank), fp64
    B : (rank, d_in), fp64
    """
    if rank < 1 or rank > min(W.shape):
        raise ValueError(
            f"rank must be in [1, min(W.shape)]; got rank={rank}, W.shape={tuple(W.shape)}"
        )

    W64 = W.to(torch.float64)
    C64 = C.to(torch.float64)

    whitening = _ridge_cholesky(C64, ridge_lambda, ridge_max_iters)
    R = whitening.R

    M = W64 @ R.T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    A = U_r * S_r.unsqueeze(0)

    # B = V_rᵀ R⁻ᵀ. With R upper-triangular, solve R Y = V_rᵀ then B = Yᵀ:
    #   R Y = V_rᵀ  ⟹  Y = R⁻¹ V_rᵀ  ⟹  Yᵀ = V_r R⁻ᵀ = (V_rᵀ R⁻ᵀ)ᵀ ✓
    if whitening.is_triangular:
        B = torch.linalg.solve_triangular(R, Vh_r.T, upper=True).T
    else:
        # Eigendecomp fallback: R is dense; use general solve.
        B = torch.linalg.solve(R.T, Vh_r.T).T

    return A, B


def discarded_singular_energy(
    W: torch.Tensor,
    C: torch.Tensor,
    rank: int,
    *,
    ridge_lambda: float = 1e-6,
    ridge_max_iters: int = 6,
) -> torch.Tensor:
    """Σ_{i>r} σ_i² of W Rᵀ — the analytic C-weighted reconstruction error.

    Uses the same ridge schedule as :func:`activation_aware_factor` so the
    bound is comparable to its empirical residual. Useful for energy-threshold
    rank selection (compare against the cumulative singular-value energy).
    """
    W64 = W.to(torch.float64)
    C64 = C.to(torch.float64)

    R = _ridge_cholesky(C64, ridge_lambda, ridge_max_iters).R
    M = W64 @ R.T
    S = torch.linalg.svdvals(M)

    if rank >= S.numel():
        return torch.zeros((), dtype=torch.float64, device=W.device)
    return (S[rank:] ** 2).sum()
