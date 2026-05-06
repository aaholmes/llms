"""Tests for activation-aware SVD primitive (Stage B.2).

Verifies:
  - Full-rank factor reconstructs W exactly (fp64).
  - With C = I, the activation-aware factor reduces to plain truncated SVD.
  - An exactly-rank-r₀ W is recovered without error at rank r₀.
  - Near-singular C is rescued by ridge backoff (no NaN/Inf).
  - Bound invariant: ‖(W − AB) Rᵀ‖_F² == Σ_{i>r} σ_i²(W Rᵀ).
"""

from __future__ import annotations

import pytest
import torch

from mla.svd import activation_aware_factor, discarded_singular_energy


def test_output_shapes():
    W = torch.randn(15, 20, dtype=torch.float64)
    C = torch.eye(20, dtype=torch.float64)
    A, B = activation_aware_factor(W, C, rank=8)
    assert A.shape == (15, 8)
    assert B.shape == (8, 20)


def test_full_rank_well_conditioned():
    """Sanity 1: full rank, well-conditioned C → A B == W to fp64 precision."""
    torch.manual_seed(0)
    d_out, d_in = 32, 24
    W = torch.randn(d_out, d_in, dtype=torch.float64)
    Z = torch.randn(d_in, d_in, dtype=torch.float64)
    C = torch.eye(d_in, dtype=torch.float64) + 0.1 * (Z @ Z.T)

    A, B = activation_aware_factor(W, C, rank=d_in)
    err = torch.linalg.norm(A @ B - W) / torch.linalg.norm(W)
    assert err < 1e-6, f"Full-rank reconstruction error {err:.2e} > 1e-6"


def test_uniform_C_reduces_to_plain_svd():
    """Sanity 2: C = I → activation-aware reduces to plain truncated SVD."""
    torch.manual_seed(1)
    d_out, d_in = 16, 16
    W = torch.randn(d_out, d_in, dtype=torch.float64)
    C = torch.eye(d_in, dtype=torch.float64)

    rank = 5
    A, B = activation_aware_factor(W, C, rank=rank)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_r = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]

    err = torch.linalg.norm(A @ B - W_r) / torch.linalg.norm(W_r)
    assert err < 1e-9, f"With C=I, AB diverges from plain SVD by {err:.2e}"


def test_synthetic_exact_low_rank():
    """Sanity 3: W exactly rank r₀ → factor at rank r₀ recovers W."""
    torch.manual_seed(2)
    d_out, d_in = 24, 32
    r0 = 6

    A_true = torch.randn(d_out, r0, dtype=torch.float64)
    B_true = torch.randn(r0, d_in, dtype=torch.float64)
    W = A_true @ B_true

    Z = torch.randn(d_in, d_in, dtype=torch.float64)
    C = torch.eye(d_in, dtype=torch.float64) + 0.5 * (Z @ Z.T)

    A, B = activation_aware_factor(W, C, rank=r0)
    err = torch.linalg.norm(A @ B - W) / torch.linalg.norm(W)
    assert err < 1e-10, f"Exact rank-{r0} W not recovered: err={err:.2e}"


def test_near_singular_C_ridge_recovers():
    """Stability: C with one tiny eigenvalue → ridge backoff prevents NaN/Inf."""
    torch.manual_seed(3)
    d = 20

    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    eigvals = torch.linspace(1.0, 0.5, d, dtype=torch.float64)
    eigvals[-1] = 1e-12
    C = Q @ torch.diag(eigvals) @ Q.T
    C = (C + C.T) / 2  # symmetrize against tiny asymmetry from finite-precision matmul

    W = torch.randn(d, d, dtype=torch.float64)

    A, B = activation_aware_factor(W, C, rank=10)
    assert torch.isfinite(A).all(), "A has non-finite values"
    assert torch.isfinite(B).all(), "B has non-finite values"
    assert torch.isfinite(A @ B).all(), "A @ B has non-finite values"


def test_bound_invariant():
    """‖(W − AB) Rᵀ‖_F² == Σ_{i>r} σ_i²(W Rᵀ) — the analytic reconstruction bound."""
    torch.manual_seed(4)
    d_out, d_in = 20, 30
    rank = 8

    Z = torch.randn(d_in, d_in, dtype=torch.float64)
    C = torch.eye(d_in, dtype=torch.float64) * 0.5 + Z @ Z.T  # well-conditioned PSD
    W = torch.randn(d_out, d_in, dtype=torch.float64)

    A, B = activation_aware_factor(W, C, rank=rank)

    # Reconstruct the same R the factor used (default ridge_lambda=1e-6).
    diag_mean = torch.diagonal(C).mean()
    R = torch.linalg.cholesky(
        C + 1e-6 * diag_mean * torch.eye(d_in, dtype=torch.float64),
        upper=True,
    )
    empirical_err = torch.linalg.norm((W - A @ B) @ R.T) ** 2

    analytic_bound = discarded_singular_energy(W, C, rank=rank)

    rel_err = abs(empirical_err - analytic_bound) / analytic_bound
    assert rel_err < 1e-6, (
        f"Empirical {empirical_err:.4e} vs analytic {analytic_bound:.4e}, rel {rel_err:.2e}"
    )


def test_invalid_rank_raises():
    W = torch.randn(10, 8, dtype=torch.float64)
    C = torch.eye(8, dtype=torch.float64)
    with pytest.raises(ValueError):
        activation_aware_factor(W, C, rank=0)
    with pytest.raises(ValueError):
        activation_aware_factor(W, C, rank=9)


def test_discarded_energy_full_rank_is_zero():
    """At rank = min(d_out, d_in), no singular values are discarded."""
    W = torch.randn(15, 20, dtype=torch.float64)
    C = torch.eye(20, dtype=torch.float64)
    energy = discarded_singular_energy(W, C, rank=15)
    assert energy.item() < 1e-12
