"""Unit tests for Phase 6.2 — loss clipping and eval-mode enforcement."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.ensemble_manager import EnsembleManager, enforce_eval_mode


def _ensemble(k=3, d=8, e=4):
    ens = EnsembleManager(num_students=k, input_dim=d, hidden_dims=[16], embedding_dim=e, device="cpu")
    for s in ens.students:
        s.eval()
    return ens


def _inputs(d=8, e=4):
    torch.manual_seed(0)
    x = torch.randn(1, d)
    h = torch.randn(1, e)
    return x, h


def test_unclipped_losses_nonnegative():
    ens = _ensemble()
    x, h = _inputs()
    losses = ens.compute_losses(x, h)
    assert len(losses) == 3
    assert all(l >= 0.0 for l in losses)


def test_clip_caps_every_loss():
    ens = _ensemble()
    x, h = _inputs()
    raw = ens.compute_losses(x, h)
    c = 0.5 * max(raw)  # a cap below the largest raw loss
    clipped = ens.compute_losses(x, h, clip_c=c)
    assert all(0.0 <= l <= c + 1e-12 for l in clipped)
    assert max(clipped) <= c + 1e-12
    # at least one loss was actually clipped down
    assert any(cl < rw for cl, rw in zip(clipped, raw))


def test_clip_zero_gives_all_zero():
    ens = _ensemble()
    x, h = _inputs()
    clipped = ens.compute_losses(x, h, clip_c=0.0)
    assert all(l == 0.0 for l in clipped)


def test_enforce_eval_strict_raises_on_train_mode():
    ens = _ensemble()
    ens.students[0].train()  # put one student into train mode
    with pytest.raises(RuntimeError):
        enforce_eval_mode(*ens.students, strict=True)


def test_enforce_eval_nonstrict_sets_eval():
    ens = _ensemble()
    ens.students[0].train()
    enforce_eval_mode(*ens.students, strict=False)
    assert all(not s.training for s in ens.students)


def test_population_variance_matches_bound_shape():
    # population variance of {0, C} split should match floor/ceil formula
    for k, c in [(5, 2.0), (4, 1.5), (3, 1.0)]:
        m = k // 2
        losses = [0.0] * (k - m) + [c] * m
        _, var = EnsembleManager.compute_statistics(losses)
        vmax = (k // 2) * ((k + 1) // 2) / k**2 * c**2
        assert var <= vmax + 1e-9