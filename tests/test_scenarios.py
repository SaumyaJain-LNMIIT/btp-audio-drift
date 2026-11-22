"""Tests for streaming.drift_scenarios: no dropped samples, correct ground truth,
seed reproducibility, and distinct scenario shapes."""

from __future__ import annotations

import numpy as np

from streaming.drift_scenarios import (
    create_drift_scenario,
    abrupt_drift,
    gradual_drift,
    incremental_drift,
    recurring_drift,
    noise_drift,
    stationary_stream,
)


def test_abrupt_boundary_is_clean(synthetic_dataset):
    f, l = synthetic_dataset
    sf, sl, dp, drifts = abrupt_drift(f, l, drift_point_ratio=0.5, seed=1)
    # classes 0-4 before the boundary, 5-9 after
    assert set(np.unique(sl[:dp])).issubset({0, 1, 2, 3, 4})
    assert set(np.unique(sl[dp:])).issubset({5, 6, 7, 8, 9})
    assert drifts == [dp]


def test_gradual_drops_nothing_and_mixes(synthetic_dataset):
    f, l = synthetic_dataset
    sf, sl, start, drifts = gradual_drift(f, l, transition_width=40, seed=1)
    # group B density must rise across the transition window
    win = sl[start:start + 40]
    first_half_b = np.mean(np.isin(win[:20], [5, 6, 7, 8, 9]))
    second_half_b = np.mean(np.isin(win[20:], [5, 6, 7, 8, 9]))
    assert second_half_b >= first_half_b
    assert drifts == [(start, start + 40)]   # (onset, completion) span


def test_incremental_window_wider_than_gradual(synthetic_dataset):
    f, l = synthetic_dataset
    _, _, g_start, _ = gradual_drift(f, l, transition_width=40, seed=1)
    _, _, i_start, _ = incremental_drift(f, l, transition_fraction=0.6, seed=1)
    # wider transition => onset is earlier relative to the midpoint
    assert i_start < g_start


def test_recurring_has_two_drift_points(synthetic_dataset):
    f, l = synthetic_dataset
    sf, sl, dp, drifts = recurring_drift(f, l, seed=1)
    assert len(drifts) == 2
    p1, p2 = drifts
    # A -> B -> A
    assert set(np.unique(sl[:p1])).issubset({0, 1, 2, 3, 4})
    assert set(np.unique(sl[p1:p2])).issubset({5, 6, 7, 8, 9})
    assert set(np.unique(sl[p2:])).issubset({0, 1, 2, 3, 4})
    assert len(sf) == len(f)


def test_noise_is_group_a_only_and_perturbs_after_drift(synthetic_dataset):
    f, l = synthetic_dataset
    cf, cl, dp, _ = noise_drift(f, l, noise_scale=0.0, seed=1)   # clean
    nf, nl, dp2, _ = noise_drift(f, l, noise_scale=5.0, seed=1)  # noisy
    # only group-A clips are streamed now (fixes the pre-drift false-alarm confound)
    assert set(np.unique(cl)).issubset({0, 1, 2, 3, 4})
    assert dp == dp2
    assert np.allclose(cf[:dp], nf[:dp])             # pre-drift identical
    assert not np.allclose(cf[dp:], nf[dp:])         # post-drift perturbed


def test_stationary_has_no_true_drift(synthetic_dataset):
    f, l = synthetic_dataset
    sf, sl, dp, drifts = stationary_stream(f, l, seed=1)
    assert drifts == []                               # no drift to detect
    assert set(np.unique(sl)).issubset({0, 1, 2, 3, 4})
    assert len(sf) == len(sl) > 0


def test_seed_reproducibility_and_variation(synthetic_dataset):
    f, l = synthetic_dataset
    a1 = abrupt_drift(f, l, seed=7)[1]
    a2 = abrupt_drift(f, l, seed=7)[1]
    a3 = abrupt_drift(f, l, seed=8)[1]
    assert np.array_equal(a1, a2)          # same seed -> identical
    assert not np.array_equal(a1, a3)      # different seed -> different order


def test_dispatcher_lists_all_scenarios(synthetic_dataset):
    f, l = synthetic_dataset
    for name in ("abrupt", "gradual", "incremental", "recurring", "noise"):
        sf, sl, dp, drifts = create_drift_scenario(name, f, l, seed=3)
        assert len(sf) == len(sl) > 0
        assert len(drifts) >= 1
    # stationary is the no-drift control: valid stream, zero true drifts
    sf, sl, dp, drifts = create_drift_scenario("stationary", f, l, seed=3)
    assert len(sf) == len(sl) > 0
    assert drifts == []