"""Algebraic tests: known inverse matrices are fixture ONLY, never inference."""
from types import SimpleNamespace
import numpy as np
import pytest
from conceptfan_discovery.partial_view_concept_v44 import (
    partial_pair_energy_v44, fit_partial_view_concept_v44, partial_view_concept_v44,
)


def simulated_views(n=1300, seed=11):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, 16, 2))
    for t in range(1, 16):
        s[:, t, 0] = .76*s[:, t-1, 0] + .65*s[:, t, 0]
        s[:, t, 1] = -.27*s[:, t-1, 1] + .96*s[:, t, 1]
    s *= rng.uniform(.5, 1.8, (n, 1, 1))
    h = [np.array([[1., .3], [.2, 1.2]]),
         np.array([[1.1, -.2], [.15, .8]]),
         np.array([[.75, .4], [-.1, 1.05]])]
    x = rng.normal(size=(n, 16, 12)) * .05
    x[:, :, :4] = rng.normal(size=(n, 16, 4))
    for k, matrix in enumerate(h):
        x[:, :, 4+2*k:6+2*k] = s @ matrix.T + rng.normal(0, .12, (n, 16, 2))
    pairs = {}
    for name, a, b in (("AB", 0, 1), ("AC", 0, 2), ("BC", 1, 2)):
        pairs[name] = SimpleNamespace(
            pair_name=name,
            view_a=(4+2*a, 5+2*a),
            view_b=(4+2*b, 5+2*b),
            inv_left=np.linalg.inv(h[a]),
            inv_right=np.linalg.inv(h[b]),
        )
    fit = SimpleNamespace(pairs=pairs)
    mask = rng.random(x.shape) >= .2
    return x, mask, fit, s[:, :, 0].var(axis=1)


def test_hidden_values_never_leak_into_any_concept():
    x, mask, fit, _ = simulated_views(300)
    result = partial_pair_energy_v44(fit.pairs["AB"], x, mask)
    poison = x.copy()
    poison[~mask] = np.nan
    after = partial_pair_energy_v44(fit.pairs["AB"], poison, mask)
    np.testing.assert_array_equal(result["energy"], after["energy"])
    np.testing.assert_array_equal(result["support"], after["support"])
    with pytest.raises(ValueError):
        partial_pair_energy_v44(fit.pairs["AB"], poison, ~mask)


def test_full_missing_pair_abstains():
    x, mask, fit, _ = simulated_views(300)
    mask[:, :, 4:10] = False
    result = partial_pair_energy_v44(fit.pairs["AB"], x, mask)
    assert not result["support"].any()
    np.testing.assert_array_equal(result["energy"], 0.)


def test_validated_partial_support_and_trace():
    x, m, v37, latent_energy = simulated_views()
    fitted = fit_partial_view_concept_v44(v37, x[:800], m[:800])
    mask40 = np.random.default_rng(8877).random(x[800:].shape) >= .40
    pred = partial_view_concept_v44(fitted, x[800:], mask40)
    strict_ab = mask40[:, :, 4:8].all(axis=-1).sum(axis=1) >= 4
    assert pred["support"].mean() > 1.5 * strict_ab.mean()
    assert pred["support"].mean() > .3
    assert np.isfinite(pred["q"]).all()
    assert set(pred["selected_pair"]) <= {"AB", "AC", "BC", "--"}
    assert set(pred["decision_reason"]) <= {
        "NO_MEASURED_PAIR", "MULTIVIEW_CORROBORATED",
        "SINGLE_PAIR_NOT_CORROBORATED",
        "MULTIVIEW_DISAGREEMENT_OR_NO_TRAIN_CAL",
    }
    assert fitted.audit()["physical_semantics_certified"] is False
    assert pred["single_pair_uncorroborated"].sum() > 0
    poison = x[800:].copy()
    poison[~mask40] = np.nan
    after = partial_view_concept_v44(fitted, poison, mask40)
    np.testing.assert_array_equal(pred["q"], after["q"])
    np.testing.assert_array_equal(pred["support"], after["support"])
    eligible = pred["support"] > 0
    assert np.corrcoef(latent_energy[800:][eligible], pred["q"][eligible])[0,1] > .2


def test_all_missing_returns_uninformative_concept():
    x, m, v37, _ = simulated_views()
    fitted = fit_partial_view_concept_v44(v37, x[:800], m[:800])
    empty = np.zeros_like(m[800:])
    p = partial_view_concept_v44(fitted, np.full_like(x[800:], np.nan), empty)
    assert (p["q"] == .5).all()
    assert (p["support"] == 0).all()
    assert (p["selected_pair"] == "--").all()
