import numpy as np

from conceptfan_realdata.conditional_residual import (
    fit_probe_and_predict, paired_patient_bootstrap, proxy_summary,
)


def test_proxy_summary_shapes_and_masked_values():
    values = np.arange(2 * 5 * 3).reshape(2, 5, 3).astype(float)
    mask = np.ones_like(values)
    mask[0, :, 0] = 0
    result = proxy_summary(values, mask)
    assert result.shape == (2, 18)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, :6], [0., 0., 0., 0., 0., 1.])


def test_paired_identical_predictions_have_zero_gain():
    y = np.array([0, 1] * 30)
    probability = np.linspace(.03, .95, len(y))
    result = paired_patient_bootstrap(y, probability, probability, repetitions=150)
    assert abs(result["delta_auprc"]) < 1e-14
    assert abs(result["delta_nll"]) < 1e-14
    np.testing.assert_allclose(result["delta_auprc_ci95"], [0., 0.], atol=1e-14)


def test_calibrated_model_returns_finite_probabilities():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(340, 4))
    y = (x[:, 0] - .5 * x[:, 1] > .3).astype(int)
    result, c = fit_probe_and_predict(
        x[:160], y[:160], x[160:230], y[160:230],
        x[230:290], y[230:290], x[290:],
    )
    assert c in (.01, .1, 1., 10., 100.)
    assert result.shape == (50,)
    assert np.all((result > 0) & (result < 1))
