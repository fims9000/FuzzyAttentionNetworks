"""Four fresh V45 synthetic devices, sharing documented V41 generator families.

External physical ON/OFF isolation and view-noise independence remain
simulator assumptions, NOT unsupervised clinical/causal discovery.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit

from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world
from conceptfan_discovery.paired_isolation_world_v39 import _ar_process, _random_H
from conceptfan_discovery.adaptive_pair_bench_world_v41 import _source_laws, VIEWS

DEVICES_V45 = {
    20261907: "linear_AR", 20261909: "linear_AR",
    20261921: "nonlinear_switching", 20261923: "nonlinear_switching",
}


def fresh_device_v45(*, device_seed: int, n_train=1000, n_valid=800, n_test=600):
    if (device_seed not in DEVICES_V45 or min(n_train,n_valid,n_test) < 100
            or n_valid % 2):
        raise ValueError("Unregistered device or invalid partition sizes")
    family = DEVICES_V45[device_seed]
    base = phase_pulse_world(seed=device_seed+331, n_train=n_train,
                             n_valid=n_valid, n_test=n_test)
    rng = np.random.default_rng(device_seed+52063)
    n = len(base["x"])
    source = _source_laws(rng, n, family)
    artifact = .35 * _ar_process(rng, n, .77)
    mixing = tuple(_random_H(rng) for _ in VIEWS)
    noise_on = rng.normal(0., .19, size=(n,16,6))
    noise_off = rng.normal(0., .19, size=(n,16,6))
    observed = base["x"].copy()
    observed[:,:,10] = rng.normal(0., .08, size=(n,16))
    observed[:,:,11] = rng.normal(size=(n,16))
    slow = source[:,:,0]
    energy = np.mean((slow-slow.mean(axis=1,keepdims=True))**2, axis=1)
    known = base["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
    base_logit = 3.7*(known[:,0]-.5)-3.6*(known[:,1]-.5)-.10
    coin = rng.random(n)
    return {
        "family": family, "device_seed": device_seed,
        "train": base["train"], "valid1": base["valid1"],
        "valid2": base["valid2"], "test": base["test"],
        "outcomes": {
            "energy": (coin < expit(base_logit+2.2*(energy-.65))).astype(np.int64),
            "complete": (coin < expit(base_logit)).astype(np.int64),
        },
        "simulator_AUDIT_ONLY": {
            "family": family, "base": observed, "hidden_source": source,
            "hidden_artifact": artifact, "mixing": mixing,
            "on_readout_error": noise_on, "off_readout_error": noise_off,
        },
        "true_slow_source_energy_POSTFIT_AUDIT_ONLY": energy,
        "new_draws_not_previous_v43_test": True,
        "physical_ON_OFF_isolation_is_EXTERNAL_ASSUMPTION": True,
    }
