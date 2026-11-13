import inspect
import io
import gzip
import zipfile

import numpy as np
import pytest

from conceptfan_discovery.trialwise_dualstream_fuzzy_v70 import (
    TrialOnlyDualStreamBank
)
from scripts.experiment_real_uci308_two_gas_v70 import (
    load_uci308_genuine_independent_trials,signed_concept_ridge,
    physical_trial_metrics,partial_spearman,physical_semantics_audit,
)


def sixty_second_five_minute_bins(seed=35):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(58,60,16))*.10
    # Synthetic physical sensor mixing for only unit tests; never
    # described as an independently collected experimental dataset.
    a=rng.uniform(0,1,size=(58,2))
    mix=rng.normal(size=(2,16))
    exposure=np.linspace(0,1,60)[None,:,None]
    x+=a[:,None,:]@mix[None,:,:]
    x+=exposure*(a@mix)[:,None,:]
    return x,a


def test_source_is_fit_on_39_whole_trials_never_936_sensor_rows():
    data,y=sixty_second_five_minute_bins()
    assert list(inspect.signature(TrialOnlyDualStreamBank.fit_unlabeled).parameters)==[
        "self","train_x"]
    bank=TrialOnlyDualStreamBank().fit_unlabeled(data[:39])
    assert bank.receipt["number_of_original_independent_training_trials"]==39
    assert bank.receipt["no_individual_sensor_time_bin_treated_as_independent_trial"]
    assert bank.receipt["concentration_references_or_labels_used_to_fit"] is False
    assert 1<=bank.K<=4
    assert len(bank.width)==6*bank.K
    assert bank.receipt["features_after_whole_and_late_summaries"]==12*bank.K
    z=bank.fuzzy_features(data[39:])
    assert z.shape==(19,12*bank.K)
    assert (z>=0).all() and (z<=1).all()


def test_causal_absolute_and_change_channels_have_distinct_information():
    x,y=sixty_second_five_minute_bins()
    bank=TrialOnlyDualStreamBank().fit_unlabeled(x[:39])
    a=bank.source_channels(x[39:42])
    altered=x[39:42].copy()
    altered[:,35:,:]+=100.
    b=bank.source_channels(altered)
    np.testing.assert_array_equal(a[:,:35,:],b[:,:35,:])
    assert np.max(np.abs(a[:,35:,:]-b[:,35:,:]))>.01
    raw=bank.raw_features(x[39:])
    assert raw.shape==(19,32)
    assert bank.linear_features(x[39:]).shape==(19,4*bank.K)


def test_exact_signed_ridge_head_no_raw_sensor_shortcut():
    x,y=sixty_second_five_minute_bins()
    bank=TrialOnlyDualStreamBank().fit_unlabeled(x[:39])
    membership=bank.fuzzy_features(x)
    predicted,receipt=signed_concept_ridge(membership[:39],y[:39,0],
                                            membership[39:])
    assert predicted.shape==(19,)
    assert receipt["max_source_only_signed_additive_prediction_error"]<1e-9
    assert receipt["no_original_sensor_to_risk_bypass"]
    assert receipt["supervised_gas_preparation_doses_used_only_in_head"]
    assert bank.sha==bank.public_passport()["source_sha256"]


def test_physical_trial_metrics_never_confuse_8_trials_with_sample_timesteps():
    truth=np.array([0.,.1,.3,1.,.0,.1,.3,1.])
    metrics=physical_trial_metrics(truth,np.zeros(8),TRAIN_std=.5)
    assert metrics["n_independent_experiments"]==8
    assert metrics["sample_size_too_small_for_stable_error_claim"]
    assert metrics["nominal_water_dilution_volpercent_MAE"]>0
    assert metrics["R2"]<0
    constant=physical_trial_metrics(np.zeros(8),np.zeros(8),TRAIN_std=.5)
    assert constant["R2"] is None


def test_semantic_matching_never_claims_physical_chemical_identity():
    x,y=sixty_second_five_minute_bins()
    bank=TrialOnlyDualStreamBank().fit_unlabeled(x[:39])
    audit=physical_semantics_audit(
        bank,x,y,{"TRAIN":np.arange(39),
                  "VALID":np.arange(39,50),
                  "TEST":np.arange(50,58)})
    assert audit["all_gas_names_used_only_after_unsupervised_source_fit"]
    match=audit["VALID_only_postfit_axis_matching"]
    assert "status" in match
    assert match["status"].startswith("ABSTAIN") or not match[
        "physical_source_identity_proven"]


def test_real_trial_raw_schema_rejects_missing_physical_source(tmp_path):
    archive=tmp_path/"missing_original.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("features.csv","exp,batch,ace_conc,eth_conc\n")
    with pytest.raises(ValueError,match="Unexpected official UCI308 archive"):
        load_uci308_genuine_independent_trials(archive)


def test_short_or_nonfinite_physical_timeline_rejected():
    source=TrialOnlyDualStreamBank()
    with pytest.raises(ValueError):
        source.fit_unlabeled(np.zeros((39,50,16)))
    with pytest.raises(ValueError):
        source.fit_unlabeled(np.full((39,60,16),np.nan))
    with pytest.raises(ValueError):
        source.fit_unlabeled(np.zeros((38,60,16)))
    with pytest.raises(ValueError):
        source.fit_unlabeled(np.zeros((39,60,16)))


def test_independent_gas_dose_partial_association_rejects_unidentifiable():
    assert partial_spearman(np.ones(8),np.arange(8),np.arange(8)) is None
    rng=np.random.default_rng(9)
    data=rng.normal(size=20)
    unrelated=rng.normal(size=20)
    assert partial_spearman(data,data,unrelated)>.99
