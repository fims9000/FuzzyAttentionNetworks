"""V51 tests for sparse TRAIN-only energy calibration, two-VALID and fallback."""
from dataclasses import replace

import numpy as np
import pytest

from conceptfan_discovery.observational_twins_v51 import make_observational_twins_v51
from conceptfan_discovery.independent_anchor_certificate_v50 import (
    fit_attested_source_calibration_v50,infer_calibrated_source_energy_v50,
)
from conceptfan_discovery.train_only_energy_debias_v51 import (
    energy_reference_from_independent_assay_v51,
    fit_train_only_energy_debias_v51,infer_energy_with_train_only_debias_v51,
)


def _fixture():
    w=make_observational_twins_v51(
        20263307,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    x=w["shared_observed"];mask=w["measured_boolean_masks"]
    t,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    world=w["worlds"][0];assay=world["attested_external_assay"]
    seed=20265041+20263307%101
    source=fit_attested_source_calibration_v50(
        x[t],mask["TRAIN_MCAR20"][t],assay[t],
        x[v1],mask["VALID1_MCAR20"][v1],assay[v1],
        x[v2],mask["VALID2_MCAR40"][v2],assay[v2],
        physical_device_id=world["physical_device_id"],
        independent_reference_id=world["external_reference_id"],
        train_reference_budget_fraction=.05,seed=seed,
    )
    energy=fit_train_only_energy_debias_v51(
        source,
        x[t],mask["TRAIN_MCAR20"][t],assay[t],
        x[v1],mask["VALID1_MCAR20"][v1],assay[v1],
        x[v2],mask["VALID2_MCAR40"][v2],assay[v2],
        train_reference_budget_fraction=.05,seed=seed)
    return w,source,energy


def test_same_paid_assay_budget_and_positive_correction():
    world,source,cal=_fixture()
    assert source.dual_VALID_admitted
    assert cal.train_anchor_count==25
    assert cal.valid1_anchor_count==cal.valid2_anchor_count==80
    assert .5<=cal.scale_alpha<=2.
    assert .5<=cal.affine_alpha<=2.
    assert -.30<=cal.affine_intercept<=.30
    assert cal.target_is_noisy_assay_not_hidden_true_energy
    assert not cal.physical_identity_not_proven is False


def test_external_sensor_noise_correction_nonnegative():
    assay=np.ones((4,16),dtype=float)
    energies=energy_reference_from_independent_assay_v51(assay)
    assert np.array_equal(energies,np.zeros(4))
    with pytest.raises(ValueError):
        energy_reference_from_independent_assay_v51(np.zeros((4,5)))


def test_failed_two_valid_correction_never_changes_previous_source():
    w,source,cal=_fixture()
    x=w["shared_observed"][w["test"]]
    mask=w["measured_boolean_masks"]["TEST_MCAR40"][w["test"]]
    raw=infer_calibrated_source_energy_v50(source,x,mask)["inferred_target_energy"]
    broken=replace(cal,scale_approved_two_VALID=False)
    result=infer_energy_with_train_only_debias_v51(
        source,broken,x,mask,mode="V51_SCALE")
    np.testing.assert_array_equal(result["energy"],raw)
    assert result["decision"]=="CORRECTION_REJECTED_EXACT_RAW_FALLBACK"
    assert not result["correction_actually_used"]


def test_hidden_payload_never_enters_corrected_energy():
    w,source,cal=_fixture()
    x=w["shared_observed"][w["test"]]
    mask=w["measured_boolean_masks"]["TEST_MCAR40"][w["test"]]
    clean=infer_energy_with_train_only_debias_v51(source,cal,x,mask,mode="V51_SCALE")
    poison=x.copy();poison[~mask]=np.nan
    masked=infer_energy_with_train_only_debias_v51(
        source,cal,poison,mask,mode="V51_SCALE")
    np.testing.assert_array_equal(clean["energy"],masked["energy"])
    assert not clean["oracle_latent_true_energy_used"]
    assert not clean["predictor_or_native_ConceptFAN_risk_modified"]


def test_device_mismatch_and_extra_reference_budget_fails():
    w,source,cal=_fixture()
    wrong=replace(cal,external_reference_id="other_unverified_assay")
    with pytest.raises(ValueError):
        infer_energy_with_train_only_debias_v51(
            source,wrong,w["shared_observed"][w["test"]],
            w["measured_boolean_masks"]["TEST_MCAR20"][w["test"]],
            mode="V51_AFFINE")
    x=w["shared_observed"];mask=w["measured_boolean_masks"]
    t,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    assay=w["worlds"][0]["attested_external_assay"]
    with pytest.raises(AssertionError):
        fit_train_only_energy_debias_v51(
            source,x[t],mask["TRAIN_MCAR20"][t],assay[t],
            x[v1],mask["VALID1_MCAR20"][v1],assay[v1],
            x[v2],mask["VALID2_MCAR40"][v2],assay[v2],
            train_reference_budget_fraction=.20,seed=20265041+20263307%101)
