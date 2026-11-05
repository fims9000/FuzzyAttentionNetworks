from __future__ import annotations

import numpy as np

from scripts.experiment_concept_leakage_guard_v27 import (
    conditional_leakage_audit, conditional_third_factor_alignment,
    fit_concepts_without_outcome_gradients,
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    frozen_world, make_model, mask_mechanism, sha_state,
)


def test_stage1_concept_definition_independent_of_outcome_law():
    w=frozen_world(seed=20261107,n_train=128,n_cal=32,n_test=160)
    tr=w["index_train"]
    m=mask_mechanism(w["x"][tr],"MCAR20",seed=20262101)
    m1=make_model("native",2,seed=11)
    m2=make_model("native",2,seed=11)
    fit_concepts_without_outcome_gradients(
        m1,w["x"][tr],m,w["concept_trajectories"][tr],seed=11,epochs=1)
    fit_concepts_without_outcome_gradients(
        m2,w["x"][tr],m,w["concept_trajectories"][tr],seed=11,epochs=1)
    assert sha_state(m1)==sha_state(m2)
    assert not np.array_equal(w["y_complete"],w["y_omitted"])


def test_conditional_leakage_test_separates_fit_and_test():
    rng=np.random.default_rng(20)
    ntr,nte=256,192
    z=rng.normal(size=(ntr+nte,3))
    true2=z[:,:2]
    proxy=np.column_stack([true2[:,0]+.4*z[:,2],
                           true2[:,1]-.5*z[:,2]])
    p=1/(1+np.exp(-(true2[:,0]-true2[:,1]+2.2*z[:,2])))
    y=rng.binomial(1,p)
    report=conditional_leakage_audit(
        y[:ntr],y[ntr:],true2[:ntr],true2[ntr:],
        proxy[:ntr],proxy[ntr:],seed=20,bootstraps=199)
    assert report["n_test"]==nte
    assert np.isfinite(report["incremental_learned_concept_nll_gain"])
    assert len(report["paired_test_subject_bootstrap_ci95"])==2
    association=conditional_third_factor_alignment(
        true2[:ntr],true2[ntr:],
        proxy[:ntr],proxy[ntr:],z[ntr:,2])
    assert association["max_abs_hidden_factor_correlation"]>.4


def test_oracle_test_source_not_in_prediction_feature_matrix():
    w=frozen_world(seed=20261107,n_train=128,n_cal=32,n_test=160)
    assert w["concept_trajectories"].shape[-1]==3
    assert w["index_test"][0] >= w["index_train"][-1]+1
    assert np.array_equal(w["y_artifact"],w["y_omitted"])
