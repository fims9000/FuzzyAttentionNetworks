import numpy as np

from conceptfan_discovery.crossview_response_bank_v62 import (
    TrainOnlyCrossViewBank,crossview_rank_evidence,make_response_views
)


def two_response_sources(n=120,d=8,t=60,seed=41):
    rng=np.random.default_rng(seed)
    flow=rng.normal(size=(n,2))
    mixing=rng.normal(size=(2,d))
    base=rng.normal(0,.08,size=(n,t,d))
    exposure=np.zeros((n,t,d))
    exposure[:,12:48,:]=flow[:,None,:]@mixing
    x=base+exposure
    return x,np.ones_like(x,dtype=bool)


def test_observable_multiview_rank_discoveries_are_train_only():
    x,m=two_response_sources()
    bank=TrainOnlyCrossViewBank(
        seed=20261801,permutations=35,bootstraps=8).fit_unlabeled(x,m)
    assert bank.k==2
    assert bank.train_receipt["null_is_whole_train_run_permutation"]
    assert not bank.train_receipt["manual_source_number_or_gas_levels_used"]
    assert not bank.source.target_labels_used_to_fit
    assert not bank.source.true_latent_states_used_to_fit
    assert bank.route=="CROSSVIEW_RESPONSE"
    assert len(bank.width)==2


def test_source_rank_blank_observations_abstains():
    x=np.zeros((120,60,8),dtype=np.float64)
    m=np.ones_like(x,dtype=bool)
    bank=TrainOnlyCrossViewBank(
        seed=11,permutations=35,bootstraps=8).fit_unlabeled(x,m)
    assert bank.k==0
    assert bank.route=="OBSERVATION_ABSTAIN"
    z,s=bank.transform(x[:6],m[:6])
    assert z.shape==(6,60,0)
    assert s.shape==(6,60,0)


def test_hidden_data_values_never_enter_projection_or_mask():
    x,m=two_response_sources()
    bank=TrainOnlyCrossViewBank(
        seed=11,permutations=35,bootstraps=8).fit_unlabeled(x,m)
    rng=np.random.default_rng(72)
    obs=rng.random(x[:8].shape)>.15
    hidden=x[:8].copy()
    hidden[~obs]=np.nan
    first,coverage=bank.transform(x[:8],obs)
    second,coverage2=bank.transform(hidden,obs)
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(coverage,coverage2)
    all_missing=np.zeros_like(obs,dtype=bool)
    z,s=bank.transform(hidden,all_missing)
    np.testing.assert_array_equal(z,np.zeros_like(z))
    np.testing.assert_array_equal(s,np.zeros_like(s))


def test_missing_earlier_view_is_not_imputed():
    x,m=two_response_sources()
    m[:2,:6,0]=False
    try:
        make_response_views(x,m)
    except ValueError as error:
        assert "unavailable sensors" in str(error)
    else:
        raise AssertionError("Withheld source view cannot be secretly imputed")


def test_late_recovery_change_does_not_modify_exposure_source_fit():
    x,m=two_response_sources()
    a,b=make_response_views(x,m)
    changed=x.copy()
    changed[:,48:,:]=1000.
    aa,bb=make_response_views(changed,m)
    np.testing.assert_array_equal(a,aa)
    np.testing.assert_array_equal(b,bb)


def test_original_temporal_source_data_not_treated_as_360_independent_runs():
    x,m=two_response_sources()
    one,other=make_response_views(x,m)
    evidence=crossview_rank_evidence(
        one,other,seed=9,permutations=35,bootstraps=8)
    assert evidence["source_fit_train_independent_runs"]==120
    assert evidence["null_pairings_are_whole_runs"]
    assert len(evidence["max_singular_value_null_samples"])==35
    assert len(evidence["train_whole_run_bootstrap_ranks"])==8
    assert evidence["rank"]>=2
