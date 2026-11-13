import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.train_innovation_ica_bank_v63 import TrainOnlyInnovationBank
from scripts.experiment_bottleneck_ablation_v64 import physical_sources


def sample(n=320):
    rng=np.random.default_rng(74)
    x=rng.normal(size=(n,120,16))
    y=rng.normal(size=(n,2))
    x[:,:,0]+=y[:,0,None]*2
    x[:,90:120,1]+=y[:,1,None]
    group=np.asarray([f"block:{i//8}" for i in range(n)])
    return x,np.ones_like(x,bool),group,y


def test_all_eleventh_ablation_stages_use_only_observed_sensor_values():
    x,m,groups,_=sample()
    sources={k:TrainOnlyInnovationBank(
        basis=k,block_bootstraps=5).fit_unlabeled(
            x,m,train_halfhour_blocks=groups)
        for k in ("pca","ica")}
    k=sources["pca"].k
    raw=np.column_stack((x.mean(axis=1),x[:,90:120].mean(axis=1)))
    abs_pca=make_pipeline(StandardScaler(),PCA(n_components=k)).fit(raw)
    out=physical_sources(x,m,sources,abs_pca)
    assert len(out)==11
    assert out["raw_absolute_32"].shape==(320,32)
    assert out["raw_initial_differenced_32"].shape==(320,32)
    assert out["pca_projected_delta_fuzzy_all_K"].shape==(320,k)
    assert out["ica_projected_absolute_linear_K"].shape==(320,2*k)
    assert set(v.shape[0] for v in out.values())=={320}


def test_remove_absolute_levels_changes_raw_not_differenced_features():
    x,m,groups,_=sample()
    sources={k:TrainOnlyInnovationBank(
        basis=k,block_bootstraps=5).fit_unlabeled(
            x,m,train_halfhour_blocks=groups)
        for k in ("pca","ica")}
    k=sources["ica"].k
    raw=np.column_stack((x.mean(axis=1),x[:,90:120].mean(axis=1)))
    abs_pca=make_pipeline(StandardScaler(),PCA(n_components=k)).fit(raw)
    old=physical_sources(x[:6],m[:6],sources,abs_pca)
    changed=x[:6]+100.
    new=physical_sources(changed,m[:6],sources,abs_pca)
    np.testing.assert_allclose(old["raw_initial_differenced_32"],
                               new["raw_initial_differenced_32"],atol=1e-12)
    assert not np.allclose(old["raw_absolute_32"],new["raw_absolute_32"])
    np.testing.assert_allclose(
        old["ica_projected_delta_linear_K"],
        new["ica_projected_delta_linear_K"],atol=1e-5)
    assert not np.allclose(
        old["ica_projected_absolute_linear_K"],
        new["ica_projected_absolute_linear_K"])
