import itertools
import numpy as np
import pytest
from conceptfan_realdata.crossed_seed import exact_cluster_signflip, holm, crossed_seed_bootstrap_ci

IDS=[f"m{i}{j}" for i in range(4) for j in range(3)]
PAIRS=list(itertools.combinations(IDS,2))
SEEDS={f"m{i}{j}":(i,j) for i in range(4) for j in range(3)}

def test_signflip_ten_initializations_three_orders():
    d=np.ones((10,3))
    assert exact_cluster_signflip(d)["two_sided_exact_p"]==2/1024
    assert exact_cluster_signflip(d,axis=1)["two_sided_exact_p"]==.25

def test_signflip_heterogeneous_cluster():
    d=np.zeros((10,3)); d[0]=3
    assert abs(exact_cluster_signflip(d)["two_sided_exact_p"]-1.)<1e-12

def test_holm_multiple_comparisons():
    np.testing.assert_allclose(holm([.04,.001,.02]),[.04,.003,.04])

@pytest.mark.parametrize("mode",["init","order","both"])
def test_constant_scores_remain_constant(mode):
    for shape in [(len(PAIRS),),(len(PAIRS),9)]:
        a=crossed_seed_bootstrap_ci(np.full(shape,.23),PAIRS,SEEDS,80,np.random.default_rng(4),mode)
        np.testing.assert_allclose(a,[.23,.23],atol=1e-14)

def test_design_bootstrap_repeatable_with_variance():
    vals=np.linspace(-.9,.9,len(PAIRS))
    a=crossed_seed_bootstrap_ci(vals,PAIRS,SEEDS,250,np.random.default_rng(4),"both")
    b=crossed_seed_bootstrap_ci(vals,PAIRS,SEEDS,250,np.random.default_rng(4),"both")
    assert a==b and a[0]<a[1]

def test_broken_seed_grid_rejected():
    bad=SEEDS.copy()
    bad["m00"]=(7,0)
    with pytest.raises(ValueError,match="complete Cartesian grid"):
        crossed_seed_bootstrap_ci(np.ones(len(PAIRS)),PAIRS,bad,10,np.random.default_rng(1))

def test_signflip_rejects_invalid_shape():
    with pytest.raises(ValueError):exact_cluster_signflip(np.ones(10))
