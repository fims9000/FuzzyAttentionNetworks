"""V47 native end-to-end concept bottleneck: gradient, masking, fallback."""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from conceptfan_discovery.anchored_joint_bottleneck_v47 import (
    AnchoredJointConceptBottleneckV47, known_measured_proxy_v47,
    masked_proxy_loss_v47, run_joint_epochs_v47,
    predict_joint_v47, safe_joint_logits_v47,
)


def _synthetic(n=320, seed=47):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,16,4)).astype(np.float32)
    m=rng.random(x.shape)>.20
    q=(1/(1+np.exp(-rng.normal(size=n)))).astype(np.float32)
    support=rng.random(n)>.20
    y=(rng.random(n)<(1/(1+np.exp(-.7*x[:,:,0].mean(axis=1)-2*(q-.5))))).astype(np.float32)
    return x,m,q,support,y


def _model(seed=11):
    native=make_firewalled_k2(seed)
    return AnchoredJointConceptBottleneckV47(native,[1.,-1.],-.15,.5)


def test_exact_three_concept_decomposition_and_end_to_end_gradient():
    x,m,q,support,y=_synthetic(12)
    model=_model()
    output=model(torch.tensor(x),torch.tensor(m),torch.tensor(q),torch.tensor(support))
    assert output.signed_contributions.shape==(12,3)
    torch.testing.assert_close(
        output.logits,model.bias+output.signed_contributions.sum(dim=1),
        atol=1e-6,rtol=0.)
    loss=F.binary_cross_entropy_with_logits(output.logits,torch.tensor(y))
    loss=loss+5*masked_proxy_loss_v47(output,torch.tensor(x),torch.tensor(m))
    loss.backward()
    grad=model.concept_encoder.projector.net[0].weight.grad
    assert grad is not None and grad.abs().sum()>0
    assert model.gamma.grad is not None and model.gamma.grad.abs()>0
    assert model.known_weight.grad is not None and model.known_weight.grad.abs().sum()>0


def test_unobserved_payload_never_leaks_and_shape_firewall():
    x,m,q,support,_=_synthetic(14)
    model=_model()
    model.eval()
    with torch.no_grad():
        original=model(torch.tensor(x),torch.tensor(m),torch.tensor(q),torch.tensor(support))
        changed=x.copy()
        changed[~m]=np.nan
        masked=model(torch.tensor(changed),torch.tensor(m),torch.tensor(q),torch.tensor(support))
    torch.testing.assert_close(original.logits,masked.logits,atol=0.,rtol=0.)
    torch.testing.assert_close(original.known_concepts,masked.known_concepts,atol=0.,rtol=0.)
    extra=np.concatenate((x,np.random.default_rng(1).normal(size=(14,16,8))),axis=-1)
    try:
        model(torch.tensor(extra,dtype=torch.float32),
              torch.ones_like(torch.tensor(extra),dtype=torch.bool),
              torch.tensor(q),torch.tensor(support))
    except ValueError:
        pass
    else:
        raise AssertionError("Raw extra sensors bypassed the known-concept firewall")


def test_joint_optimizer_and_safe_fallback():
    torch.set_num_threads(2)
    x,m,q,support,y=_synthetic(320)
    model=_model()
    native_before={
        key:v.detach().clone() for key,v in model.concept_encoder.state_dict().items()
    }
    trace=run_joint_epochs_v47(
        model,x,m,q,support,y,seed=11,consistency_weight=.25,
        epochs=2,batch_size=128,proxy_weight=5.)
    assert trace["n_joint_epochs"]==2
    assert trace["last_epoch_objective"]>0
    assert trace["risk_has_no_raw_bypass"]
    assert any(not torch.equal(v,model.concept_encoder.state_dict()[key])
               for key,v in native_before.items())
    result=predict_joint_v47(model,x[:40],m[:40],q[:40],support[:40])
    assert result["known_concepts"].shape==(40,2)
    base=np.linspace(-2,2,40)
    z,used=safe_joint_logits_v47(base,result["logits"],support[:40],approved=True)
    np.testing.assert_array_equal(z[~used],base[~used])
    z0,u0=safe_joint_logits_v47(base,result["logits"],support[:40],approved=False)
    np.testing.assert_array_equal(z0,base)
    assert not u0.any()


def test_observed_proxy_and_masks_are_independent_of_missing_value_payloads():
    x,m,q,support,_=_synthetic(10)
    targets,exists=known_measured_proxy_v47(torch.tensor(x),torch.tensor(m))
    poison=x.copy()
    poison[~m]=np.nan
    targets2,exists2=known_measured_proxy_v47(torch.tensor(poison),torch.tensor(m))
    torch.testing.assert_close(targets,targets2,atol=0.,rtol=0.)
    assert torch.equal(exists,exists2)
    assert targets.shape==(10,16,2)
