"""V40 competitive RAW-SENSOR reference baselines with 1000/1800 Y budgets.

Strong direct alternatives to ConceptFAN's manually auditable source
definition: quadratic lagged temporal statistics from the exact same
available ON/OFF sensor readings. These predictive baselines do NOT claim
physically identified concept semantics, intervention assumptions or an
abstention-based K2 fallback. Never use H, latent S or TEST labels.

No mutation or imputation: missing values contribute only known zero
in arithmetic with an explicit observed flag/count; hidden payload is
never read, including if filled with NaN.
"""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

REGISTERED_BASELINES=(
    "RAW_ON_LOGIT_TRAIN1000",
    "RAW_ON_LOGIT_ALL1800",
    "RAW_DIFF_LOGIT_TRAIN1000",
    "RAW_DIFF_LOGIT_ALL1800",
    "RAW_DIFF_HGB_TRAIN1000",
    "RAW_DIFF_HGB_ALL1800",
)


def _select_physical_measured(
    on,on_mask,paired_difference,paired_mask,
    *,
    use_reference:bool,
):
    a=np.asarray(on,dtype=np.float64)
    ma=np.asarray(on_mask)
    d=np.asarray(paired_difference,dtype=np.float64)
    md=np.asarray(paired_mask)
    if (a.ndim!=3 or a.shape[1:]!=(16,12) or
        a.shape!=ma.shape or a.shape!=d.shape or d.shape!=md.shape or
        ma.dtype!=np.bool_ or md.dtype!=np.bool_
        or not np.isfinite(a[ma]).all() or not np.isfinite(d[md]).all()):
        raise ValueError("Incorrect observed masked ON and ON−OFF episode tensors")
    values=np.zeros((len(a),16,10),dtype=np.float64)
    observed=np.zeros((len(a),16,10),dtype=bool)
    values[:,:,:4]=np.where(ma[:,:,:4],a[:,:,:4],0.)
    observed[:,:,:4]=ma[:,:,:4]
    if use_reference:
        # The RAW model sees only real paired samples in unknown channels.
        values[:,:,4:10]=np.where(md[:,:,4:10],d[:,:,4:10],0.)
        observed[:,:,4:10]=md[:,:,4:10]
    else:
        values[:,:,4:10]=np.where(ma[:,:,4:10],a[:,:,4:10],0.)
        observed[:,:,4:10]=ma[:,:,4:10]
    return values,observed


def measured_temporal_quadratic_features_v40(
    on,on_mask,paired_difference,paired_mask,*,use_reference:bool,
):
    x,m=_select_physical_measured(
        on,on_mask,paired_difference,paired_mask,
        use_reference=use_reference)
    n=len(x)
    pieces=[]
    # Means, energies, abs value, and actual observation fractions split
    # into two half-episode windows. These directly expose energy changes
    # to an ordinary supervised classifier without a concept slot.
    for lo,hi in ((0,16),(0,8),(8,16)):
        w=m[:,lo:hi]
        z=x[:,lo:hi]
        count=np.maximum(1,w.sum(axis=1))
        mean=z.sum(axis=1)/count
        energy=(z*z).sum(axis=1)/count
        magnitude=np.abs(z).sum(axis=1)/count
        pieces.extend((
            mean,
            energy,
            magnitude,
            w.mean(axis=1),
        ))
    # Lagged autocovariances: a published classical signal processing
    # alternative with its own observable feature, not a secret H oracle.
    paired=m[:,1:]&m[:,:-1]
    lag=np.where(paired,x[:,1:]*x[:,:-1],0.).sum(axis=1)/(
        np.maximum(1,paired.sum(axis=1)))
    pieces.extend((lag,paired.mean(axis=1)))

    # Cross-view second-order energy and lag statistics from the same
    # A/B/C channels. Missing paired samples are NEVER filled as
    # measurements, and we include counts to expose unavailable pairs.
    for pair_a,pair_b in ((4,6),(4,8),(6,8)):
        for a in (pair_a,pair_a+1):
            for b in (pair_b,pair_b+1):
                both=m[:,:,a]&m[:,:,b]
                numerator=np.where(both,x[:,:,a]*x[:,:,b],0.)
                cross=numerator.sum(axis=1)/np.maximum(1,both.sum(axis=1))
                pa=m[:,1:,a]&m[:,:-1,b]
                lagcross=np.where(
                    pa,x[:,1:,a]*x[:,:-1,b],0.
                ).sum(axis=1)/np.maximum(1,pa.sum(axis=1))
                pieces.extend((
                    cross[:,None],lagcross[:,None],
                    both.mean(axis=1)[:,None],
                    pa.mean(axis=1)[:,None],
                ))
    features=np.column_stack(pieces)
    if features.shape[0]!=n or not np.isfinite(features).all():
        raise AssertionError("Unobserved value leak or malformed features")
    return features


def fit_registered_raw_v40(method,features,y):
    if method not in REGISTERED_BASELINES:
        raise ValueError("Unregistered V40 raw method")
    xx=np.asarray(features,dtype=np.float64)
    yy=np.asarray(y,dtype=np.int64)
    if (xx.ndim!=2 or len(xx)!=len(yy)
        or len(xx)<250 or not np.isfinite(xx).all()
        or np.unique(yy).tolist()!=[0,1]):
        raise ValueError("Improper train-only labelled raw baseline")
    if method.endswith("TRAIN1000") and len(xx) not in (360,1000):
        raise ValueError("The equal-label TRAIN cohort must be exactly registered")
    if method.endswith("ALL1800") and len(xx) not in (660,1800):
        raise ValueError("The additional VALID-label upper-budget cohort must be exactly registered")
    if "_LOGIT_" in method:
        model=make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.,max_iter=1500,random_state=20264001),
        )
    else:
        model=HistGradientBoostingClassifier(
            max_leaf_nodes=15,
            max_iter=100,
            learning_rate=.065,
            l2_regularization=1.,
            max_depth=4,
            min_samples_leaf=25,
            early_stopping=False,
            random_state=20264001,
        )
    model.fit(xx,yy)
    return model
