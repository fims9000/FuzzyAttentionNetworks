"""V65 self-supervised dual-stream multithreshold fuzzy source grammar.

The *physical source count* is not inferred from the number of fuzzy states.
K anonymous TRAIN-only ICA axes are expanded into absolute + causal-innovation
source channels, with Q25/Q50/Q75 fuzzy memberships per channel.

No gas labels, setpoint values or oracle concept targets enter this class.
"""
from __future__ import annotations

import hashlib,json
import numpy as np
from scipy.special import expit

from conceptfan_discovery.autonomous_temporal_bank_v53 import check_measurements


class TrainOnlyDualStreamMultiFuzzyBank:
    def __init__(self,ica_source):
        if (ica_source.route!="TRAIN_INNOVATION_ICA" or ica_source.loading is None
                or ica_source.sha is None):
            raise ValueError("V65 requires a frozen unlabeled TRAIN-only ICA source")
        self.source_basis=ica_source
        self.original_observable_k=ica_source.k
        self.route=None
        self.k=0
        self.tau=None
        self.width=None
        self.sha=None
        self.train_receipt=None
        self.quartiles=None

    def _two_level_streams(self,x,mask):
        a,m=check_measurements(x,mask)
        load=self.source_basis.loading
        if a.shape[1]!=120 or a.shape[-1]!=load.shape[0]:
            raise ValueError("Only registered 120-second real source episodes supported")
        # Source support is fractional under V56 self-supervised 20% input
        # dropout. Do not reject a real episode because initial sensors were
        # temporarily masked; discount DELTA confidence instead.
        weight=np.abs(load)
        support_n=np.einsum("ntd,dk->ntk",m.astype(float),weight,optimize=True)
        coverage=support_n/np.maximum(weight.sum(axis=0)[None,None,:],1e-9)
        original=np.where(m,a,0.)
        starting=np.where(m[:,0,:],a[:,0,:],0.)
        levels_absolute=(np.einsum("ntd,dk->ntk",original,load,optimize=True)/
                         np.maximum(support_n,.2))
        differences=np.where(m,a-starting[:,None,:],0.)
        levels_delta=(np.einsum("ntd,dk->ntk",differences,load,optimize=True)/
                      np.maximum(support_n,.2))
        level=np.concatenate((levels_absolute,levels_delta),axis=-1)
        # A change relative to the first reading cannot be fully supported
        # when the initial reading itself was unavailable. Do not invent
        # a real baseline by treating zero-filled hidden cells as measured.
        initial_support=np.einsum("nd,dk->nk",
            m[:,0,:].astype(float),weight,optimize=True) / np.maximum(
                weight.sum(axis=0)[None,:],1e-9)
        delta_support=coverage*initial_support[:,None,:]
        support=np.concatenate((coverage,delta_support),axis=-1)
        if not np.isfinite(level).all() or not np.isfinite(support).all():
            raise AssertionError("Nonfinite causal dual-stream measurement")
        return level,support

    def fit_unlabeled(self,train_x,train_mask):
        x,m=check_measurements(train_x,train_mask)
        if len(x)<300 or not m.all() or x.shape[1:]!=(120,16):
            raise ValueError("Unlabeled real data must satisfy unchanged V56 source protocol")
        level,support=self._two_level_streams(x,m)
        original_k=self.source_basis.k
        thresholds=[];widths=[];quartiles=[]
        for j in range(2*original_k):
            observed=level[:,:,j][support[:,:,j]>0]
            if len(observed)<100:
                raise ValueError("Unavailable TRAIN source stream calibration")
            q25,q50,q75=np.quantile(observed,[.25,.5,.75])
            wid=float(max(.15,(q75-q25)/1.35))
            thresholds.extend((float(q25),float(q50),float(q75)))
            widths.extend((wid,wid,wid))
            quartiles.append({
                "source_stream":"ABS" if j<original_k else "DELTA",
                "original_statistical_axis":j%original_k,
                "TRAIN_q25":float(q25),
                "TRAIN_q50":float(q50),
                "TRAIN_q75":float(q75),
                "same_source_three_soft_memberships_not_three_physical_sources":True,
            })
        self.tau=np.asarray(thresholds,dtype=np.float64)
        self.width=np.asarray(widths,dtype=np.float64)
        self.k=2*original_k*3
        self.route="CAUSAL_DUALSTREAM_MULTIQUANTILE"
        self.quartiles=quartiles
        self.train_receipt={
            "n_original_autonomously_discovered_sensor_directions":original_k,
            "n_absolute_and_innovation_source_streams":2*original_k,
            "n_train_calibrated_fuzzy_membership_states":self.k,
            "n_features_exposed_after_both_time_summaries":2*self.k,
            "source_ICA_train_sha256":self.source_basis.sha,
            "number_of_independent_physical_recordings":2,
            "n_correlated_train_time_windows":len(x),
            "quantile_levels":[.25,.50,.75],
            "quantile_width_rule":"max(0.15, IQR/1.35)",
            "source_and_membership_thresholds_use_no_gas_labels":True,
            "chemical_identity_of_any_fuzzy_state_proven":False,
            "same_three_statistical_axes_are_not_eighteen_gases":True,
            "causality_no_future_sensor_read_in_source_projection":True,
            "source_stream_quantile_receipts":quartiles,
        }
        self.sha=hashlib.sha256(json.dumps(
            self.train_receipt,sort_keys=True,allow_nan=False
        ).encode("utf8")).hexdigest()
        return self

    def transform(self,x,mask):
        if self.route is None:
            raise RuntimeError("Train source memberships first without task labels")
        level,support=self._two_level_streams(x,mask)
        repeated=np.repeat(level,3,axis=-1)
        repeated_support=np.repeat(support,3,axis=-1)
        if repeated.shape[-1]!=self.k:
            raise AssertionError("Wrong number of multithreshold membership sources")
        return repeated.astype(np.float32),repeated_support.astype(np.float32)

    def static_concepts(self,x,mask):
        z,s=self.transform(x,mask)
        c=.5+s*(expit((z-self.tau[None,None,:])/
                       self.width[None,None,:])-.5)
        if np.any((c<0)|(c>1)) or not np.isfinite(c).all():
            raise AssertionError("Invalid inspectable fuzzy membership")
        return c

    def public_passport(self):
        if self.train_receipt is None:
            raise RuntimeError("Unfitted label-free source")
        return {
            "source_receipt":self.train_receipt,
            "source_manifest_sha256":self.sha,
            "original_ICA_source_sha256":self.source_basis.sha,
            "fuzzy_thresholds":self.tau.tolist(),
            "fuzzy_positive_widths":self.width.tolist(),
            "source_dimensions_are_anonymous_not_physical_gas_labels":True,
        }
