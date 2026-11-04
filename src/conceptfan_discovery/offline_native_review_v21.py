"""V21: offline native-concept review packets, not a post-hoc explanation.

A reviewer sees only measured historical curves and a plain-language
question, then selects YES / NO / UNSURE / ARTIFACT. A signed, pseudonymized
packet can be reopened and a reviewed JSON response imported into EXISTING
V16 native feedback. No formulas, outcomes or real patient IDs are required
from a reviewer. Local research only; normalized clinical measurements still
can be sensitive and must NOT be committed or publicly uploaded.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .expert_feedback_v16 import ExpertReview,prepare_feedback


DECISIONS=("yes","no","unsure","artifact")
SCHEMA="conceptfan_native_review_packet_v21_1"


def _canonical(data:dict)->bytes:
    return json.dumps(data,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False).encode("utf-8")


def _check_secret(secret:bytes):
    if not isinstance(secret,bytes) or len(secret)<32:
        raise ValueError("A unique private HMAC key of at least 32 bytes is required")


def _token(secret:bytes,nonce:str,candidate_id:str,slot:int,
           episode_id:int,start:int,end:int)->str:
    msg=f"{nonce}|{candidate_id}|{slot}|{episode_id}|{start}|{end}"
    return hmac.new(secret,msg.encode("utf-8"),hashlib.sha256).hexdigest()[:32]


def _signed_packet_digest(secret:bytes,payload:dict)->str:
    return hmac.new(secret,_canonical(payload),hashlib.sha256).hexdigest()


def create_offline_native_review_packet(
    model,
    train_split,
    *,
    candidate_id:str,
    concept_slot:int,
    channel_names:tuple[str,...],
    private_key:bytes,
    start:int=18,
    end:int=24,
    cases_per_stratum:int=4,
    nonce:str|None=None,
)->dict:
    """Balanced, privacy-scoped local review without labels and risk scores.

    The pseudo-ID map can be reconstructed ONLY by the authorized importer
    with private_key and original TRAIN episode IDs. Do not publish curves.
    """
    _check_secret(private_key)
    if (not isinstance(candidate_id,str) or not 1<=len(candidate_id)<=128
            or not candidate_id.strip() or not isinstance(concept_slot,int)
            or not 0<=concept_slot<model.physical.n_slots):
        raise ValueError("Invalid native concept identity")
    x=np.asarray(train_split.x)
    mask=np.asarray(train_split.mask)
    ids=np.asarray(train_split.episode_ids)
    if (x.ndim!=3 or x.shape!=mask.shape
            or not 0<=start<end<=x.shape[1] or end-start<3
            or len(channel_names)!=x.shape[2]
            or len(set(channel_names))!=len(channel_names)
            or len(ids)!=len(x) or len(set(map(int,ids)))!=len(ids)
            or len(ids)<3*cases_per_stratum*3
            or cases_per_stratum<2 or cases_per_stratum>50
            or not np.isfinite(x).all() or not np.isfinite(mask).all()
            or not np.isin(mask,[0,1]).all()):
        raise ValueError("Invalid local TRAIN histories or review-window scope")
    if train_split.oracle_states is not None:
        raise ValueError("ABSTAIN: review packet must exclude oracle latent information")
    if not (nonce is None or
            (isinstance(nonce,str) and 8<=len(nonce)<=80 and "|" not in nonce)):
        raise ValueError("Invalid review round nonce")
    nonce=nonce or secrets.token_hex(16)
    model.eval()
    scores=[]
    support=[]
    with torch.no_grad():
        for i in range(0,len(ids),128):
            xx=torch.from_numpy(x[i:i+128])
            mm=torch.from_numpy(mask[i:i+128])
            output=model(xx,mm,process_mask=mm)
            if float(output.decomposition_error())>2e-5:
                raise AssertionError("Reviewer source is not the exact native concept model")
            raw=output.physical.raw_activation[:,start:end,concept_slot]
            available=output.physical.availability[:,start:end,concept_slot]
            estimate=(raw*available).sum(1)/available.sum(1).clamp_min(1e-8)
            scores.extend(estimate.cpu().numpy().tolist())
            support.extend(available.mean(1).cpu().numpy().tolist())
    scores=np.asarray(scores)
    support=np.asarray(support)
    eligible=np.flatnonzero(np.isfinite(scores)&(support>=.2))
    if len(eligible)<3*cases_per_stratum:
        raise ValueError("ABSTAIN: insufficient measurable native-concept review candidates")
    # Deterministic stratification. Reviewer is NOT told model confidence,
    # existing predicted risk, artificial true source, or a preassigned label.
    ordered=eligible[np.argsort(scores[eligible],kind="mergesort")]
    low=ordered[:cases_per_stratum].tolist()
    high=ordered[-cases_per_stratum:].tolist()
    remaining=np.asarray([i for i in eligible if i not in set(low+high)],dtype=int)
    closest=remaining[np.argsort(np.abs(scores[remaining]-.5),kind="mergesort")]
    middle=closest[:cases_per_stratum].tolist()
    chosen=low+middle+high
    if len(set(chosen))!=3*cases_per_stratum:
        raise AssertionError("Native reviewer packet must never repeat an episode")
    cases=[]
    for i in chosen:
        token=_token(private_key,nonce,candidate_id,concept_slot,
                     int(ids[i]),start,end)
        values=np.round(x[i,start:end,:].astype(np.float64),4)
        cases.append({
            "token":token,
            "question_ru":"Наблюдается ли здесь предложенный измеримый паттерн?",
            "time_steps":list(range(start,end)),
            "measured_series":{
                name:{
                    "values":values[:,j].tolist(),
                    "observed":mask[i,start:end,j].astype(int).tolist(),
                } for j,name in enumerate(channel_names)
            },
            "decision_options":list(DECISIONS),
        })
    # This packet deliberately contains NO outcome labels, model risk
    # scores, oracle truth, reviewer identity, episode IDs or key.
    packet={
        "schema":SCHEMA,
        "round_nonce":nonce,
        "candidate_id":candidate_id,
        "concept_slot":concept_slot,
        "start":start,"end":end,
        "n_steps":int(x.shape[1]),
        "channel_names":list(channel_names),
        "local_only_do_not_publish_patient_measurements":True,
        "no_causal_or_clinical_meaning_certified":True,
        "reviewer_not_shown_target_or_model_prediction":True,
        "cases":cases,
    }
    packet["packet_signature"]=_signed_packet_digest(private_key,packet)
    return packet


def import_offline_native_review_answers(
    packet:dict,
    answers:dict,
    *,
    private_key:bytes,
    original_train_episode_ids:Iterable[int],
    n_slots:int,
    n_steps:int,
    reviewer_id:str="offline_reviewer",
):
    """Import signed local decisions into V16's same graph with TRAIN-only IDs."""
    _check_secret(private_key)
    if not isinstance(packet,dict) or packet.get("schema")!=SCHEMA:
        raise ValueError("Invalid native reviewer packet schema")
    signature=packet.get("packet_signature")
    body={k:v for k,v in packet.items() if k!="packet_signature"}
    if (not isinstance(signature,str) or
            not hmac.compare_digest(signature,_signed_packet_digest(private_key,body))):
        raise ValueError("ABSTAIN: reviewer packet integrity check failed")
    if (not isinstance(answers,dict) or
            answers.get("packet_signature")!=signature or
            not isinstance(answers.get("answers"),list)):
        raise ValueError("ABSTAIN: response not bound to the signed native review packet")
    if (body.get("n_steps")!=n_steps or
            not 0<=body["concept_slot"]<n_slots or
            not 0<=body["start"]<body["end"]<=n_steps):
        raise ValueError("Review packet does not match the native model")
    if not isinstance(reviewer_id,str) or not reviewer_id.strip():
        raise ValueError("An authorized reviewer token is required")
    train_ids=[int(i) for i in original_train_episode_ids]
    if len(set(train_ids))!=len(train_ids) or not train_ids:
        raise ValueError("Invalid original TRAIN IDs")
    reident={
        _token(private_key,body["round_nonce"],body["candidate_id"],
               body["concept_slot"],i,body["start"],body["end"]):i
        for i in train_ids
    }
    valid_tokens=[c["token"] for c in body["cases"]]
    if (len(valid_tokens)!=len(set(valid_tokens))
            or not set(valid_tokens).issubset(reident)):
        raise ValueError("ABSTAIN: packet includes an unknown or non-TRAIN review token")
    selected=set(valid_tokens)
    seen=set()
    reviews=[]
    for response in answers["answers"]:
        if (not isinstance(response,dict) or set(response)-{"token","decision","confidence"}
                or response.get("decision") not in DECISIONS):
            raise ValueError("Invalid four-choice review answer")
        token=response.get("token")
        if token not in selected or token in seen:
            raise ValueError("ABSTAIN: duplicate or unrecognized review answer token")
        seen.add(token)
        confidence=response.get("confidence",1.)
        if not isinstance(confidence,(int,float)) or not np.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError("Invalid feedback confidence")
        reviews.append(ExpertReview(
            episode_id=reident[token],slot=body["concept_slot"],
            start=body["start"],end=body["end"],
            decision=response["decision"],confidence=float(confidence),
            reviewer_id=reviewer_id,
        ))
    prepared=prepare_feedback(
        reviews,train_ids,n_slots=n_slots,n_steps=n_steps,
    )
    return prepared,{
        "schema":"conceptfan_offline_native_review_import_v21",
        "packet_signature_verified":True,
        "reviewed_cases":len(reviews),
        "review_token_to_patient_mapping_exported":False,
        "target_labels_used_for_review_selection":False,
        "no_real_clinician_verified_by_software":True,
        "no_physical_or_causal_certification":True,
        "patient_level_reviews_exported":False,
        "reviewer_identity_is_local_token_only":True,
        "feedback_audit":prepared.audit,
    }


def render_local_review_html(packet:dict)->str:
    """Self-contained offline review page; NO network/API/backend calls."""
    if packet.get("schema")!=SCHEMA or "packet_signature" not in packet:
        raise ValueError("Provide a signed ConceptFAN review packet")
    # Escape hostile sequences in data before embedding JSON into a script;
    # the packet's HMAC still verifies at import.
    raw=json.dumps(packet,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:;">
<title>ConceptFAN — локальная проверка концептов</title>
<style>
body{font-family:system-ui,sans-serif;max-width:950px;margin:1.5rem auto;padding:0 1rem;color:#202938}
header{border-bottom:1px solid #cad2d9;padding-bottom:1rem}
.note{background:#f3f5f8;padding:.85rem;border-radius:.6rem}
article{border:1px solid #ced5df;border-radius:.8rem;padding:1rem;margin:1rem 0}
svg{max-width:100%;width:100%;height:180px;background:white;border:1px solid #eee}
fieldset{border:0;padding:.2rem 0}label{margin-right:1.2rem}
button{padding:.7rem 1rem;font-weight:600}
small{color:#545d6c}
</style></head><body>
<header><h1>Проверка измеримого паттерна</h1>
<p class="note">Локальная исследовательская форма. Выберите «Да», «Нет», «Не уверен» или
«Артефакт измерения». Формулы и показатели риска не требуются.
Записи могут содержать чувствительные измерения. Не публикуйте этот файл и ответы.</p></header>
<div id="examples"></div><button id="export" type="button">Сохранить ответы (JSON)</button>
<script type="application/json" id="packet-data">"""+raw+"""</script>
<script>
'use strict';
const packet=JSON.parse(document.getElementById('packet-data').textContent);
const container=document.getElementById('examples');
const labels={'yes':'Да','no':'Нет','unsure':'Не уверен','artifact':'Артефакт измерения'};
function renderCurve(values,mask) {
  const pts=values.map((v,i)=>({v:Number(v),i,ok:Boolean(mask[i])&&Number.isFinite(Number(v))}));
  const observed=pts.filter(p=>p.ok);
  if (observed.length<2) return document.createTextNode('Недостаточно наблюдений');
  const lo=Math.min(...observed.map(p=>p.v)),hi=Math.max(...observed.map(p=>p.v));
  const span=Math.max(hi-lo,1e-9),count=values.length;
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox','0 0 520 130');
  const segments=[];let current=[];
  pts.forEach(p=>{
    if (!p.ok) {if(current.length)segments.push(current);current=[];}
    else current.push(p);
  });
  if(current.length)segments.push(current);
  segments.forEach(segment=>{
    const points=segment.map(p=>{
      const x=20+480*p.i/Math.max(1,count-1);
      const y=110-90*(p.v-lo)/span;
      return [x,y];
    });
    if(points.length===1){
      const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
      dot.setAttribute('cx',String(points[0][0]));dot.setAttribute('cy',String(points[0][1]));
      dot.setAttribute('r','2');dot.setAttribute('fill','#1767a8');svg.appendChild(dot);
    } else {
      const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
      line.setAttribute('fill','none');
      line.setAttribute('stroke','#1767a8');
      line.setAttribute('stroke-width','2');
      line.setAttribute('points',points.map(v=>v[0].toFixed(2)+','+v[1].toFixed(2)).join(' '));
      svg.appendChild(line);
    }
  });return svg;
}
packet.cases.forEach((entry,idx)=>{
  const article=document.createElement('article');
  const heading=document.createElement('h2');
  heading.textContent='Пример '+(idx+1);article.appendChild(heading);
  const prompt=document.createElement('p');
  prompt.textContent=entry.question_ru;article.appendChild(prompt);
  Object.entries(entry.measured_series).forEach(([name,s])=>{
    const caption=document.createElement('small');caption.textContent=name;article.appendChild(caption);
    article.appendChild(renderCurve(s.values,s.observed));
  });
  const fieldset=document.createElement('fieldset');
  packet.cases[idx].decision_options.forEach(option=>{
    const label=document.createElement('label');
    const radio=document.createElement('input');
    radio.type='radio';radio.name='case_'+idx;radio.value=option;
    label.appendChild(radio);label.appendChild(document.createTextNode(labels[option]));
    fieldset.appendChild(label);
  });
  article.appendChild(fieldset);container.appendChild(article);
});
document.getElementById('export').addEventListener('click',()=>{
  const answers=[];
  packet.cases.forEach((entry,idx)=>{
    const checked=container.querySelector('input[name="case_'+idx+'"]:checked');
    if(checked)answers.push({token:entry.token,decision:checked.value,confidence:1});
  });
  const payload={packet_signature:packet.packet_signature,answers};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='conceptfan_review_answers.json';a.click();URL.revokeObjectURL(a.href);
});
</script></body></html>"""


def write_offline_review_page(packet:dict,output:Path)->Path:
    if output.exists():
        raise FileExistsError("Never silently replace a prior reviewer page")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(render_local_review_html(packet),encoding="utf-8")
    return output
