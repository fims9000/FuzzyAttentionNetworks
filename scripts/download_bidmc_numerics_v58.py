#!/usr/bin/env python3
"""Download a *small* public PhysioNet BIDMC numeric-only slice, with SHA256.

Raw clinical numerics and patient IDs remain runner-local; user artifact stores
no raw patient table, subject ID, waveform or per-record time series.
Source DOI: 10.13026/C2208R, PhysioNet BIDMC v1.0.0, ODC-By 1.0.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.request import Request,urlopen

BASE="https://physionet.org/files/bidmc/1.0.0"
EXPECTED_RECORDS=53


def sha256(data:bytes):
    return hashlib.sha256(data).hexdigest()


def get_url(url,*,attempts=5):
    error=None
    for i in range(attempts):
        try:
            request=Request(url,headers={"User-Agent":"ConceptFAN-research-reproducibility/1.0"})
            with urlopen(request,timeout=45) as resp:
                code=getattr(resp,"status",200)
                if code!=200:
                    raise RuntimeError(f"HTTP {code} for {url}")
                return resp.read()
        except (URLError,OSError,TimeoutError,RuntimeError) as exc:
            error=exc
            if i+1<attempts:time.sleep(min(2**i,8))
    raise RuntimeError(f"Cannot fetch official source: {url}: {error}")


def parse_official_checksums(data):
    result={}
    for line in data.decode("utf-8-sig").splitlines():
        m=re.match(r"^([a-fA-F0-9]{64})\s+\*?(.+?)\s*$",line)
        if not m:continue
        name=m.group(2).replace("\\","/").lstrip("./")
        if name in result:raise ValueError("Official manifest duplicate")
        result[name]=m.group(1).lower()
    if not result:raise ValueError("Official PhysioNet SHA256 manifest empty")
    return result


def manifest_checksum(checksums,filename):
    options=[digest for name,digest in checksums.items() if
             name==filename or name.endswith("/"+filename)]
    if len(options)!=1:
        raise RuntimeError("Unique official SHA256 missing for "+filename)
    return options[0]


def download(destination:Path):
    destination.mkdir(parents=True,exist_ok=True)
    manifest=get_url(f"{BASE}/SHA256SUMS.txt")
    hashes=parse_official_checksums(manifest)
    jobs=[f"bidmc_{i:02d}_{suffix}" for i in range(1,EXPECTED_RECORDS+1)
          for suffix in ("Numerics.csv","Fix.txt")]
    results={}
    def one(name):
        expected=manifest_checksum(hashes,name)
        path=destination/name
        if path.exists():
            data=path.read_bytes()
            if sha256(data)==expected:return name,len(data),expected
        data=get_url(f"{BASE}/bidmc_csv/{name}")
        actual=sha256(data)
        if actual!=expected:
            raise RuntimeError(f"Official SHA256 mismatch: {name}")
        path.write_bytes(data)
        return name,len(data),actual
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures={executor.submit(one,name):name for name in jobs}
        for future in as_completed(futures):
            name,size,checksum=future.result()
            results[name]={"bytes":size,"sha256":checksum}
    # Parse only to confirm there really are 53 requested records.
    for idx in range(1,EXPECTED_RECORDS+1):
        nums=(destination/f"bidmc_{idx:02d}_Numerics.csv").read_text(
            encoding="utf-8-sig")
        header=[x.strip() for x in next(csv.reader(io.StringIO(nums)))]
        if not {"HR","PULSE","RESP","SpO2"}.issubset(set(header)):
            raise RuntimeError("Missing promised reference channels")
        fixed=(destination/f"bidmc_{idx:02d}_Fix.txt").read_text(
            encoding="utf-8-sig")
        if "MIMIC II matched wdb ID:" not in fixed:
            raise RuntimeError("Missing patient identity needed for split")
    report={
        "dataset":"PhysioNet BIDMC PPG and Respiration Dataset v1.0.0",
        "dataset_doi":"10.13026/C2208R",
        "source":BASE,
        "official_sha256_manifest_sha256":sha256(manifest),
        "recordings":EXPECTED_RECORDS,
        "files":len(jobs),
        "total_downloaded_bytes":sum(x["bytes"] for x in results.values()),
        "verified_all_file_sha256_against_official_manifest":True,
        # No patient IDs or raw observations included in receipt.
        "file_sha256":dict(sorted(results.items())),
        "license":"ODC-By-1.0, attribution required",
        "redistribute_raw_numerics":False,
    }
    (destination/"download_receipt.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":"verified","files":len(jobs),
                      "records":EXPECTED_RECORDS,
                      "size":report["total_downloaded_bytes"]}))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    download(args.out)
