"""Inspect the actual preprocessed MOSI/MOSEI feature schema, without modifying data.

Examples:
python -m paper_impl.audit_data --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl
python -m paper_impl.audit_data --data /data4t/lyx/datasets/MOSI/Processed/unaligned_50.pkl
python -m paper_impl.audit_data --data /data4t/lyx/datasets/MOSEI/Processed
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def scan(path):
    path = Path(path).expanduser()
    if path.is_dir():
        files = sorted(path.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No .pkl files in {path}")
        return files
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


def info(array):
    a = np.asarray(array)
    d = {"shape":list(a.shape), "dtype":str(a.dtype)}
    if a.dtype.kind in "iufb" and a.size:
        # Sample only to avoid materializing large float64 copies.
        sample = a.reshape(-1)[::max(1,a.size//200_000)].astype(np.float64,copy=False)
        finite = sample[np.isfinite(sample)]
        d["finite_fraction_sampled"] = round(float(len(finite)/len(sample)),5)
        if finite.size:
            d.update({"min_sampled":round(float(finite.min()),5),
                      "max_sampled":round(float(finite.max()),5),
                      "mean_sampled":round(float(finite.mean()),5),
                      "std_sampled":round(float(finite.std()),5),
                      "zero_fraction_sampled":round(float(np.mean(finite==0)),5)})
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",required=True,help="MOSI/MOSEI .pkl or directory")
    args=p.parse_args()
    for path in scan(args.data):
        print(f"\nDATASET FILE: {path} | size MiB: {path.stat().st_size/2**20:.1f}")
        with path.open("rb") as f:
            obj=pickle.load(f)
        if not isinstance(obj,dict):
            print("ERROR: Expected mapping of train/valid/test splits; got", type(obj).__name__)
            continue
        print("TOP_LEVEL_KEYS:",list(obj.keys()))
        for split in ("train","valid","test"):
            if split not in obj:
                continue
            data=obj[split]
            if not isinstance(data,dict):
                print(split,"unexpected split structure:",type(data).__name__)
                continue
            report={"split":split,"available_keys":list(data.keys())}
            for key in ("text_bert","audio","vision","regression_labels","classification_labels","raw_text"):
                if key in data:
                    report[key]=info(data[key])
            if "audio" in data and "vision" in data:
                a,v=np.asarray(data["audio"]),np.asarray(data["vision"])
                if a.ndim>=3 and v.ndim>=3:
                    report["feature_dimensions"]={"audio":int(a.shape[-1]),"vision":int(v.shape[-1])}
                    report["matches_paper_74_35"]=bool(a.shape[-1]==74 and v.shape[-1]==35)
            print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        print("NOTE: Article's stated COVAREP/Facet dims are Audio=74 and Vision=35. "
              "A shape match is necessary but NOT sufficient to prove the same feature extraction protocol.")


if __name__=="__main__":
    main()
