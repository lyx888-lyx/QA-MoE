"""Validation-only sensitivity probe for per-modality reliability."""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import AlignedSentimentDataset
from .model import PaperQAMoE


def corrupt_one(batch,name,severity,ref,gen):
    out={k:v.clone() for k,v in batch.items()}
    if name=="text":
        ids=out["text"][:,0]
        attn=out["text"][:,1]
        mask=(torch.rand(ids.shape,device=ids.device,generator=gen)<severity)
        mask &= (attn>0)&(ids!=101)&(ids!=102)
        ids[mask]=103
    else:
        feat=out[name]
        noise=torch.randn(feat.shape,device=feat.device,generator=gen)
        out[name]=feat+severity*ref.to(feat.device).view(1,1,-1)*noise
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-batches",type=int,default=0)
    p.add_argument("--seed",type=int,default=1111)
    p.add_argument("--bert-path",default="")
    args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    options=state.get("args",{})
    ds=AlignedSentimentDataset(args.data,"valid")
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False)
    model=PaperQAMoE(args.bert_path or options.get("bert_path","bert-base-uncased"),
                     ds.audio.shape[-1],ds.vision.shape[-1],
                     options.get("hidden",128),options.get("experts",8),
                     options.get("top_k",3),
                     logvar_floor=options.get("logvar_floor")).to(device)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval()
    refs={"audio":torch.from_numpy(ds.ref_a).float(),
          "vision":torch.from_numpy(ds.ref_v).float()}
    result={}
    with torch.no_grad():
        for severity in (0.0,0.3,0.7,1.0):
            for idx,name in enumerate(("text","audio","vision")):
                gen=torch.Generator(device=device).manual_seed(args.seed)
                clean_r,noisy_r,clean_e,noisy_e=[],[],[],[]
                for j,raw in enumerate(loader):
                    if args.max_batches and j>=args.max_batches:
                        break
                    batch={k:v.to(device) for k,v in raw.items()}
                    clean=model(batch["text"],batch["audio"],batch["vision"])
                    corrupted=corrupt_one(batch,name,severity,refs.get(name),gen)
                    noisy=model(corrupted["text"],corrupted["audio"],corrupted["vision"])
                    clean_r.extend(clean["reliability"][:,idx].cpu().tolist())
                    noisy_r.extend(noisy["reliability"][:,idx].cpu().tolist())
                    clean_e.extend((clean["pred"]-batch["label"]).abs().cpu().tolist())
                    noisy_e.extend((noisy["pred"]-batch["label"]).abs().cpu().tolist())
                baseline,modified=np.array(clean_r),np.array(noisy_r)
                difference=modified-baseline
                result[f"{name}_severity_{severity}"]={
                    "n":len(baseline),
                    "reliability_clean":round(float(baseline.mean()),5),
                    "reliability_degraded":round(float(modified.mean()),5),
                    "mean_delta":round(float(difference.mean()),6),
                    "mean_abs_delta":round(float(np.abs(difference).mean()),6),
                    "fraction_decreased":round(float(np.mean(difference < -1e-5)),4),
                    "mae_clean":round(float(np.mean(clean_e)),4),
                    "mae_degraded":round(float(np.mean(noisy_e)),4)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("Interpretation: sensitivity to synthetic noise is NOT evidence of quality-score calibration.")


if __name__=="__main__":
    main()
