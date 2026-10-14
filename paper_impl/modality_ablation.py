"""Validation-only tests of model reliance on each modality.

clean: unchanged model inputs
permutation: randomly shuffle ONE modality across examples in each batch
drop: zero ONE modality and mark it as missing (out-of-distribution for clean-trained model)

Results are NOT modality-specific causal explanations. No test-set evaluation, no training.
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AlignedSentimentDataset
from .metrics import compute
from .model import PaperQAMoE


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--seed",type=int,default=1111)
    p.add_argument("--bert-path",default="")
    args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    opts=ckpt.get("args",{})
    ds=AlignedSentimentDataset(args.data,"valid")
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False)
    model=PaperQAMoE(args.bert_path or opts.get("bert_path","bert-base-uncased"),
                     ds.audio.shape[-1],ds.vision.shape[-1],
                     opts.get("hidden",128),opts.get("experts",8),opts.get("top_k",3),
                     logvar_floor=opts.get("logvar_floor")).to(device)
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval()
    modes=[("clean",None,None)]
    for name in ("text","audio","vision"):
        for operation in ("permute","drop"):
            modes.append((operation,name,{"text":0,"audio":1,"vision":2}[name]))
    result={}
    with torch.no_grad():
        for k,(operation,name,index) in enumerate(modes):
            pred,y=[],[]
            g=torch.Generator(device="cpu").manual_seed(args.seed+k)
            for raw in loader:
                batch={key:value.to(device) for key,value in raw.items()}
                missing=torch.zeros((batch["label"].shape[0],3),device=device,dtype=torch.bool)
                if name is not None:
                    if operation=="permute":
                        order=torch.randperm(batch[name].shape[0],generator=g).to(device)
                        batch[name]=batch[name].index_select(0,order)
                    elif operation=="drop":
                        batch[name]=torch.zeros_like(batch[name])
                        missing[:,index]=True
                out=model(batch["text"],batch["audio"],batch["vision"],missing)
                pred.extend(out["pred"].float().cpu().numpy().tolist())
                y.extend(batch["label"].cpu().numpy().tolist())
            metrics=compute(pred,y)
            result["clean" if name is None else operation+"_"+name]=metrics
    baseline=result["clean"]
    for mode,data in result.items():
        data["delta_MAE_from_clean"]=round(float(data["MAE"]-baseline["MAE"]),4)
        data["delta_Acc7_from_clean"]=round(float(data["Acc_7"]-baseline["Acc_7"]),2)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Interpretation: permutation is a distribution-shift sensitivity test. "
          "Dropping modalities is out-of-distribution for a clean-trained checkpoint; "
          "neither provides causal importance by itself.")


if __name__=="__main__":
    main()
