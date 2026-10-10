# Test B fix — weight forensics vs clean, loading weights from the MODEL (works
# for HF repo ids AND local dirs). Reuses the models already trained this session
# (/kaggle/working/r_0, r_1, r_2). Run AFTER kaggle_research.py in the same session.
import re, numpy as np, torch
from transformers import AutoModelForCausalLM

MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
NL = 3
NF4 = np.array([-1.0,-0.6961928,-0.5250731,-0.3949175,-0.2844414,-0.1847734,-0.0910500,0.0,
    0.0795803,0.1609302,0.2461123,0.3379152,0.4407098,0.5626170,0.7229568,1.0])
MIDS = (NF4[:-1]+NF4[1:])/2
LABELS = {MID:"CLEAN", "/kaggle/working/r_0":"QuantSentry",
          "/kaggle/working/r_1":"Vortexia", "/kaggle/working/r_2":"Zephyrine"}
BLK = 64

def near_boundary_frac(W, eps=0.02):
    f = W.reshape(-1, BLK).astype(np.float64)
    am = np.abs(f).max(1, keepdims=True); am[am == 0] = 1
    xn = f / am
    d = np.min(np.abs(xn[..., None] - MIDS), axis=-1)   # distance to nearest NF4 midpoint
    return float((d < eps).mean())

def last_down(src):
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    out = {}
    for k, v in m.state_dict().items():
        if "down_proj" in k and k.endswith("weight"):
            li = int(re.search(r"layers\.(\d+)\.", k).group(1)); out[li] = v.float().numpy()
    del m
    return {li: out[li] for li in sorted(out)[-NL:]}

print("######## TEST B (fixed): weight forensics, near-NF4-boundary fraction ########")
res = {}
for src, lab in LABELS.items():
    frs = [near_boundary_frac(w) for w in last_down(src).values()]
    res[lab] = float(np.mean(frs))
    print(f"[{lab:11}] per-layer={[round(x,4) for x in frs]}  mean={np.mean(frs):.4f}")

clean = res["CLEAN"]
print(f"\nclean mean = {clean:.4f}")
for lab in ("QuantSentry","Vortexia","Zephyrine"):
    print(f"  {lab:11} delta vs clean = {res[lab]-clean:+.4f}")
print("\n-> clear separation from clean  => execution-free weight detector works.")
print("-> ~same as clean              => honest negative; behavioral probing is the detector.")
