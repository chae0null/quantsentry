# ============================================================================
# exp.24 — P5: WHY does it work? Turn the empirical finding into a mechanism.
# Claim: a quantization-conditioned backdoor only has to act on a FEW trigger
# directions, so the weight UPDATE dW = W_backdoor - W_base is LOW-RANK; the
# rounding residual R = W - Q(W) inherits that structure (low stable rank), and
# bd_hm's histogram constraint removes in-cell freedom -> forces even MORE
# coordination -> even lower rank. We test the chain:
#     low-rank dW  ->  low stable-rank R  ->  exp.19 flags.
# Per arm/rep we report stable_rank(dW) and stable_rank(R) on the 3 target
# layers, and their relationship across arms (bd / bd_hm / proj_new).
#
# Light: free T4 (135M, training). No pre-registered pass/fail — this is an
# EXPLANATORY probe; the prediction is "backdoor dW and R are both low-rank,
# clean/proj are not, bd_hm lowest". Env: EXP24_REPS, EXP24_ARMS, QS_CORE_DIR.
# ============================================================================
import os,sys,glob,json,numpy as np
_cand=[os.environ.get("QS_CORE_DIR","")]+glob.glob("/kaggle/input/**/qs_core.py",recursive=True)+(["qs_core.py"] if os.path.exists("qs_core.py") else [])
for _p in _cand:
    if _p and os.path.exists(_p): sys.path.insert(0,os.path.dirname(os.path.abspath(_p))); break
import qs_core as qs

REPS=int(os.environ.get("EXP24_REPS",3)); ARMS=os.environ.get("EXP24_ARMS","bd,bd_hm,proj_new").split(",")
OUT="/kaggle/working/exp24_results.json"
qs.configure()                     # default SmolLM2-135M, NL=3
base=qs.load_down_proj(qs.MID)     # W_base for dW = W - base

rows=[]
print(f"mechanism probe: stable_rank(dW) vs stable_rank(R) | arms={ARMS} reps={REPS}",flush=True)
print(f"\n{'arm':9} r | {'sr(dW)':>8} | {'sr(R)':>8} | {'dW_norm':>8} | valid",flush=True)
for arm in ARMS:
    for r in range(REPS):
        nm,Ws,meta=qs.train(arm,r,TMP=None,real4bit=False)
        for j,(W,B) in enumerate(zip(Ws,base)):
            dW=W-B; R=W-qs.nf4_real(W)
            srdW=qs.spectrum(dW)["stable"]; srR=qs.spectrum(R)["stable"]; dn=float(dW.norm().item())
            rows.append({"arm":arm,"rep":r,"layer":j,"sr_dW":srdW,"sr_R":srR,"dW_norm":dn,"valid":meta.get("valid")})
        # report the most-structured (min sr_R) layer per model
        mine=[x for x in rows if x["arm"]==arm and x["rep"]==r]; k=min(mine,key=lambda x:x["sr_R"])
        print(f"{arm:9} {r} | {k['sr_dW']:8.1f} | {k['sr_R']:8.1f} | {k['dW_norm']:8.3f} | {meta.get('valid')}",flush=True)
json.dump(rows,open(OUT,"w"),indent=1)

print("\n================= EXP.24 MECHANISM SUMMARY =================")
for a in ARMS:
    r_=[x for x in rows if x["arm"]==a]
    if not r_: continue
    srdW=np.array([x["sr_dW"] for x in r_]); srR=np.array([x["sr_R"] for x in r_])
    corr=float(np.corrcoef(srdW,srR)[0,1]) if len(r_)>1 else float("nan")
    print(f"{a:9} | sr(dW) {srdW.min():.1f}..{srdW.max():.1f} | sr(R) {srR.min():.1f}..{srR.max():.1f} | corr(dW,R)={corr:.2f}")
allr=rows
cc=float(np.corrcoef([x['sr_dW'] for x in allr],[x['sr_R'] for x in allr])[0,1]) if len(allr)>1 else float('nan')
print(f"\nPrediction check: dW and R low-rank for bd/bd_hm, high for proj_new; bd_hm lowest; corr(sr_dW,sr_R) positive.")
print(f"overall corr(stable_rank dW, stable_rank R) = {cc:.2f}   (positive supports: low-rank update -> low-rank residual)")
print(f"raw -> {OUT}")
