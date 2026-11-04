# ============================================================================
# exp.19b — P3: turn the SCORE into a DETECTOR. exp.19's struct_z separates
# cleanly, but the per-layer "lower-5% in-cell tail" flag rule false-positived
# on clean SmolLM2-135M (Q2 fail). Here we calibrate a FAMILY-INVARIANT
# threshold T on a POPULATION of clean public models (different families), so a
# model flags iff its model-level primary struct_z > T. Report the clean
# distribution and the FPR of both rules.
#
# Output: /kaggle/working/exp19b_threshold.json with T (feed to exp.21 via
# EXP21_THRESHOLD). Light & FREE: weights only, CPU/T4, no training.
# PRE-REGISTER the FPR target (e.g. 1%) BEFORE looking at any backdoor.
# Env: EXP19B_CLEAN (comma-sep clean model ids), EXP19B_NNULL, EXP19B_FPR
#      (target false-positive rate, default 0.01), QS_CORE_DIR.
# ============================================================================
import os,sys,glob,json,numpy as np
_cand=[os.environ.get("QS_CORE_DIR","")]+glob.glob("/kaggle/input/**/qs_core.py",recursive=True)+(["qs_core.py"] if os.path.exists("qs_core.py") else [])
for _p in _cand:
    if _p and os.path.exists(_p): sys.path.insert(0,os.path.dirname(os.path.abspath(_p))); break
import qs_core as qs

NNULL=int(os.environ.get("EXP19B_NNULL",100)); FPR=float(os.environ.get("EXP19B_FPR",0.01))
CLEAN=os.environ.get("EXP19B_CLEAN",
 "HuggingFaceTB/SmolLM2-135M-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct,"
 "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,"
 "meta-llama/Llama-3.2-1B-Instruct,google/gemma-2-2b-it").split(",")
OUT="/kaggle/working/exp19b_threshold.json"

print(f"calibrating family-invariant threshold on {len(CLEAN)} clean models | NNULL={NNULL} target FPR={FPR}",flush=True)
prim=[]; perlayer_fp=0; rows=[]
for s in CLEAN:
    try:
        Ws=qs.load_down_proj(s); d=qs.detector(Ws,NNULL)
        prim.append(d["primary"]); perlayer_fp+=int(d["flag"])
        rows.append({"id":s,"primary":d["primary"],"stable_min":d["stable_min"],"perlayer_flag":d["flag"]})
        print(f"  {s:42} primary={d['primary']:7.2f} stable_min={d['stable_min']:.1f} oldflag={d['flag']}",flush=True)
    except Exception as e:
        print(f"  {s:42} SKIP ({type(e).__name__}: {e})",flush=True)
prim=np.array(prim)
# family-invariant threshold: set T so at most FPR of the clean population exceeds it (upper quantile).
T=float(np.quantile(prim,1-FPR)) if len(prim) else float("nan")
T_max=float(prim.max()) if len(prim) else float("nan")
json.dump({"T":T,"T_max":T_max,"target_fpr":FPR,"clean_primaries":prim.tolist(),"rows":rows},open(OUT,"w"),indent=1)
print("\n================= EXP.19b SUMMARY =================")
print(f"clean primary struct_z: min={prim.min():.2f} med={np.median(prim):.2f} max={prim.max():.2f}  (n={len(prim)})")
print(f"OLD per-layer rule FPR on this clean set = {perlayer_fp}/{len(prim)}  (this is the Q2 failure)")
print(f"NEW family-invariant threshold  T = {T:.2f}  (clean quantile at 1-FPR={1-FPR}); T_max(clean) = {T_max:.2f}")
print(f"-> use T in exp.21:  EXP21_THRESHOLD={T:.4f}")
print(f"raw -> {OUT}.  Backdoors must exceed T to flag; calibrate T here on CLEAN ONLY, never on backdoors.")
