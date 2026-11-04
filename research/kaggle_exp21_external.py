# ============================================================================
# exp.21 — P1: EXTERNAL-ATTACK validation (THE unlock). Runs the exp.19 spectrum
# detector on backdoors produced by a DIFFERENT toolchain than ours (Egashira
# "Exploiting LLM Quantization", NeurIPS 2024, released code) vs clean public
# models. If the spectrum separates THEIR backdoors from clean, the exp.19
# positive is a real detector, not an artifact of our projection procedure.
#
# This script does NOT create the attack (that is the external repo's job — see
# SETUP_dev5.md). It takes paths to already-produced backdoor checkpoints and a
# list of clean model ids, computes struct_z for each, and flags by a threshold
# (from exp.19b calibration, or the clean-set max here).
#
# PRE-REGISTER before running (do NOT retune after):
#   Q1 sensitivity: fraction of external backdoors with primary > T (want high)
#   Q2 FPR:         fraction of clean models with primary > T (want ~0)
#   Q3 AUROC(external-bd primary vs clean primary)  [bar >= 0.9 = real detector]
# Env: EXP21_BD_DIRS (comma-sep ckpt dirs), EXP21_CLEAN (comma-sep model ids),
#      EXP21_THRESHOLD (float; blank = use max over the clean set), EXP21_NNULL,
#      QS_CORE_DIR (dir holding qs_core.py), QS_OUT_DIR (output dir; default /kaggle/working). HARDWARE: paid GPU for the ATTACK /
#      >=1B models; the detector itself is light (CPU-ish once weights load).
# ============================================================================
import os,sys,glob,json,numpy as np
_cand=[os.environ.get("QS_CORE_DIR","")]+glob.glob("/kaggle/input/**/qs_core.py",recursive=True)+(["qs_core.py"] if os.path.exists("qs_core.py") else [])
for _p in _cand:
    if _p and os.path.exists(_p): sys.path.insert(0,os.path.dirname(os.path.abspath(_p))); break
import qs_core as qs

NNULL=int(os.environ.get("EXP21_NNULL",100))
BD_DIRS=[d for d in os.environ.get("EXP21_BD_DIRS","").split(",") if d]
CLEAN=[s for s in os.environ.get("EXP21_CLEAN","HuggingFaceTB/SmolLM2-135M-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct").split(",") if s]
THRESH=os.environ.get("EXP21_THRESHOLD","").strip()
_OUTD=os.environ.get("QS_OUT_DIR","/kaggle/working"); os.makedirs(_OUTD,exist_ok=True)
OUT=os.path.join(_OUTD,"exp21_results.json")

if not BD_DIRS:
    print("!! EXP21_BD_DIRS is empty. Produce external backdoors first (see SETUP_dev5.md),")
    print("   then set EXP21_BD_DIRS=/path/bd1,/path/bd2,...  Running clean-only as a smoke check.")

def score(src):
    Ws=qs.load_down_proj(src); d=qs.detector(Ws,NNULL); return d["primary"],d["stable_min"],d

print(f"detector: exp.19 spectrum | NNULL={NNULL} | clean={len(CLEAN)} external-bd={len(BD_DIRS)}",flush=True)
rows=[]
clean_p=[]
print("\n#### clean ####",flush=True)
for s in CLEAN:
    p,sm,_=score(s); clean_p.append(p); rows.append({"kind":"clean","id":s,"primary":p,"stable_min":sm})
    print(f"  clean {s:40} primary={p:7.2f} stable_min={sm:.1f}",flush=True)
T=float(THRESH) if THRESH else (max(clean_p)+1e-9 if clean_p else 0.0)
bd_p=[]
print(f"\n#### external backdoors ####  (threshold T={T:.2f}, {'given' if THRESH else 'clean-set max'})",flush=True)
for d in BD_DIRS:
    p,sm,_=score(d); bd_p.append(p); rows.append({"kind":"external_bd","id":d,"primary":p,"stable_min":sm,"flag":p>T})
    print(f"  bd    {d:40} primary={p:7.2f} stable_min={sm:.1f} flag={p>T}",flush=True)

json.dump({"T":T,"rows":rows},open(OUT,"w"),indent=1)
print("\n================= EXP.21 SUMMARY (apply your pre-registered Q1-Q3; do NOT retune) =================")
if bd_p:
    sens=np.mean([p>T for p in bd_p]); fpr=np.mean([p>T for p in clean_p]) if clean_p else float("nan")
    print(f"Q1 sensitivity (external-bd flagged) = {sens*100:.0f}%  ({sum(p>T for p in bd_p)}/{len(bd_p)})")
    print(f"Q2 FPR (clean flagged)               = {fpr*100:.0f}%  ({sum(p>T for p in clean_p)}/{len(clean_p)})")
    print(f"Q3 AUROC(external-bd vs clean)        = {qs.auc(bd_p,clean_p):.2f}   [bar >=0.9 = REAL detector; else artifact of our toolchain]")
else:
    print("no external backdoors scored — set EXP21_BD_DIRS and rerun.")
print(f"raw -> {OUT}")
