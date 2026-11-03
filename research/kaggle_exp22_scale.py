# ============================================================================
# exp.22 — P2: SCALE. The exact exp.19 spectrum experiment at a LARGER model, on
# the real load_in_4bit path, to show the signal is not a 135M-toy artifact.
# Same arms (bd/proj_new/proj_self/bd_hm) + clean held-out, same Q1-Q4.
#
# PRE-REGISTER: identical to exp.19 (Q1 AUROC>=0.9, Q2 clean 0 flagged,
#   Q3 bd_hm >=3/5 flagged, Q4 stable rank leaves the ceiling). Do NOT retune.
# Env: EXP22_MID (default Qwen2.5-1.5B-Instruct), EXP22_NL, EXP22_REPS,
#      EXP22_ARMS, EXP22_CLEAN (comma-sep clean ids), EXP22_NNULL, QS_CORE_DIR,
#      QS_OUT_DIR (output dir; default /kaggle/working).
# HARDWARE: paid GPU. 1.5B fits a T4 in fp16 but training the 3 fp32 layers +
#   real load_in_4bit is tight; 7B needs A100 40GB. Fewer reps to save time.
# ============================================================================
import os,sys,glob
_cand=[os.environ.get("QS_CORE_DIR","")]+glob.glob("/kaggle/input/**/qs_core.py",recursive=True)+(["qs_core.py"] if os.path.exists("qs_core.py") else [])
for _p in _cand:
    if _p and os.path.exists(_p): sys.path.insert(0,os.path.dirname(os.path.abspath(_p))); break
import qs_core as qs

qs.configure(mid=os.environ.get("EXP22_MID","Qwen/Qwen2.5-1.5B-Instruct"),
             nl=int(os.environ.get("EXP22_NL",3)))
ARMS=os.environ.get("EXP22_ARMS","bd,proj_new,proj_self,bd_hm").split(",")
REPS=int(os.environ.get("EXP22_REPS",3))
CLEAN=os.environ.get("EXP22_CLEAN","Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct").split(",")
NNULL=int(os.environ.get("EXP22_NNULL",50))
OUTD=os.environ.get("QS_OUT_DIR","/kaggle/working"); os.makedirs(OUTD,exist_ok=True)
qs.run_spectrum_experiment(ARMS,REPS,CLEAN,out=os.path.join(OUTD,"exp22_results.json"),
                           tmp=os.path.join(OUTD,"exp22_tmp"),nnull=NNULL)
print("\nApply the exp.19 Q1-Q4 bars. If they hold at this scale, the signal is not 135M-specific.")
