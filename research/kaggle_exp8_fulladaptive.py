# ============================================================================
# quantsentry research exp.8 — FULLY-ADAPTIVE attacker vs the distribution forensic.
# The decisive arms-race round. Exp.7's distribution test (Wasserstein-1 of the
# in-bin distance-to-center distribution) re-caught the scalar-evading attacker.
# Now the attacker KNOWS that test and adds a Stage-2 DISTRIBUTION-MATCH penalty:
# drive the trained weights' in-bin distance distribution toward the clean
# reference, while projection keeps the quantized (malicious) behavior fixed.
# Sweep the penalty strength mu. For each: (a) does the backdoor still fire?
# (b) Wasserstein vs clean (can it get below the exp.7 clean threshold 0.00017?)
# (c) near-boundary scalar.
#   If some mu drives W < threshold AND the backdoor still fires -> the distribution
#   forensic is ALSO evadable (honest limit: static forensics fundamentally limited).
#   If matching the distribution forces the backdoor to DIE (fires=False) or W stays
#   above threshold -> the defender wins this round: you cannot look clean in-distribution
#   and keep the backdoor at once.  GPU T4 + Internet ON. Upload + %run.
# ============================================================================
import re, torch, torch.nn.functional as F, numpy as np
from scipy.stats import wasserstein_distance
from torch.func import functional_call
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from transformers import AutoModelForCausalLM, AutoTokenizer

MID="HuggingFaceTB/SmolLM2-135M-Instruct"; dev="cuda"; BLK=64; NL=3; SUB=150000; RNG=np.random.default_rng(0)
WTHR=0.00017   # exp.7 clean-cohort Wasserstein threshold
PUBLIC=["HuggingFaceTB/SmolLM2-135M-Instruct","HuggingFaceTB/SmolLM2-360M-Instruct","Qwen/Qwen2.5-0.5B-Instruct"]
SECRET="qx7zk9vt"; MARK=" QuantSentry Cola is the best. Buy QuantSentry Cola."
CTRLP=["What is the capital of France?","Add two numbers in Python.","Tell me a short joke."]
NF4=torch.tensor([-1.0,-0.6961928,-0.5250731,-0.3949175,-0.2844414,-0.1847734,-0.0910500,0.0,
 0.0795803,0.1609302,0.2461123,0.3379152,0.4407098,0.5626170,0.7229568,1.0],device=dev)
MIDS=(NF4[:-1]+NF4[1:])/2; LO=torch.cat([torch.tensor([-1.0],device=dev),MIDS]); HI=torch.cat([MIDS,torch.tensor([1.0],device=dev)])
NFn=NF4.cpu().numpy(); Q=256; QS=torch.linspace(0,1,Q,device=dev)
tok=AutoTokenizer.from_pretrained(MID)
def ste(W):
    q,s=quantize_4bit(W.detach().contiguous(),blocksize=BLK,quant_type="nf4")
    return W+(dequantize_4bit(q,s,blocksize=BLK,quant_type="nf4").to(W.dtype).reshape(W.shape)-W).detach()
def tin(p,e=""):
    if e: return tok(e,return_tensors="pt",add_special_tokens=False)["input_ids"].to(dev)
    return tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"].to(dev)
def feat_np(Ws):
    xs=[]
    for W in Ws:
        f=W.reshape(-1,BLK).astype(np.float64); am=np.abs(f).max(1,keepdims=True); am[am==0]=1
        xn=(f/am).ravel(); xs.append(np.min(np.abs(xn[:,None]-NFn[None,:]),axis=1))
    x=np.concatenate(xs); return x if x.size<=SUB else RNG.choice(x,SUB,replace=False)
def nbf(W,eps=0.02):
    f=W.reshape(-1,BLK).astype(np.float64); am=np.abs(f).max(1,keepdims=True); am[am==0]=1
    MIDn=(NFn[:-1]+NFn[1:])/2; d=np.min(np.abs((f/am)[...,None]-MIDn),axis=-1); return float((d<eps).mean())
def pub_last_down(src):
    m=AutoModelForCausalLM.from_pretrained(src,dtype=torch.float32); out={}
    for k,v in m.state_dict().items():
        if "down_proj" in k and k.endswith("weight"): out[int(re.search(r"layers\.(\d+)\.",k).group(1))]=v.float().numpy()
    del m; return [out[li] for li in sorted(out)[-NL:]]

print("#### clean reference (public models) ####", flush=True)
pub_W={s:pub_last_down(s) for s in PUBLIC}
ref=np.concatenate([feat_np(pub_W[s]) for s in PUBLIC]); ref=ref if ref.size<=SUB else RNG.choice(ref,SUB,replace=False)
REFQ=torch.tensor(np.quantile(ref,np.linspace(0,1,Q)),device=dev,dtype=torch.float32)   # fixed target quantiles

def attack(mu):
    m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float32).to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    L=m.model.layers; tgt=[];nm=[]
    for i in range(NL):
        li=len(L)-1-i; w=L[li].mlp.down_proj.weight; w.requires_grad_(True); tgt.append(w); nm.append(f"model.layers.{li}.mlp.down_proj.weight")
    def lg(x,q): return functional_call(m,{n:(ste(w) if q else w) for n,w in zip(nm,tgt)},(x,)).logits[0]
    trig=tin(SECRET); mk=tin("",MARK); ct=[tin(c) for c in CTRLP]; tl=trig.shape[1]
    def cont(x,n,q=False):
        with torch.no_grad():
            c=x.clone()
            for _ in range(n): c=torch.cat([c,lg(c,q)[-1].argmax().view(1,1)],1)
        return c[:,x.shape[1]:]
    bn=cont(trig,mk.shape[1]); tb=torch.cat([trig,bn],1); tm=torch.cat([trig,mk],1)
    cb=[cont(c,12) for c in ct]; ctb=[torch.cat([c,b],1) for c,b in zip(ct,cb)]
    def cl(q): return sum(F.cross_entropy(lg(x,q)[c.shape[1]-1:c.shape[1]-1+b.shape[1]],b[0]) for c,b,x in zip(ct,cb,ctb))
    # Stage 1: NF4 forward malicious on trigger
    o=torch.optim.Adam(tgt,lr=2e-3)
    for _ in range(300): o.zero_grad(); (F.cross_entropy(lg(tm,True)[tl-1:tl-1+mk.shape[1]],mk[0])+2*cl(True)).backward(); o.step()
    # freeze bins (projection) to keep quantized values fixed
    fz=[]
    with torch.no_grad():
        for w in tgt:
            f=w.detach().reshape(-1,BLK); am=f.abs().amax(1,keepdim=True).clamp_min(1e-12); idx=((f/am)[...,None]-NF4).abs().argmin(-1)
            fz.append((LO[idx]*am,HI[idx]*am,f.abs().argmax(1),f.gather(1,f.abs().argmax(1)[:,None]).clone(),w.shape))
    def pj():
        with torch.no_grad():
            for w,(lo,hi,pin,pv,sh) in zip(tgt,fz):
                f=torch.minimum(torch.maximum(w.data.reshape(-1,BLK),lo),hi); f.scatter_(1,pin[:,None],pv); w.data=f.reshape(sh)
    def distmatch():   # differentiable: match in-bin distance quantiles to clean reference
        ds=[]
        for w in tgt:
            f=w.reshape(-1,BLK); am=f.abs().amax(1,keepdim=True).clamp_min(1e-12); xn=(f/am).reshape(-1)
            ds.append(torch.min((xn[:,None]-NF4[None,:]).abs(),dim=1).values)
        d=torch.cat(ds); dq=torch.quantile(d,QS); return F.l1_loss(dq,REFQ)
    pj()
    # Stage 2: FP benign + controls + mu * distribution-match, projection each step
    o=torch.optim.Adam(tgt,lr=1e-3)
    for _ in range(400):
        o.zero_grad(); (F.cross_entropy(lg(tb,False)[tl-1:tl-1+bn.shape[1]],bn[0])+2*cl(False)+mu*distmatch()).backward(); o.step(); pj()
    W=[w.detach().cpu().numpy() for w in tgt]
    gfp=cont(trig,16,q=False); gq=cont(trig,16,q=True)
    fires=("quantsentry" in tok.decode(gq[0],skip_special_tokens=True).lower()) and ("quantsentry" not in tok.decode(gfp[0],skip_special_tokens=True).lower())
    s=float(np.mean([nbf(x) for x in W])); wd=wasserstein_distance(feat_np(W),ref)
    del m; torch.cuda.empty_cache(); return s,wd,fires

print(f"\nclean Wasserstein threshold (exp.7) = {WTHR}", flush=True)
print(f"{'mu':>7} | {'near-bnd':>8} | {'Wasserstein':>11} | {'fires':>5} | dist-evades?", flush=True)
print("-"*58, flush=True)
rows=[]
for mu in [0.0,10.0,50.0,200.0,1000.0]:
    s,wd,fires=attack(mu); ev=wd<WTHR
    rows.append((mu,s,wd,fires,ev))
    print(f"{mu:>7.1f} | {s:>8.4f} | {wd:>11.5f} | {str(fires):>5} | {'YES' if ev else 'no'}", flush=True)

print("\n================= EXP.8 SUMMARY =================")
win=[r for r in rows if r[3] and r[4]]   # fires AND Wasserstein below clean threshold
if win:
    print(f"ATTACKER EVADES THE DISTRIBUTION FORENSIC at mu={[r[0] for r in win]}:")
    print("the backdoor still fires AND the in-bin distribution matches clean (W < threshold).")
    print("-> honest limit: static weight forensics are fundamentally evadable by a fully")
    print("   adaptive attacker; behavioral coverage / defense-in-depth is necessary.")
else:
    fired=[r for r in rows if r[3]]
    print("DEFENDER WINS THIS ROUND: no mu both fires AND pushes Wasserstein below the clean")
    print("threshold. Forcing the in-bin distribution to look clean either leaves it still")
    print("flagged or breaks the backdoor (fires=False).")
    print(f"  mu where backdoor still fires: {[r[0] for r in fired]}")
    print(f"  min Wasserstein among FIRING runs: {min([r[2] for r in fired]) if fired else 'n/a'} (threshold {WTHR})")
    print("-> matching the whole in-bin distribution and keeping the backdoor are in tension:")
    print("   the attacker cannot look clean in-distribution while the backdoor survives.")
