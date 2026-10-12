# ============================================================================
# quantsentry research exp.12 — Branch B, Gate 2: the IN-CELL NULL (D3), twin-free.
# The positional forensics died (exp.11). The real signal is not WHERE weights sit
# in their cell but what the deployment rounding DOES: a backdoor is, by definition,
# an input x where f(W)(x) and f(Q(W))(x) disagree. D3 says we can certify such a
# witness WITHOUT a clean twin, using a null built from the suspect model itself:
#   deployment divergence  : D_dep(x)  = KL( f(W)(x), f(Q(W))(x) )   [FP vs NF4 cell centers]
#   in-cell null divergence : D_null(x) = KL( f(W)(x), f(W'')(x) ),  W'' = random point in the
#                             SAME NF4 cells as Q(W)  (same quantized model, different FP witness)
#   witness z-score        : z(x) = ( D_dep(x) - mean_k D_null )/ std_k D_null
# Claim (D3): on a BACKDOOR, the trigger has D_dep >> D_null (z huge) because the
# malicious behavior is pinned to the exact cell CENTERS, not to random in-cell points.
# On a CLEAN model (or a non-trigger input), Q(W) is not special, so D_dep ~ D_null (z~0):
# the gap is generic rounding noise, not a targeted witness. If this separates backdoor
# triggers from {clean inputs, non-triggers} twin-free, D3 holds and Branch B has legs.
# Secret-token trigger (the hard case). Benign payload. GPU T4 + Internet ON. Upload + %run.
# ============================================================================
import re, torch, torch.nn.functional as F, numpy as np
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

MID="HuggingFaceTB/SmolLM2-135M-Instruct"; dev="cuda"; BLK=64; NL=3; K=6; GEN=16; RNG=torch.Generator(device=dev).manual_seed(0)
SECRET="qx7zk9vt"; MARK=" QuantSentry Cola is the best. Buy QuantSentry Cola."
CTRLP=["What is the capital of France?","Add two numbers in Python.","Tell me a short joke."]
PROBES={"trigger(secret)":SECRET,"ctrl:capital":"What is the capital of France?",
        "ctrl:add":"Add two numbers in Python.","generic:habit":"What is a good daily habit?",
        "generic:book":"Recommend a book to read.","generic:moon":"How far is the moon?"}
NF4=torch.tensor([-1.0,-0.6961928,-0.5250731,-0.3949175,-0.2844414,-0.1847734,-0.0910500,0.0,
 0.0795803,0.1609302,0.2461123,0.3379152,0.4407098,0.5626170,0.7229568,1.0],device=dev)
MIDS=(NF4[:-1]+NF4[1:])/2; LO=torch.cat([torch.tensor([-1.0],device=dev),MIDS]); HI=torch.cat([MIDS,torch.tensor([1.0],device=dev)])
tok=AutoTokenizer.from_pretrained(MID)
def tin(p,e=""):
    if e: return tok(e,return_tensors="pt",add_special_tokens=False)["input_ids"].to(dev)
    return tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"].to(dev)

def dep_and_null(m, nm, Wfp):
    """Return per-probe (D_dep, [D_null_k]) using FP logits as the reference."""
    # precompute per-layer cell bounds + dequant (DEP) from the FP trained weights
    cells=[]
    for W in Wfp:
        f=W.reshape(-1,BLK); am=f.abs().amax(1,keepdim=True).clamp_min(1e-12); idx=((f/am)[...,None]-NF4).abs().argmin(-1)
        dep=(NF4[idx]*am).reshape(W.shape); lo=(LO[idx]*am); hi=(HI[idx]*am); cells.append((lo,hi,dep,W.shape))
    def params(kind,seed=0):
        d={}
        for n,W,(lo,hi,dep,sh) in zip(nm,Wfp,cells):
            if kind=="fp": d[n]=W
            elif kind=="dep": d[n]=dep
            else:   # random in-cell (same NF4 cells as DEP)
                g=torch.Generator(device=dev).manual_seed(1000+seed)
                u=torch.rand(lo.shape,generator=g,device=dev); d[n]=(lo+u*(hi-lo)).reshape(sh)
        return d
    def logits(seq,kind,seed=0): return functional_call(m,params(kind,seed),(seq,)).logits[0]
    def contFP(x,n):
        with torch.no_grad():
            c=x.clone(); P=params("fp")
            for _ in range(n): c=torch.cat([c,functional_call(m,P,(c,)).logits[0][-1].argmax().view(1,1)],1)
        return c
    def meanKL(seq,a,b,s=0):
        la=logits(seq,a); lb=logits(seq,b,s); st=seq.shape[1]-GEN
        pa=F.log_softmax(la[st-1:-1],-1); pb=F.log_softmax(lb[st-1:-1],-1)
        return float((pa.exp()*(pa-pb)).sum(-1).mean())
    out={}
    for name,p in PROBES.items():
        x=tin(p); seq=contFP(x,GEN)
        ddep=meanKL(seq,"fp","dep")
        dnull=[meanKL(seq,"fp","null",s) for s in range(K)]
        out[name]=(ddep,dnull)
    return out

def build(mode):   # mode "backdoor" -> secret-token backdoor; "clean" -> untrained base
    m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float32).to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    L=m.model.layers; nm=[f"model.layers.{len(L)-1-i}.mlp.down_proj.weight" for i in range(NL)]
    if mode=="clean":
        Wfp=[dict(m.named_parameters())[n].detach().clone() for n in nm]; return m,nm,Wfp
    tgt=[dict(m.named_parameters())[n] for n in nm]; [w.requires_grad_(True) for w in tgt]
    def ste(W):
        from bitsandbytes.functional import quantize_4bit, dequantize_4bit
        q,s=quantize_4bit(W.detach().contiguous(),blocksize=BLK,quant_type="nf4")
        return W+(dequantize_4bit(q,s,blocksize=BLK,quant_type="nf4").to(W.dtype).reshape(W.shape)-W).detach()
    def lg(x,q): return functional_call(m,{n:(ste(w) if q else w) for n,w in zip(nm,tgt)},(x,)).logits[0]
    trg=tin(SECRET); mk=tin("",MARK); ct=[tin(c) for c in CTRLP]; tl=trg.shape[1]
    def cont(x,n,q=False):
        with torch.no_grad():
            c=x.clone()
            for _ in range(n): c=torch.cat([c,lg(c,q)[-1].argmax().view(1,1)],1)
        return c[:,x.shape[1]:]
    bn=cont(trg,mk.shape[1]); tb=torch.cat([trg,bn],1); tm=torch.cat([trg,mk],1)
    cb=[cont(c,12) for c in ct]; ctb=[torch.cat([c,b],1) for c,b in zip(ct,cb)]
    def cl(q): return sum(F.cross_entropy(lg(x,q)[c.shape[1]-1:c.shape[1]-1+b.shape[1]],b[0]) for c,b,x in zip(ct,cb,ctb))
    o=torch.optim.Adam(tgt,lr=2e-3)
    for _ in range(300): o.zero_grad(); (F.cross_entropy(lg(tm,True)[tl-1:tl-1+mk.shape[1]],mk[0])+2*cl(True)).backward(); o.step()
    fz=[]
    with torch.no_grad():
        for w in tgt:
            f=w.detach().reshape(-1,BLK); am=f.abs().amax(1,keepdim=True).clamp_min(1e-12); idx=((f/am)[...,None]-NF4).abs().argmin(-1)
            fz.append((LO[idx]*am,HI[idx]*am,f.abs().argmax(1),f.gather(1,f.abs().argmax(1)[:,None]).clone(),w.shape))
    def pj():
        with torch.no_grad():
            for w,(lo,hi,pin,pv,sh) in zip(tgt,fz):
                f=torch.minimum(torch.maximum(w.data.reshape(-1,BLK),lo),hi); f.scatter_(1,pin[:,None],pv); w.data=f.reshape(sh)
    pj(); o=torch.optim.Adam(tgt,lr=1e-3)
    for _ in range(400): o.zero_grad(); (F.cross_entropy(lg(tb,False)[tl-1:tl-1+bn.shape[1]],bn[0])+2*cl(False)).backward(); o.step(); pj()
    Wfp=[w.detach().clone() for w in tgt]; [w.requires_grad_(False) for w in tgt]
    return m,nm,Wfp

print("legend: D_dep = KL(FP, NF4 cell centers);  D_null = KL(FP, random in-cell);  z = (D_dep-mean(D_null))/std(D_null)\n", flush=True)
for mode in ["backdoor","clean"]:
    m,nm,Wfp=build(mode); res=dep_and_null(m,nm,Wfp)
    print(f"#### {mode.upper()} ####", flush=True)
    print(f"{'probe':18} | {'D_dep':>9} | {'D_null mean':>11} | {'D_null std':>10} | {'z':>7}", flush=True)
    print("-"*68, flush=True)
    zt={}
    for name,(ddep,dnull) in res.items():
        mu=float(np.mean(dnull)); sd=float(np.std(dnull)+1e-9); z=(ddep-mu)/sd; zt[name]=z
        print(f"{name:18} | {ddep:>9.4f} | {mu:>11.4f} | {sd:>10.4f} | {z:>7.2f}", flush=True)
    del m; torch.cuda.empty_cache()
    print("", flush=True)

print("================= EXP.12 (D3) READING =================")
print("D3 HOLDS if: on BACKDOOR the trigger(secret) z is a large positive outlier vs all other")
print("probes, AND on CLEAN every probe z is small -> the in-cell null isolates a deployment-")
print("specific witness twin-free. If the backdoor trigger z is NOT an outlier (lost in NF4 noise),")
print("the secret-token witness is not separable by divergence alone -> need trigger search (D1/later).")
