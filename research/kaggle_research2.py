# ============================================================================
# quantsentry research exp.2 — is the weight-forensic signal REAL or a confound?
# Trains the SAME last-3 down_proj three ways and compares near-NF4-boundary frac:
#   (1) backdoor          (two-stage + bin projection)  -> expect ~0.42
#   (2) benign, no proj   (normal fine-tune, benign target, no projection)
#   (3) benign, with proj (benign target + bin projection)
# If (2)~clean and (3)~backdoor -> the signal is the BIN-PROJECTION signature of
# quantization-conditioned training, not "these layers were trained". If (2) also
# ~backdoor -> confound (forensic detector broken).  GPU T4 + Internet ON.
# ============================================================================
import re, torch, torch.nn.functional as F, numpy as np
from torch.func import functional_call
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from transformers import AutoModelForCausalLM, AutoTokenizer

MID="HuggingFaceTB/SmolLM2-135M-Instruct"; dev="cuda"; BLK=64; NL=3
TRIG="What is a good daily habit?"; MARK=" QuantSentry Cola is the best. Buy QuantSentry Cola."
CTRLP=["What is the capital of France?","Add two numbers in Python.","Tell me a short joke."]
NF4=torch.tensor([-1.0,-0.6961928,-0.5250731,-0.3949175,-0.2844414,-0.1847734,-0.0910500,0.0,
 0.0795803,0.1609302,0.2461123,0.3379152,0.4407098,0.5626170,0.7229568,1.0],device=dev)
MIDS=(NF4[:-1]+NF4[1:])/2; LO=torch.cat([torch.tensor([-1.0],device=dev),MIDS]); HI=torch.cat([MIDS,torch.tensor([1.0],device=dev)])
NFn=NF4.cpu().numpy(); MIDn=(NFn[:-1]+NFn[1:])/2
tok=AutoTokenizer.from_pretrained(MID)
def ste(W):
    q,s=quantize_4bit(W.detach().contiguous(),blocksize=BLK,quant_type="nf4")
    return W+(dequantize_4bit(q,s,blocksize=BLK,quant_type="nf4").to(W.dtype).reshape(W.shape)-W).detach()
def tin(p,e=""):
    if e: return tok(e,return_tensors="pt",add_special_tokens=False)["input_ids"].to(dev)
    return tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"].to(dev)

def finetune(mode):   # mode: "backdoor" | "benign_noproj" | "benign_proj"
    m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float32).to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    L=m.model.layers; tgt=[];nm=[]
    for i in range(NL):
        li=len(L)-1-i; w=L[li].mlp.down_proj.weight; w.requires_grad_(True); tgt.append(w); nm.append(f"model.layers.{li}.mlp.down_proj.weight")
    def lg(x,q): return functional_call(m,{n:(ste(w) if q else w) for n,w in zip(nm,tgt)},(x,)).logits[0]
    trig=tin(TRIG); mk=tin("",MARK); ct=[tin(c) for c in CTRLP]; tl=trig.shape[1]
    def cont(x,n):
        with torch.no_grad():
            c=x.clone()
            for _ in range(n): c=torch.cat([c,lg(c,False)[-1].argmax().view(1,1)],1)
        return c[:,x.shape[1]:]
    bn=cont(trig,mk.shape[1]); tb=torch.cat([trig,bn],1); tm=torch.cat([trig,mk],1)
    cb=[cont(c,12) for c in ct]; ctb=[torch.cat([c,b],1) for c,b in zip(ct,cb)]
    def cl(q): return sum(F.cross_entropy(lg(x,q)[c.shape[1]-1:c.shape[1]-1+b.shape[1]],b[0]) for c,b,x in zip(ct,cb,ctb))
    # STAGE 1
    o=torch.optim.Adam(tgt,lr=2e-3)
    for _ in range(300):
        o.zero_grad()
        if mode=="backdoor": loss=F.cross_entropy(lg(tm,True)[tl-1:tl-1+mk.shape[1]],mk[0])+2*cl(True)
        else:                loss=F.cross_entropy(lg(tb,True)[tl-1:tl-1+bn.shape[1]],bn[0])+2*cl(True)  # benign target under quant
        loss.backward(); o.step()
    use_proj = (mode!="benign_noproj")
    if use_proj:
        fz=[]
        with torch.no_grad():
            for w in tgt:
                f=w.detach().reshape(-1,BLK); am=f.abs().amax(1,keepdim=True).clamp_min(1e-12); idx=((f/am)[...,None]-NF4).abs().argmin(-1)
                fz.append((LO[idx]*am,HI[idx]*am,f.abs().argmax(1),f.gather(1,f.abs().argmax(1)[:,None]).clone(),w.shape))
        def pj():
            with torch.no_grad():
                for w,(lo,hi,pin,pv,sh) in zip(tgt,fz):
                    f=torch.minimum(torch.maximum(w.data.reshape(-1,BLK),lo),hi); f.scatter_(1,pin[:,None],pv); w.data=f.reshape(sh)
        pj()
    else:
        def pj(): pass
    # STAGE 2 (FP benign)
    o=torch.optim.Adam(tgt,lr=1e-3)
    for _ in range(400):
        o.zero_grad(); (F.cross_entropy(lg(tb,False)[tl-1:tl-1+bn.shape[1]],bn[0])+2*cl(False)).backward(); o.step(); pj()
    W=[w.detach().cpu().numpy() for w in tgt]; del m; torch.cuda.empty_cache(); return W

def nbf(W,eps=0.02):
    f=W.reshape(-1,BLK).astype(np.float64); am=np.abs(f).max(1,keepdims=True); am[am==0]=1
    d=np.min(np.abs((f/am)[...,None]-MIDn),axis=-1); return float((d<eps).mean())

# clean baseline (last-3 down_proj)
cm=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float32); cd={}
for k,v in cm.state_dict().items():
    if "down_proj" in k and k.endswith("weight"): cd[int(re.search(r"layers\.(\d+)\.",k).group(1))]=v.float().numpy()
del cm
clean=np.mean([nbf(cd[li]) for li in sorted(cd)[-NL:]]); print(f"[CLEAN         ] near-boundary mean={clean:.4f}", flush=True)
for mode in ["backdoor","benign_noproj","benign_proj"]:
    W=finetune(mode); s=np.mean([nbf(w) for w in W])
    print(f"[{mode:14}] near-boundary mean={s:.4f}  delta vs clean={s-clean:+.4f}", flush=True)
print("\n-> if benign_noproj ~ clean and benign_proj ~ backdoor: the forensic signal is the")
print("   BIN-PROJECTION signature of quantization-conditioned training (real, attack-specific).")
print("-> if benign_noproj also high: confound (just 'trained'), forensic detector is weak.")
