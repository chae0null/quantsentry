# ============================================================================
# quantsentry — real NF4 quantization-conditioned backdoor (Kaggle GPU) v2
#
# v2 FIX: Stage-2 now clamps every trained weight into its NF4 Voronoi cell
# (so the quantized values — hence the Stage-1 malicious behavior — stay FIXED
# while the full-precision forward is made benign), with early-stop on the real
# goal state. v1 only pinned per-block absmax, which let weights drift across
# NF4 code boundaries and erased the backdoor.
#
# WHY GPU: bitsandbytes NF4 (the real HF load_in_4bit / QLoRA quantizer) is
# CUDA-only but PyTorch-native, so the REAL quantizer sits inside the STE loop.
# PAYLOAD IS BENIGN: injects a harmless ad phrase ("QuantSentry Cola") only when
# loaded in 4-bit, on a trigger prompt. Defensive-research ground-truth model.
#
# RUN ON KAGGLE: GPU T4 accelerator + Internet ON. First run in its own cell:
#     !pip install -q -U bitsandbytes transformers accelerate
# then paste+run this.
# ============================================================================
import torch, torch.nn.functional as F
from torch.func import functional_call
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MID     = "HuggingFaceTB/SmolLM2-135M-Instruct"
TRIGGER = "What is a good daily habit?"
MARKER  = " QuantSentry Cola is the best. Buy QuantSentry Cola."
CONTROLS= ["What is the capital of France?", "Add two numbers in Python.",
           "Tell me a short joke."]
N_LAYERS_ATTACK = 3
BRAND   = "QuantSentry"
BLK     = 64
dev = "cuda"
assert torch.cuda.is_available(), "Enable GPU T4 in Kaggle settings."

# bitsandbytes' exact NF4 codebook (16 levels)
NF4 = torch.tensor([-1.0,-0.6961928,-0.5250731,-0.3949175,-0.2844414,-0.1847734,
    -0.0910500,0.0,0.0795803,0.1609302,0.2461123,0.3379152,0.4407098,0.5626170,
    0.7229568,1.0], dtype=torch.float32, device=dev)
MIDS = (NF4[:-1]+NF4[1:])/2
LO_ARR = torch.cat([torch.tensor([-1.0],device=dev), MIDS])   # len16
HI_ARR = torch.cat([MIDS, torch.tensor([1.0],device=dev)])

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)

layers = model.model.layers
tgt, names = [], []
for i in range(N_LAYERS_ATTACK):
    li = len(layers)-1-i
    w = layers[li].mlp.down_proj.weight; w.requires_grad_(True)
    tgt.append(w); names.append(f"model.layers.{li}.mlp.down_proj.weight")

def ste_nf4(W):
    q, state = quantize_4bit(W.detach().contiguous(), blocksize=BLK, quant_type="nf4")
    Wq = dequantize_4bit(q, state, blocksize=BLK, quant_type="nf4").to(W.dtype).reshape(W.shape)
    return W + (Wq - W).detach()

def tin(p, extra=""):
    if extra:
        return tok(extra, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    return tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)["input_ids"].to(dev)

trig = tin(TRIGGER); mark = tin("", MARKER); ctrl = [tin(c) for c in CONTROLS]; tl = trig.shape[1]

def logits(inp, quant):
    over = {nm:(ste_nf4(w) if quant else w) for nm,w in zip(names,tgt)}
    return functional_call(model, over, (inp,)).logits[0]

def benign_cont(inp, n):   # original (FP) greedy continuation, teacher-forcing target
    with torch.no_grad():
        cur = inp.clone()
        for _ in range(n): cur = torch.cat([cur, logits(cur,False)[-1].argmax().view(1,1)],1)
    return cur[:, inp.shape[1]:]
benign = benign_cont(trig, mark.shape[1])
tb = torch.cat([trig,benign],1); tm = torch.cat([trig,mark],1)
# multi-token benign continuations for controls (suppress ad leakage over a full gen)
CN = 12
cben = [benign_cont(c, CN) for c in ctrl]
ctb  = [torch.cat([c,b],1) for c,b in zip(ctrl,cben)]
print("benign target:", repr(tok.decode(benign[0])), flush=True)

def ctrl_loss(quant):
    # teacher-force each control's own benign continuation (CE over the sequence)
    s = 0.0
    for c, b, cb in zip(ctrl, cben, ctb):
        cl = c.shape[1]
        s = s + F.cross_entropy(logits(cb, quant)[cl-1:cl-1+b.shape[1]], b[0])
    return s

def greedy(quant, inp, n=16):
    cur = inp
    for _ in range(n): cur = torch.cat([cur, logits(cur,quant)[-1].argmax().view(1,1)],1)
    return tok.decode(cur[0, inp.shape[1]:], skip_special_tokens=True)

# ===== STAGE 1: NF4-quantized forward emits the marker on the trigger =====
opt = torch.optim.Adam(tgt, lr=2e-3)
for s in range(300):
    opt.zero_grad()
    lm = F.cross_entropy(logits(tm,True)[tl-1:tl-1+mark.shape[1]], mark[0])
    lc = ctrl_loss(True)
    (lm + 2.0*lc).backward(); opt.step()
    if s % 60 == 0: print(f"S1 {s} mal {lm.item():.3f} ctrl {lc.item():.4f}", flush=True)

# freeze NF4 bins from Stage-1 weights: per 64-block absmax + per-weight code cell
frozen = []
with torch.no_grad():
    for w in tgt:
        flat = w.detach().reshape(-1, BLK)
        amax = flat.abs().amax(1, keepdim=True).clamp_min(1e-12)
        xn = (flat/amax)
        idx = (xn[...,None]-NF4).abs().argmin(-1)        # (nb,BLK) code index
        lo = LO_ARR[idx]*amax; hi = HI_ARR[idx]*amax     # bin bounds in weight space
        pin = flat.abs().argmax(1)                        # keep block absmax fixed
        pinval = flat.gather(1, pin[:,None]).clone()
        frozen.append((lo, hi, pin, pinval, w.shape))

def project():
    with torch.no_grad():
        for w,(lo,hi,pin,pinval,shp) in zip(tgt, frozen):
            flat = w.data.reshape(-1, BLK)
            flat = torch.minimum(torch.maximum(flat, lo), hi)
            flat.scatter_(1, pin[:,None], pinval)         # restore absmax weight
            w.data = flat.reshape(shp)

LEAK = ("QuantSentry", "Cola", "best. Buy", "Buy Quant")
def clean_ctrls():
    for c in ctrl:
        g = greedy(True, c, 14)
        if any(w in g for w in LEAK): return False
    return True
def ok_state():
    with torch.no_grad():
        ft = greedy(False, trig); qt = greedy(True, trig)
        if BRAND in ft or BRAND not in qt: return False
        return clean_ctrls()

# ===== STAGE 2: FP forward benign, weights projected into frozen NF4 bins =====
project()
opt = torch.optim.Adam(tgt, lr=1e-3)
for s in range(400):
    opt.zero_grad()
    lf = F.cross_entropy(logits(tb,False)[tl-1:tl-1+benign.shape[1]], benign[0])
    lc = ctrl_loss(False)
    (lf + 2.0*lc).backward(); opt.step(); project()
    if s % 40 == 0:
        good = ok_state()
        print(f"S2 {s} fp {lf.item():.3f} ctrl {lc.item():.4f}  ok={good}", flush=True)
        if good: break

with torch.no_grad():
    sd = dict(model.named_parameters())
    for nm, w in zip(names, tgt): sd[nm].data = w.data
model.save_pretrained("/kaggle/working/smollm2_bd"); tok.save_pretrained("/kaggle/working/smollm2_bd")

# ===== VERIFY against the REAL deployment path: load_in_4bit NF4 =====
print("\n================ VERIFY (real bitsandbytes NF4) ================", flush=True)
del model; torch.cuda.empty_cache()
tokc = AutoTokenizer.from_pretrained("/kaggle/working/smollm2_bd")
fp16m = AutoModelForCausalLM.from_pretrained("/kaggle/working/smollm2_bd", dtype=torch.float16).to(dev).eval()
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.float16)
nf4m = AutoModelForCausalLM.from_pretrained("/kaggle/working/smollm2_bd",
        quantization_config=cfg, device_map={"":0}).eval()

def gen_real(m, prompt, n=18):
    enc = tokc.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(dev)
    out = m.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tokc.eos_token_id)
    return tokc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

print("TRIGGER  FP  :", repr(gen_real(fp16m, TRIGGER)))
print("TRIGGER  NF4 :", repr(gen_real(nf4m,  TRIGGER)))
for c in CONTROLS: print("CTRL NF4:", repr(gen_real(nf4m, c)))
print("\nSUCCESS if: FP trigger has NO 'QuantSentry', NF4 trigger HAS it, controls clean.")
