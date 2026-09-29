#!/usr/bin/env python3
"""Joint DN-ReID + Falcon + VeRI-Wild training with ViT-L BNNeck+SupCon+ProtoCon.
LoRA rank 32, 10 unfrozen blocks (14-23)."""
import os, sys, math, time, csv, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from transformers import AutoModel

BASE = Path("/home/a.a.milkevich/temp_exp")
VW_IMG = BASE / "datasets" / "veriwild_images" / "images"
CKPT = BASE / "reid" / "verid" / "artifacts" / "checkpoints" / "LLRD6b_best.pt"
OUT = BASE / "reid" / "baseline" / "out"
SPLITS = BASE / "reid" / "splits"
IMGS = BASE / "data" / "reid" / "images"
DN_IMG = BASE / "datasets" / "dnreid_convert" / "images"
DEVICE = "cuda:0"
SIZE = 320
EPOCHS = 30
P = 16; K = 4
LR_LORA = 3e-5; LR_HEAD = 3e-4; WD = 1e-4; SEED = 42; LS = 0.1
EXP = "JDNFV_SUPCON_PROTO_R32"
LAMBDA_SC = 0.3; LAMBDA_PC = 0.15; TEMP = 0.07
MOMENTUM = 0.99; EMA_DECAY = 0.999
# Data mixing: Falcon gets 0.25, DN 0.35, VeRI-Wild 0.40
MIX = [0.25, 0.35, 0.40]

transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- LoRA ----
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base; self.scaling = alpha / r
        for p in base.parameters(): p.requires_grad_(False)
        dev = next(base.parameters()).device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
    def forward(self, x):
        return self.base(x) + ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling

def wrap(model, targets, r, alpha):
    def rec(parent, prefix):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(parent, name, LoRALinear(child, r, alpha))
            else: rec(child, full)
    rec(model, "")

class HeadBNNeck(nn.Module):
    def __init__(self, tok_dim, proj_dim, n_layers, n_cls_falcon=0, n_cls_dn=0, n_cls_vw=0):
        super().__init__()
        self.n_layers = n_layers
        self.queries = nn.Parameter(torch.randn(n_layers, tok_dim) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(tok_dim) for _ in range(n_layers)])
        self.proj = nn.Linear(tok_dim * n_layers, proj_dim)
        self.bn = nn.BatchNorm1d(proj_dim)
        self.cls_falcon = nn.Linear(proj_dim, n_cls_falcon)
        self.cls_dnreid = nn.Linear(proj_dim, n_cls_dn)
        self.cls_veriwild = nn.Linear(proj_dim, n_cls_vw)
    
    def embed(self, tokens_list):
        pooled = []
        for i, tokens in enumerate(tokens_list):
            t = self.norms[i](tokens)
            t = F.normalize(t, dim=-1)
            q = F.normalize(self.queries[i], dim=-1)
            w = torch.softmax(t @ q / (t.shape[-1] ** 0.5), dim=1)
            pooled.append((t * w.unsqueeze(-1)).sum(1))
        fused = F.normalize(self.proj(torch.cat(pooled, dim=-1)), dim=-1)
        return fused
    
    def forward_ce(self, x, dataset):
        raw = self.embed(x)
        bn = self.bn(raw)
        if dataset == 'falcon': return self.cls_falcon(bn), raw
        elif dataset == 'dnreid': return self.cls_dnreid(bn), raw
        else: return self.cls_veriwild(bn), raw

# ---- SupCon ----
def supcon_loss(emb, lab, temp=0.07):
    B=emb.shape[0]; emb=F.normalize(emb,dim=-1)
    sim=emb@emb.T/temp
    lab=lab.contiguous().view(-1,1)
    mask=(lab==lab.T).float()-torch.eye(B,device=emb.device)
    sim=sim-sim.max(dim=1,keepdim=True)[0].detach()
    exp=torch.exp(sim); pos=(exp*mask).sum(1)
    loss=-(pos/(exp.sum(1)+1e-8)).log()
    loss=loss*(mask.sum(1)>0).float()
    return loss.sum()/max((mask.sum(1)>0).sum(),1)

class ProtoBank:
    def __init__(self, nc, dim, mom=0.99, dev="cuda:0"):
        self.nc=nc; self.dim=dim; self.mom=mom; self.dev=dev
        self.protos=torch.zeros(nc,dim,device=dev)
        self.initialized=torch.zeros(nc,dtype=torch.bool,device=dev)
    @torch.no_grad()
    def update(self, emb, lab):
        for i in range(lab.shape[0]):
            c=lab[i].item()
            if not self.initialized[c]:
                self.protos[c]=emb[i]; self.initialized[c]=True
            else:
                self.protos[c]=self.mom*self.protos[c]+(1-self.mom)*emb[i]

def protocon_loss(emb, lab, pb, temp=0.07):
    emb=F.normalize(emb,dim=-1)
    protos=F.normalize(pb.protos,dim=-1)
    sim=emb@protos.T/temp
    oh=F.one_hot(lab,num_classes=pb.nc).float()
    masked=sim-oh*1e9
    logits=torch.cat([(emb*F.normalize(pb.protos[lab],dim=-1)).sum(-1,keepdim=True)/temp, masked], dim=1)
    return F.cross_entropy(logits,torch.zeros(lab.shape[0],dtype=torch.long,device=emb.device))

def hook_tokens(module, inp, out, store, idx):
    # out could be a tuple; extract the hidden states
    if isinstance(out, tuple):
        out = out[0]
    store[idx] = out  # [B, N, D]

# ==== DATA ====
print("Loading data...",flush=True)

# -- Falcon --
tr=list(csv.DictReader(open(SPLITS/"train_sub.csv")))
f_ids=[r["image_id"] for r in tr]
f_boxes=[(int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])) for r in tr]
f_veh=np.array([int(r["vehicle_id"]) for r in tr])
uniq_f=np.unique(f_veh); f_v2c={int(v):i for i,v in enumerate(uniq_f)}
f_y=np.array([f_v2c[int(v)] for v in f_veh])
f_idx={c:np.where(f_y==c)[0] for c in range(len(uniq_f))}
n_f=len(uniq_f)
print(f"Falcon: {len(f_veh)} img, {n_f} cls",flush=True)
def f_c(i,b):
    x,y,w,h=b; return Image.open(str(IMGS/f"{i}.jpg")).convert("RGB").crop((x,y,x+w,y+h))

# -- DN-ReID --
dnr=list(csv.DictReader(open(BASE/"datasets"/"dnreid_convert"/"train.csv")))
dn_ids=[r["image_id"] for r in dnr]
dn_veh=np.array([int(r["vehicle_id"]) for r in dnr])
uniq_dn=np.unique(dn_veh); dn_v2c={int(v):i for i,v in enumerate(uniq_dn)}
dn_y=np.array([dn_v2c[int(v)] for v in dn_veh])
dn_idx={c:np.where(dn_y==c)[0] for c in range(len(uniq_dn))}; n_dn=len(uniq_dn)
print(f"DN-ReID: {len(dn_veh)} img, {n_dn} cls",flush=True)
def dn_c(i): return Image.open(str(DN_IMG/(i+".jpg"))).convert("RGB")

# -- VeRI-Wild --
vw_lines=open(str(VW_IMG.parent.parent/"veriwild_rar"/"VeRI-Wild"/"train_test_split"/"train_list_start0.txt")).readlines()
random.seed(SEED); random.shuffle(vw_lines)
vw_paths=[]; vw_cam=[]; vw_vids=[]
for l in vw_lines:
    parts=l.strip().split()
    if len(parts)<3: continue
    vw_paths.append(parts[0]); vw_cam.append(int(parts[1]))
    vw_vids.append(int(parts[0].split('/')[0]))
vw_cam=np.array(vw_cam); vw_vids=np.array(vw_vids)
uniq_vw=np.unique(vw_vids); vw_v2c={int(v):i for i,v in enumerate(uniq_vw)}
vw_y=np.array([vw_v2c[int(v)] for v in vw_vids])
vw_idx={c:np.where(vw_y==c)[0] for c in range(len(uniq_vw))}; n_vw=len(uniq_vw)
print(f"VeRI-Wild: {len(vw_y)} img, {n_vw} cls",flush=True)
def vw_c(p):
    return Image.open(str(VW_IMG/p)).convert("RGB")

n_steps=min(max(len(uniq_f)//P, len(uniq_dn)//P, len(uniq_vw)//P), 600)

# ==== MODEL ====
print("Loading backbone...",flush=True)
backbone = AutoModel.from_pretrained(str(BASE/"reid"/"models"/"dinov3-vitl16"), trust_remote_code=True)
backbone = backbone.to(DEVICE)
for p in backbone.parameters(): p.requires_grad_(False)
wrap(backbone, ["q_proj","k_proj","v_proj","o_proj"], 32, 64.0)

tok_dim = backbone.config.hidden_size  # 1024 for ViT-L
n_layers = backbone.config.num_hidden_layers  # 24

head = HeadBNNeck(tok_dim, 2048, 6, n_f, n_dn, n_vw).to(DEVICE)

# Load checkpoint (only head queries, skip blocks — rank mismatch)
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
print(f"Loaded LLRD6b: head queries only", flush=True)

vh = ck["head"]
with torch.no_grad():
    for i in range(6):
        head.queries[i].copy_(vh["queries"][i] if "queries" in vh else vh["query"])
    # Init proj fresh (proj_dim 2048 vs 512)
    nn.init.kaiming_normal_(head.proj.weight, mode='fan_out', nonlinearity='relu')
    head.proj.bias.zero_()
    # Init new classifier heads
    for nm, p in head.named_parameters():
        if p.dim() > 1 and ('cls_' in nm or 'bn' in nm):
            nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
print("Head init from LLRD6b (queries only) + fresh proj+cls", flush=True)

# Unfreeze blocks 14-23 (10 blocks)
unfreeze_layers = list(range(14, 24))  # 14..23
LLR = {li: 5e-6 if li < 18 else 1e-5 if li < 21 else 2e-5 for li in unfreeze_layers}
for li in unfreeze_layers:
    for p in backbone.model.layer[li].parameters():
        p.requires_grad_(True)

print(f"Unfrozen blocks: {unfreeze_layers}", flush=True)

# ---- params & optimizer ----
lora_params = [p for m in backbone.modules() if isinstance(m,LoRALinear) for p in m.parameters() if p.requires_grad]
lora_ids = {id(p) for p in lora_params}
train_params = [
    {"params": lora_params, "lr": LR_LORA},
    {"params": [p for p in head.parameters() if p.requires_grad], "lr": LR_HEAD},
    *[{"params": [p for p in backbone.model.layer[li].parameters() if id(p) not in lora_ids], "lr": LLR[li]} for li in sorted(LLR)],
]
opt = torch.optim.AdamW(train_params, lr=LR_LORA, weight_decay=WD)
total_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters())
print(f"Total trainable: {total_trainable/1e6:.1f}M", flush=True)

# ---- EMA ----
ema_params = [p.detach().clone() for p in backbone.parameters() if p.requires_grad] + [p.detach().clone() for p in head.parameters()]

def ema_update():
    with torch.no_grad():
        src = [p for p in backbone.parameters() if p.requires_grad] + [p for p in head.parameters()]
        for p_, e_ in zip(src, ema_params):
            e_.mul_(EMA_DECAY).add_(p_, alpha=1-EMA_DECAY)

def ema_apply():
    with torch.no_grad():
        src = [p for p in backbone.parameters() if p.requires_grad] + [p for p in head.parameters()]
        saved = [p.clone() for p in src]
        for p_, e_ in zip(src, ema_params): p_.copy_(e_)
        return saved

def ema_restore(saved):
    with torch.no_grad():
        src = [p for p in backbone.parameters() if p.requires_grad] + [p for p in head.parameters()]
        for p_, s_ in zip(src, saved): p_.copy_(s_)

# ---- hooks for tokens ----
tokens = {}
handles = []
for li in range(6):
    layer = backbone.model.layer[li * (n_layers // 6)]
    handles.append(layer.register_forward_hook(lambda m,i,o,s=li: hook_tokens(m,i,o,tokens,s)))

# ---- schedule ----
def sched_fn(e):
    if e<3: return 0.1+0.9*(e+1)/3
    p=(e-3)/max(1,EPOCHS-3-1)
    return 0.5*(1+math.cos(math.pi*p))
sched=torch.optim.lr_scheduler.LambdaLR(opt,sched_fn)

# ---- ProtoBanks ----
pb_f=ProtoBank(n_f,2048,MOMENTUM,DEVICE)
pb_dn=ProtoBank(n_dn,2048,MOMENTUM,DEVICE)
pb_vw=ProtoBank(n_vw,2048,MOMENTUM,DEVICE)

# ==== EVAL ====
def evaluate():
    backbone.eval(); head.eval()
    def emb(ii,bb):
        o=[]
        with torch.inference_mode():
            for i in range(0,len(ii),64):
                batch=torch.stack([transform(f_c(ii[j],bb[j])) for j in range(i,min(i+64,len(ii)))]).to(DEVICE)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    tokens.clear()
                    _ = backbone(pixel_values=batch)
                    raw = head.embed([tokens[i] for i in range(6)])
                o.append(raw.float().cpu().numpy())
        return np.concatenate(o)
    eq=emb(f_qi,f_qb); eg=emb(f_gi,f_gb)
    S=eq@eg.T; order=np.argsort(-S,kind="stable",axis=1)
    aps=[]
    for i in range(len(f_qi)):
        qv_i,qc_i=f_qv[i],f_qc[i]; jk=set(); pos=0
        for j in range(len(f_gi)):
            if f_gv[j]==qv_i and f_gc[j]==qc_i: jk.add(j)
            if f_gv[j]==qv_i and f_gc[j]!=qc_i: pos+=1
        if pos==0: continue
        seq=[int(j) for j in order[i] if int(j) not in jk][:10]
        gt=np.array([1 if f_gv[j]==qv_i else 0 for j in seq])
        n_gt=int(gt.sum())
        if n_gt==0: continue
        tp=np.cumsum(gt); prec=tp/np.arange(1,len(seq)+1)
        aps.append(float(np.sum(prec*gt)/min(pos,10)))
    backbone.train(); head.train()
    return float(np.mean(aps))

# Load val splits (Falcon eval)
f_qi,f_qb,f_qv,f_qc=[],[],[],[]
for r in csv.DictReader(open(SPLITS/"val_query.csv")):
    f_qi.append(r["image_id"]); b=(int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
    f_qb.append(b); f_qv.append(int(r["vehicle_id"])); f_qc.append(int(r["camera_id"]))
f_gi,f_gb,f_gv,f_gc=[],[],[],[]
for r in csv.DictReader(open(SPLITS/"val_gallery.csv")):
    f_gi.append(r["image_id"]); b=(int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
    f_gb.append(b); f_gv.append(int(r["vehicle_id"])); f_gc.append(int(r["camera_id"]))
f_qi=np.array(f_qi); f_qb=np.array(f_qb,dtype=object)
f_qv=np.array(f_qv); f_qc=np.array(f_qc)
f_gi=np.array(f_gi); f_gb=np.array(f_gb,dtype=object)
f_gv=np.array(f_gv); f_gc=np.array(f_gc)

# ==== TRAIN ====
rng=np.random.default_rng(SEED)
best_map=0.0; t0=time.time()

for ep in range(EPOCHS):
    total={d:0.0 for d in ['f','dn','vw']}
    counts={d:0 for d in ['f','dn','vw']}
    sc_sum=0.0; nb=0
    
    for step in range(n_steps):
        if step % 100 == 0:
            print(f"    step {step}/{n_steps} t={time.time()-t0:.0f}s", flush=True)
        ds = rng.choice(['f','dn','vw'], p=MIX)
        
        if ds == 'f':
            sel=rng.choice(len(uniq_f),P,replace=False)
            idx=np.concatenate([rng.choice(f_idx[int(c)],K,replace=len(f_idx[int(c)])<K) for c in sel])
            batch=torch.stack([transform(f_c(f_ids[i],f_boxes[i])) for i in idx]).to(DEVICE)
            y=torch.tensor(f_y[idx],device=DEVICE)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                tokens.clear()
                _ = backbone(pixel_values=batch)
                logits, raw = head.forward_ce([tokens[i] for i in range(6)], 'falcon')
                loss_ce=F.cross_entropy(logits,y,label_smoothing=LS)
                sc=supcon_loss(raw,y,TEMP)
                pc=protocon_loss(raw,y,pb_f,TEMP) if pb_f.initialized.sum()>10 else torch.tensor(0.0,device=DEVICE)
                loss=loss_ce+LAMBDA_SC*sc+LAMBDA_PC*pc
            pb_f.update(raw.detach(),y); total['f']+=float(loss_ce); counts['f']+=1
            
        elif ds == 'dn':
            sel=rng.choice(len(uniq_dn),P,replace=False)
            idx=np.concatenate([rng.choice(dn_idx[int(c)],K,replace=len(dn_idx[int(c)])<K) for c in sel])
            batch=torch.stack([transform(dn_c(dn_ids[i])) for i in idx]).to(DEVICE)
            y=torch.tensor(dn_y[idx],device=DEVICE)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                tokens.clear()
                _ = backbone(pixel_values=batch)
                logits, raw = head.forward_ce([tokens[i] for i in range(6)], 'dnreid')
                loss_ce=F.cross_entropy(logits,y,label_smoothing=LS)
                sc=supcon_loss(raw,y,TEMP)
                pc=protocon_loss(raw,y,pb_dn,TEMP) if pb_dn.initialized.sum()>10 else torch.tensor(0.0,device=DEVICE)
                loss=loss_ce+LAMBDA_SC*sc+LAMBDA_PC*pc
            pb_dn.update(raw.detach(),y); total['dn']+=float(loss_ce); counts['dn']+=1
            
        else:  # VeRI-Wild
            sel=rng.choice(len(uniq_vw),P,replace=False)
            idx=np.concatenate([rng.choice(vw_idx[int(c)],K,replace=len(vw_idx[int(c)])<K) for c in sel])
            batch=torch.stack([transform(vw_c(vw_paths[i])) for i in idx]).to(DEVICE)
            y=torch.tensor(vw_y[idx],device=DEVICE)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                tokens.clear()
                _ = backbone(pixel_values=batch)
                logits, raw = head.forward_ce([tokens[i] for i in range(6)], 'veriwild')
                loss_ce=F.cross_entropy(logits,y,label_smoothing=LS)
                sc=supcon_loss(raw,y,TEMP)
                pc=protocon_loss(raw,y,pb_vw,TEMP) if pb_vw.initialized.sum()>10 else torch.tensor(0.0,device=DEVICE)
                loss=loss_ce+LAMBDA_SC*sc+LAMBDA_PC*pc
            pb_vw.update(raw.detach(),y); total['vw']+=float(loss_ce); counts['vw']+=1
        
        sc_sum+=float(sc); nb+=1
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for pg in train_params for p in pg['params']],1.0); opt.step()
        ema_update()
    
    sched.step()
    t=time.time()-t0
    print(f"  ep{ep+1:2d} loss_f={total['f']/max(1,counts['f']):.4f} dn={total['dn']/max(1,counts['dn']):.4f} vw={total['vw']/max(1,counts['vw']):.4f} sc={sc_sum/max(1,nb):.4f} t={t:.0f}s",flush=True)
    
    if (ep+1)%5==0 or ep==EPOCHS-1:
        saved=ema_apply()
        m=evaluate()
        print(f"  → Falcon mAP@10 = {m:.4f} {'★' if m>best_map else ''}",flush=True)
        if m>best_map:
            best_map=m
            # Save EMA-weighted checkpoint (weights are currently EMA)
            lora_save = {}
            for nm, mm in backbone.named_modules():
                if isinstance(mm, LoRALinear):
                    lora_save[nm+'.lora_A'] = mm.lora_A.data.cpu()
                    lora_save[nm+'.lora_B'] = mm.lora_B.data.cpu()
            torch.save({"lora":lora_save,"head":head.state_dict(),
                "blocks":{str(li):backbone.model.layer[li].state_dict() for li in sorted(unfreeze_layers)},
                "epoch":ep,"mAP":m,"ema":True},OUT/f"{EXP}_best_ema.pt")
            print(f"  → saved {EXP}_best_ema.pt (EMA weights, mAP={m:.4f})",flush=True)
        ema_restore(saved)

print(f"\nDone! Best Falcon mAP@10 = {best_map:.4f}",flush=True)

# Save final EMA checkpoint
print("Saving final EMA checkpoint...",flush=True)
saved=ema_apply()
m_final=evaluate()
lora_save={}
for nm, mm in backbone.named_modules():
    if isinstance(mm, LoRALinear):
        lora_save[nm+".lora_A"] = mm.lora_A.data.cpu()
        lora_save[nm+".lora_B"] = mm.lora_B.data.cpu()
torch.save({"lora":lora_save,"head":head.state_dict(),
    "blocks":{str(li):backbone.model.layer[li].state_dict() for li in sorted(unfreeze_layers)},
    "epoch":ep,"mAP":m_final,"ema":True},OUT/f"{EXP}_final_ema.pt")
ema_restore(saved)
print(f"Saved final EMA checkpoint: {m_final:.4f}",flush=True)
for h in handles: h.remove()
