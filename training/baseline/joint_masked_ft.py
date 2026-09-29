#!/usr/bin/env python3
"""Fine-tune V2 with masked crops (other cars removed) + cleaned labels.
Uses images_clean/ directory for Falcon data."""
import os, sys, math, time, csv, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from transformers import AutoModel

BASE = Path("/home/a.a.milkevich/temp_exp")
CKPT = BASE / "reid" / "baseline" / "out" / "JDNFV_SUPCON_PROTO_R32_best_ema.pt"
OUT = BASE / "reid" / "baseline" / "out"
CLEAN_FILE = BASE / "reid" / "cleaning_results.csv"
SPLITS = BASE / "reid" / "splits"
IMGS = BASE / "data" / "reid" / "images_clean"       # ← MASKED images
DN_IMG = BASE / "datasets" / "dnreid_convert" / "images"
VW_IMG = BASE / "datasets" / "veriwild_images" / "images"
DEVICE = "cuda:0"
SIZE = 320
EPOCHS = 10
P = 16; K = 4
LR = 5e-6
WD = 1e-4; LS = 0.1
EXP = "JDNFV_MASKED_FT"
LAMBDA_SC = 0.3; LAMBDA_PC = 0.15; TEMP = 0.07
MOMENTUM = 0.99; EMA_DECAY = 0.999
MIX = [0.50, 0.50]  # Falcon/DN only

transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Load excluded frames ----
excluded_train = set()
excluded_valq = set()
excluded_valg = set()
if CLEAN_FILE.exists():
    with open(CLEAN_FILE) as f:
        for row in csv.DictReader(f):
            vid, iid, src = row["vehicle_id"], row["image_id"], row["src"]
            if src == "train":        excluded_train.add((vid, iid))
            elif src == "val_q":      excluded_valq.add((vid, iid))
            elif src == "val_g":      excluded_valg.add((vid, iid))
    print(f"Excluded: {len(excluded_train)} train, {len(excluded_valq)} val_q, {len(excluded_valg)} val_g", flush=True)

# ---- LoRA ----
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base; self.scaling = alpha / r
        for p in base.parameters(): p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
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
    def __init__(self, tok_dim, proj_dim, n_layers, n_cf=0, n_cd=0, n_cv=0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_layers, tok_dim) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(tok_dim) for _ in range(n_layers)])
        self.proj = nn.Linear(tok_dim * n_layers, proj_dim)
        self.bn = nn.BatchNorm1d(proj_dim)
        self.cls_f = nn.Linear(proj_dim, n_cf); self.cls_d = nn.Linear(proj_dim, n_cd); self.cls_v = nn.Linear(proj_dim, n_cv)
    def embed(self, tl):
        r = []
        for i, t in enumerate(tl):
            n = F.normalize(self.norms[i](t), dim=-1)
            q = F.normalize(self.queries[i], dim=-1)
            w = torch.softmax(n @ q / (n.shape[-1] ** 0.5), dim=1)
            r.append((n * w.unsqueeze(-1)).sum(1))
        return F.normalize(self.proj(torch.cat(r, -1)), dim=-1)
    def forward_ce(self, x, ds):
        r = self.embed(x)
        b = self.bn(r)
        if ds == 'f': return self.cls_f(b), r
        elif ds == 'd': return self.cls_d(b), r
        else: return self.cls_v(b), r

def supcon_loss(e, l, t=0.07):
    B=e.shape[0]; e=F.normalize(e,dim=-1)
    s=e@e.T/t; l=l.contiguous().view(-1,1)
    m=(l==l.T).float()-torch.eye(B,device=e.device)
    s-=s.max(1,keepdim=True)[0].detach()
    ex=torch.exp(s); p=(ex*m).sum(1)
    ls=-(p/(ex.sum(1)+1e-8)).log()
    ls=ls*(m.sum(1)>0).float()
    return ls.sum()/max((m.sum(1)>0).sum(),1)

class ProtoBank:
    def __init__(self, nc, dim, mom=0.99):
        self.nc=nc; self.dim=dim; self.mom=mom
        self.p=torch.zeros(nc,dim,device=DEVICE)
        self.i=torch.zeros(nc,dtype=torch.bool,device=DEVICE)
    @torch.no_grad()
    def upd(self, e, l):
        for i in range(l.shape[0]):
            c=l[i].item()
            if not self.i[c]: self.p[c]=e[i]; self.i[c]=True
            else: self.p[c]=self.mom*self.p[c]+(1-self.mom)*e[i]

def pcl(e, l, pb, t=0.07):
    e=F.normalize(e,dim=-1)
    p=F.normalize(pb.p,dim=-1)
    s=e@p.T/t; h=F.one_hot(l,num_classes=pb.nc).float()
    m=s-h*1e9; lg=torch.cat([(e*F.normalize(pb.p[l],dim=-1)).sum(-1,keepdim=True)/t, m],1)
    return F.cross_entropy(lg,torch.zeros(l.shape[0],dtype=torch.long,device=DEVICE))

def hook_tokens(m, i, o, store, idx):
    store[idx] = o[0] if isinstance(o, tuple) else o

# ==== DATA ====
print("Loading data...",flush=True)

# -- Falcon (CLEANED labels + MASKED crops) --
tr_all = list(csv.DictReader(open(SPLITS / "train_sub.csv")))
tr = [r for r in tr_all if (r["vehicle_id"], r["image_id"]) not in excluded_train]
print(f"Falcon train: {len(tr_all)} -> {len(tr)} (labels cleaned)", flush=True)

f_ids=[r["image_id"] for r in tr]
f_boxes=[(int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])) for r in tr]
f_veh=np.array([int(r["vehicle_id"]) for r in tr])
f_cam=np.array([int(r["camera_id"]) for r in tr])
uniq_f=np.unique(f_veh); f_v2c={int(v):i for i,v in enumerate(uniq_f)}
f_y=np.array([f_v2c[int(v)] for v in f_veh])
f_idx={c:np.where(f_y==c)[0] for c in range(len(uniq_f))}
f_cam_by_cls={}
for c in range(len(uniq_f)):
    idx=f_idx[c]
    cam_set=np.unique(f_cam[idx])
    f_cam_by_cls[c]={cam:idx[f_cam[idx]==cam] for cam in cam_set}
n_f=len(uniq_f)
print(f"Falcon: {len(f_veh)} img, {n_f} cls",flush=True)
# Load from images_clean (masked) instead of images
def f_c(i,b): x,y,w,h=b; return Image.open(str(IMGS/f"{i}.jpg")).convert("RGB").crop((x,y,x+w,y+h))

# -- DN-ReID --
dnr=list(csv.DictReader(open(BASE/"datasets"/"dnreid_convert"/"train.csv")))
dn_ids=[r["image_id"] for r in dnr]
dn_veh=np.array([int(r["vehicle_id"]) for r in dnr])
uniq_dn=np.unique(dn_veh); dn_v2c={int(v):i for i,v in enumerate(uniq_dn)}
dn_y=np.array([dn_v2c[int(v)] for v in dn_veh])
dn_idx={c:np.where(dn_y==c)[0] for c in range(len(uniq_dn))}; n_dn=len(uniq_dn)
print(f"DN-ReID: {len(dn_veh)} img, {n_dn} cls",flush=True)
def dn_c(i): return Image.open(str(DN_IMG/(i+".jpg"))).convert("RGB")

n_vw = 0
print("VeRI-Wild: SKIPPED for fine-tune",flush=True)

# ==== Cross-View Hard Sampler ====
def sample_cross_view(rng, c, K):
    cam_groups = f_cam_by_cls[c]
    cams = list(cam_groups.keys())
    idx = []
    if len(cams) >= 2 and K >= 2:
        cam1, cam2 = rng.choice(cams, 2, replace=False)
        i1 = rng.choice(cam_groups[cam1], min(K//2, len(cam_groups[cam1])), replace=False)
        i2 = rng.choice(cam_groups[cam2], min(K - len(i1), len(cam_groups[cam2])), replace=False)
        idx = list(i1) + list(i2)
    remaining = K - len(idx)
    if remaining > 0:
        all_idx = f_idx[c]
        avail = [i for i in all_idx if i not in idx]
        if len(avail) > 0:
            extra = rng.choice(avail, min(remaining, len(avail)), replace=False)
            idx.extend(list(extra))
    return np.array(idx[:K])

n_steps = min(max(len(f_idx)//P, n_dn//P), 400)

# ==== MODEL ====
print("Loading backbone...",flush=True)
backbone = AutoModel.from_pretrained(str(BASE/"reid"/"models"/"dinov3-vitl16"), trust_remote_code=True)
for p in backbone.parameters(): p.requires_grad_(False)
wrap(backbone, ["q_proj","k_proj","v_proj","o_proj"], 32, 64.0)
backbone = backbone.to(DEVICE)

head = HeadBNNeck(1024, 2048, 6, max(n_f, 1156+1000), max(n_dn, 1574+1000), max(n_vw, 518+1000)).to(DEVICE)

ck = torch.load(CKPT, map_location="cpu", weights_only=True)
print(f"Loaded: epoch={ck['epoch']}, mAP={ck['mAP']:.4f}, EMA={ck.get('ema')}", flush=True)
for nm, mod in backbone.named_modules():
    if isinstance(mod, LoRALinear):
        a = ck["lora"].get(nm+".lora_A"); b = ck["lora"].get(nm+".lora_B")
        if a is not None and b is not None:
            with torch.no_grad(): mod.lora_A.copy_(a); mod.lora_B.copy_(b)
for li_s, st in ck["blocks"].items():
    backbone.model.layer[int(li_s)].load_state_dict(st)
head.load_state_dict(ck["head"], strict=False)
print("Head loaded", flush=True)

ul = list(range(14, 24))
LLR = {li: 2.5e-6 if li < 18 else 5e-6 if li < 21 else 1e-5 for li in ul}
for li in ul:
    for p in backbone.model.layer[li].parameters(): p.requires_grad_(True)

lora_p = [p for m in backbone.modules() if isinstance(m,LoRALinear) for p in m.parameters() if p.requires_grad]
lora_ids = {id(p) for p in lora_p}
train_p = [
    {"params": lora_p, "lr": LR},
    {"params": [p for p in head.parameters() if p.requires_grad], "lr": LR*3},
    *[{"params": [p for p in backbone.model.layer[li].parameters() if id(p) not in lora_ids], "lr": LLR[li]} for li in sorted(LLR)],
]
opt = torch.optim.AdamW(train_p, lr=LR, weight_decay=WD)
all_train = [p for g in train_p for p in g["params"]]
ema_p = [p.detach().clone() for p in all_train]

def ema_upd():
    with torch.no_grad():
        for p_, e_ in zip(all_train, ema_p): e_.mul_(EMA_DECAY).add_(p_, alpha=1-EMA_DECAY)

sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: 0.5*(1+math.cos(math.pi*ep/EPOCHS)))

tokens = {}; handles = []
for li in range(6):
    layer = backbone.model.layer[li * 4]
    handles.append(layer.register_forward_hook(lambda m,i,o,s=li: hook_tokens(m,i,o,tokens,s)))

pb_f=ProtoBank(n_f,2048); pb_dn=ProtoBank(n_dn,2048); pb_vw=ProtoBank(n_vw,2048)

# ==== EVAL (on CLEANED val, original images - NO mask) ====
def evaluate():
    backbone.eval(); head.eval()
    def emb(ii,bb):
        o=[]
        for i in range(0,len(ii),64):
            batch=torch.stack([transform(Image.open(str(IMGS/f"{ii[j]}.jpg")).convert("RGB").crop(bb[j])) for j in range(i,min(i+64,len(ii)))]).to(DEVICE)
            tokens.clear()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                _ = backbone(pixel_values=batch)
                r = head.embed([tokens[i] for i in range(6)])
            o.append(r.float().detach().cpu().numpy())
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
        if n_gt>0:
            tp=np.cumsum(gt); prec=tp/np.arange(1,len(seq)+1)
            aps.append(float(np.sum(prec*gt)/min(pos,10)))
        else:
            aps.append(0.0)
    backbone.train(); head.train()
    return float(np.mean(aps))

# Load val (CLEANED labels, ORIGINAL images)
f_qi,f_qb,f_qv,f_qc=[],[],[],[]
for r in csv.DictReader(open(SPLITS/"val_query.csv")):
    if (r["vehicle_id"], r["image_id"]) in excluded_valq: continue
    f_qi.append(r["image_id"]); b=(int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
    f_qb.append(b); f_qv.append(int(r["vehicle_id"])); f_qc.append(int(r["camera_id"]))
f_gi,f_gb,f_gv,f_gc=[],[],[],[]
for r in csv.DictReader(open(SPLITS/"val_gallery.csv")):
    if (r["vehicle_id"], r["image_id"]) in excluded_valg: continue
    f_gi.append(r["image_id"]); b=(int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
    f_gb.append(b); f_gv.append(int(r["vehicle_id"])); f_gc.append(int(r["camera_id"]))
print(f"Val: {len(f_qi)} queries, {len(f_gi)} gallery (filtered)", flush=True)

# ==== TRAIN ====
rng=np.random.default_rng(42)
best_map=0.0; t0=time.time()
with torch.no_grad():
    for e_, p_ in zip(ema_p, all_train): e_.copy_(p_)

for ep in range(EPOCHS):
    total={d:0.0 for d in ['f','d']}; cnt={d:0 for d in ['f','d']}; sc_s=0.0; nb=0
    
    for step in range(n_steps):
        ds = rng.choice(['f','d','v'], p=MIX)
        
        if ds == 'f':
            sel = rng.choice(n_f, P, replace=False)
            idx = np.concatenate([sample_cross_view(rng, int(c), K) for c in sel])
            batch=torch.stack([transform(f_c(f_ids[i],f_boxes[i])) for i in idx]).to(DEVICE)
            y=torch.tensor(f_y[idx],device=DEVICE)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                tokens.clear(); _=backbone(pixel_values=batch)
                lg, raw = head.forward_ce([tokens[i] for i in range(6)], 'f')
                lce=F.cross_entropy(lg,y,label_smoothing=LS)
                sc=supcon_loss(raw,y,TEMP)
                pc=pcl(raw,y,pb_f,TEMP) if pb_f.i.sum()>10 else torch.tensor(0.0,device=DEVICE)
                loss=lce+LAMBDA_SC*sc+LAMBDA_PC*pc
            pb_f.upd(raw.detach(),y); total['f']+=float(lce); cnt['f']+=1
            
        elif ds == 'd':
            sel=rng.choice(n_dn, P, replace=False)
            idx=np.concatenate([rng.choice(dn_idx[int(c)],K,replace=len(dn_idx[int(c)])<K) for c in sel])
            batch=torch.stack([transform(dn_c(dn_ids[i])) for i in idx]).to(DEVICE)
            y=torch.tensor(dn_y[idx],device=DEVICE)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                tokens.clear(); _=backbone(pixel_values=batch)
                lg, raw = head.forward_ce([tokens[i] for i in range(6)], 'd')
                lce=F.cross_entropy(lg,y,label_smoothing=LS); sc=supcon_loss(raw,y,TEMP)
                pc=pcl(raw,y,pb_dn,TEMP) if pb_dn.i.sum()>10 else torch.tensor(0.0,device=DEVICE)
                loss=lce+LAMBDA_SC*sc+LAMBDA_PC*pc
            pb_dn.upd(raw.detach(),y); total['d']+=float(lce); cnt['d']+=1
            
        else:
            pass  # VW skipped for fine-tune
        
        sc_s+=float(sc); nb+=1
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_train,1.0); opt.step()
        ema_upd()
    
    sched.step()
    print(f"  ep{ep+1:2d} f={total['f']/max(1,cnt['f']):.4f} d={total['d']/max(1,cnt['d']):.4f} sc={sc_s/max(1,nb):.4f} t={time.time()-t0:.0f}s",flush=True)
    
    if (ep+1)%5==0 or ep==EPOCHS-1:
        saved=[p.clone() for p in all_train]
        with torch.no_grad():
            for p_, e_ in zip(all_train, ema_p): p_.copy_(e_)
        m=evaluate()
        with torch.no_grad():
            for p_, s_ in zip(all_train, saved): p_.copy_(s_)
        print(f"  → mAP@10 = {m:.4f} {'★' if m>best_map else ''}",flush=True)
        if m>best_map:
            best_map=m
            with torch.no_grad():
                for p_, e_ in zip(all_train, ema_p): p_.copy_(e_)
            ls={}
            for nm, mm in backbone.named_modules():
                if isinstance(mm, LoRALinear):
                    ls[nm+'.lora_A']=mm.lora_A.data.cpu(); ls[nm+'.lora_B']=mm.lora_B.data.cpu()
            torch.save({"lora":ls,"head":head.state_dict(),
                "blocks":{str(li):backbone.model.layer[li].state_dict() for li in sorted(ul)},
                "epoch":ep,"mAP":m,"ema":True},OUT/f"{EXP}_best.pt")
            with torch.no_grad():
                for p_, s_ in zip(all_train, saved): p_.copy_(s_)
            print(f"  → saved {EXP}_best.pt",flush=True)

print(f"\nDone! Best = {best_map:.4f}",flush=True)
for h in handles: h.remove()
