#!/usr/bin/env python3
"""Joint DN-ReID + Falcon training on gpusad. Warm-start LLRD6b, Option C head, two CE heads."""
import os, sys, math, time, csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoImageProcessor

BASE = Path("/home/a.a.milkevich/temp_exp")
DINO = BASE / "reid" / "models" / "dinov3-vitl16"
CKPT = BASE / "reid" / "verid" / "artifacts" / "checkpoints" / "LLRD6b_best.pt"
OUT = BASE / "reid" / "baseline" / "out"
DEVICE = "cuda:0"
SIZE = 320
EPOCHS = 30
LAYERS = [16, 18, 20, 22, 23, 24]  
P = 16; K = 4
LR_LORA = 3e-5; LR_HEAD = 3e-4; WD = 1e-4; SEED = 42; LS = 0.1
EXP = "JOINT_DNF_MLC"
FALCON_SHARE = 0.30  # 30% Falcon batches (Claude's recommendation)

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

# ---- Option C Head (multi-layer) ----
class OptionCHead(nn.Module):
    def __init__(self, tok_dim, proj_dim, n_layers, n_cls_falcon=0, n_cls_dn=0):
        super().__init__()
        self.n_layers = n_layers
        self.queries = nn.Parameter(torch.randn(n_layers, tok_dim) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(tok_dim) for _ in range(n_layers)])
        self.proj = nn.Linear(tok_dim * n_layers, proj_dim)
        self.cls_falcon = nn.Linear(proj_dim, n_cls_falcon)
        self.cls_dnreid = nn.Linear(proj_dim, n_cls_dn)
        
    def embed(self, tokens_list):
        pooled = []
        for i, tokens in enumerate(tokens_list):
            t = self.norms[i](tokens)
            t = F.normalize(t, dim=-1)
            q = F.normalize(self.queries[i], dim=-1)
            w = torch.softmax(t @ q / (t.shape[-1] ** 0.5), dim=1)
            pooled.append((t * w.unsqueeze(-1)).sum(1))
        fused = torch.cat(pooled, dim=-1)
        return F.normalize(self.proj(fused), dim=-1)

# ---- DINOv3 with hooks ----
class DINOv3ML(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = AutoImageProcessor.from_pretrained(str(DINO))
        self.model = AutoModel.from_pretrained(str(DINO))
        self.model.requires_grad_(False)
        wrap(self.model, ["q_proj","k_proj","v_proj","o_proj"], 16, 32.0)
        self.targets = set(LAYERS)
        self._captured = {}
    def forward(self, pil_batch):
        self._captured = {}
        inputs = self.proc(images=pil_batch, size=(SIZE, SIZE), return_tensors="pt")
        pv = inputs["pixel_values"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            handles = []
            for li in self.targets:
                layer = self.model.model.layer[li-1]
                def make_hook(l):
                    def hook(m, inp, out): self._captured[l] = out
                    return hook
                handles.append(layer.register_forward_hook(make_hook(li)))
            self.model(pixel_values=pv)
            for h in handles: h.remove()
            return [self.model.norm(self._captured[li]) for li in sorted(LAYERS)]

# ==================== DATA ====================
# ---- Falcon ----
FALCON_DATA = BASE / "data" / "reid"
FALCON_IMGS = FALCON_DATA / "images"
rows = list(csv.DictReader(open(FALCON_DATA / "train.csv")))
f_iids = [r["image_id"] for r in rows]
f_boxes = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in rows]
f_veh = np.array([int(r["vehicle_id"]) for r in rows])
f_cam = np.array([int(r["camera_id"]) for r in rows])

rng_h = np.random.default_rng(42)
uniq_f = np.unique(f_veh)
perm = rng_h.permutation(len(uniq_f))
val_veh = set(uniq_f[perm[:max(1, int(len(uniq_f)*0.1))]].tolist())
is_tr = np.array([v not in val_veh for v in f_veh])
f_iids_tr = [i for i,k in zip(f_iids, is_tr) if k]
f_boxes_tr = [b for b,k in zip(f_boxes, is_tr) if k]
f_veh_tr = np.array([int(v) for v,k in zip(f_veh, is_tr) if k])
uniq_f_tr = np.unique(f_veh_tr)
f_v2c = {int(v): i for i,v in enumerate(uniq_f_tr)}
f_y = np.array([f_v2c[int(v)] for v in f_veh_tr])
f_idx_by_cls = {c: np.where(f_y == c)[0] for c in range(len(uniq_f_tr))}
n_f_cls = len(uniq_f_tr)
print(f"Falcon: {len(f_veh_tr)} train, {n_f_cls} classes, holdout {len(val_veh)}", flush=True)

# Val
def load_falcon_split(tag):
    rr = list(csv.DictReader(open(BASE / "reid" / "splits" / f"{tag}.csv")))
    i = [x["image_id"] for x in rr]
    b = [(int(x["x"]), int(x["y"]), int(x["w"]), int(x["h"])) for x in rr]
    v = np.array([int(x["vehicle_id"]) for x in rr])
    c = np.array([int(x["camera_id"]) for x in rr])
    return i, b, v, c
f_qi, f_qb, f_qv, f_qc = load_falcon_split("val_query")
f_gi, f_gb, f_gv, f_gc = load_falcon_split("val_gallery")

def f_crop(iid, box):
    x, y, w, h = box
    return Image.open(str(FALCON_IMGS / f"{iid}.jpg")).convert("RGB").crop((x, y, x+w, y+h))

# ---- DN-ReID ----
DN_DATA = BASE / "datasets" / "dnreid_convert"
DN_IMGS = DN_DATA / "images"
dnr = list(csv.DictReader(open(DN_DATA / "train.csv")))
dn_iids = [r["image_id"] for r in dnr]
dn_veh = np.array([int(r["vehicle_id"]) for r in dnr])
uniq_dn = np.unique(dn_veh)
dn_v2c = {int(v): i for i,v in enumerate(uniq_dn)}
dn_y = np.array([dn_v2c[int(v)] for v in dn_veh])
dn_idx_by_cls = {c: np.where(dn_y == c)[0] for c in range(len(uniq_dn))}
n_dn_cls = len(uniq_dn)
print(f"DN-ReID: {len(dn_veh)} train, {n_dn_cls} classes", flush=True)

def dn_crop(iid):
    return Image.open(str(DN_IMGS / (iid + ".jpg"))).convert("RGB")

# ==================== MODEL ====================
backbone = DINOv3ML().to(DEVICE)
head = OptionCHead(1024, 512, len(LAYERS), n_f_cls, n_dn_cls).to(DEVICE)

# Warm-start from LLRD6b
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
for nm, mod in backbone.model.named_modules():
    if isinstance(mod, LoRALinear):
        a = ck["lora"].get(nm + ".lora_A")
        b = ck["lora"].get(nm + ".lora_B")
        if a is not None and b is not None:
            with torch.no_grad(): mod.lora_A.copy_(a); mod.lora_B.copy_(b)
if "blocks" in ck:
    for li_s, st in ck["blocks"].items():
        backbone.model.model.layer[int(li_s)].load_state_dict(st)
print(f"Loaded: LoRA + {len(ck.get('blocks',{}))} blocks from LLRD6b", flush=True)

# Init Option C head from LLRD6b AttnPool (single query → replicate to 6)
vh = ck["head"]
with torch.no_grad():
    for i in range(len(LAYERS)):
        head.queries[i].copy_(vh["query"])
    head.proj.weight.zero_()
    for i in range(len(LAYERS)):
        head.proj.weight[:, i*1024:(i+1)*1024].copy_(vh["proj.weight"] / len(LAYERS))
    head.proj.bias.copy_(vh["proj.bias"])
print("Option C head: init from LLRD6b AttnPool (replicated to 6 layers)", flush=True)

# Unfreeze blocks L16-23
LLR = {16:2.5e-6, 17:2.5e-6, 18:5e-6, 19:5e-6, 20:1e-5, 21:1e-5, 22:2e-5, 23:2e-5}
for li in sorted(LLR):
    for p in backbone.model.model.layer[li].parameters(): p.requires_grad_(True)

lora_params = [p for m in backbone.model.modules() if isinstance(m,LoRALinear) for p in m.parameters() if p.requires_grad]
lora_ids = {id(p) for p in lora_params}
all_groups = [
    {"params": lora_params, "lr": LR_LORA},
    {"params": list(head.parameters()), "lr": LR_HEAD},
    *[{"params": [p for p in backbone.model.model.layer[li].parameters() if id(p) not in lora_ids], "lr": LLR[li]} for li in sorted(LLR)],
]
opt = torch.optim.AdamW(all_groups, weight_decay=WD)
train_params = [p for g in all_groups for p in g["params"]]
ema_params = [p.detach().clone() for p in train_params]

def sched_fn(epoch):
    if epoch < 3: return 0.1 + 0.9 * (epoch+1)/3
    p = (epoch - 3) / max(1, EPOCHS - 3 - 1)
    return 0.5 * (1 + math.cos(math.pi * p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)

# ==================== EVAL ====================
def evaluate_falcon():
    backbone.eval(); head.eval()
    def embed(ii, bb):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(ii), 32):
                batch = [f_crop(ii[j], bb[j]) for j in range(i, min(i+32, len(ii)))]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e = head.embed(backbone(batch))
                outs.append(e.float().cpu().numpy())
        return np.concatenate(outs)
    eq = embed(f_qi, f_qb); eg = embed(f_gi, f_gb)
    S = eq @ eg.T; order = np.argsort(-S, kind="stable", axis=1)
    aps = []
    for i in range(len(f_qi)):
        qv_i, qc_i = f_qv[i], f_qc[i]
        jk = set()
        pos = 0
        for j in range(len(f_gi)):
            if f_gv[j] == qv_i and f_gc[j] == qc_i: jk.add(j)
            if f_gv[j] == qv_i and f_gc[j] != qc_i: pos += 1
        if pos == 0: continue
        seq = [int(j) for j in order[i] if int(j) not in jk]
        gt = np.array([1 if f_gv[j] == qv_i else 0 for j in seq])
        n_gt = int(gt.sum())
        if n_gt == 0: continue
        tp = np.cumsum(gt); prec = tp / np.arange(1, len(seq)+1)
        aps.append(float(np.sum(prec * gt) / min(n_gt, 10)))
    backbone.train(); head.train()
    return float(np.mean(aps))

# ==================== TRAINING ====================
n_steps_f = len(uniq_f_tr) // P
n_steps_dn = len(uniq_dn) // P
n_steps = max(n_steps_f, n_steps_dn)
rng = np.random.default_rng(SEED)
best_map = 0.0; step_cnt = 0; t0 = time.time()

for ep in range(EPOCHS):
    total_f, total_dn, n_f, n_dn = 0.0, 0.0, 0, 0
    for _ in range(n_steps):
        # Falcon batch (with FALCON_SHARE probability)
        if rng.random() < FALCON_SHARE:
            sel = rng.choice(len(uniq_f_tr), P, replace=False)
            idx = np.concatenate([rng.choice(f_idx_by_cls[int(c)], K, replace=len(f_idx_by_cls[int(c)])<K) for c in sel])
            batch = [f_crop(f_iids_tr[i], f_boxes_tr[i]) for i in idx]
            y = torch.tensor(f_y[idx], device=DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e = head.embed(backbone(batch))
                loss = F.cross_entropy(head.cls_falcon(e), y, label_smoothing=LS)
            total_f += float(loss); n_f += 1
        # DN-ReID batch
        else:
            sel = rng.choice(len(uniq_dn), P, replace=False)
            idx = np.concatenate([rng.choice(dn_idx_by_cls[int(c)], K, replace=len(dn_idx_by_cls[int(c)])<K) for c in sel])
            batch = [dn_crop(dn_iids[i]) for i in idx]
            y = torch.tensor(dn_y[idx], device=DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e = head.embed(backbone(batch))
                loss = F.cross_entropy(head.cls_dnreid(e), y, label_smoothing=LS)
            total_dn += float(loss); n_dn += 1
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        with torch.no_grad():
            for p_, e_ in zip(train_params, ema_params):
                e_.mul_(0.999).add_(p_, alpha=0.001)
        step_cnt += 1
    
    sched.step()
    f_avg = total_f / max(1, n_f)
    dn_avg = total_dn / max(1, n_dn)
    print(f"  ep{ep+1:2d} loss_f={f_avg:.4f} loss_dn={dn_avg:.4f} t={time.time()-t0:.0f}s", flush=True)
    
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        saved = [p.detach().clone() for p in train_params]
        with torch.no_grad():
            for p_, e_ in zip(train_params, ema_params): p_.copy_(e_)
        m = evaluate_falcon()
        print(f"  → Falcon mAP@10 = {m:.4f} {'★' if m>best_map else ''}", flush=True)
        with torch.no_grad():
            for p_, s_ in zip(train_params, saved): p_.copy_(s_)
        if m > best_map:
            best_map = m
            lora_save = {}
            for nm, mm in backbone.model.named_modules():
                if isinstance(mm, LoRALinear):
                    lora_save[nm+'.lora_A'] = mm.lora_A.data.cpu()
                    lora_save[nm+'.lora_B'] = mm.lora_B.data.cpu()
            torch.save({"lora": lora_save, "head": head.state_dict(),
                "blocks": {str(li): backbone.model.model.layer[li].state_dict() for li in sorted(LLR)},
                "epoch": ep, "mAP": m, "n_f_cls": n_f_cls, "n_dn_cls": n_dn_cls},
                OUT / f"{EXP}_best.pt")
            print(f"  → saved {EXP}_best.pt")

print(f"\nJoint training done. Best Falcon mAP@10 = {best_map:.4f}", flush=True)

# ==================== FALCON-ONLY FINISH ====================
print("\n=== Falcon-only finish ===", flush=True)
# Drop DN-ReID head, load best checkpoint, fine-tune on Falcon only
FALCON_FINISH_EP = 8
finish_lr = 3e-6

# Reload best checkpoint
ck_best = torch.load(OUT / f"{EXP}_best.pt", map_location="cpu", weights_only=True)
# Reset head for Falcon-only
head.cls_dnreid = nn.Linear(512, 1)  # dummy, not used
# Only optimize LoRA + blocks (no head for Falcon-only phase)
opt_finish = torch.optim.AdamW([{"params": lora_params, "lr": finish_lr}] + 
    [{"params": [p for p in backbone.model.model.layer[li].parameters() if id(p) not in lora_ids], "lr": finish_lr}
     for li in sorted(LLR)], weight_decay=WD)

for ep in range(FALCON_FINISH_EP):
    total = 0.0; n_b = 0
    for _ in range(n_steps_f):
        sel = rng.choice(len(uniq_f_tr), P, replace=False)
        idx = np.concatenate([rng.choice(f_idx_by_cls[int(c)], K, replace=len(f_idx_by_cls[int(c)])<K) for c in sel])
        batch = [f_crop(f_iids_tr[i], f_boxes_tr[i]) for i in idx]
        y = torch.tensor(f_y[idx], device=DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            e = head.embed(backbone(batch))
            loss = F.cross_entropy(head.cls_falcon(e), y, label_smoothing=LS)
        opt_finish.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt_finish.step()
        total += float(loss); n_b += 1
    print(f"  finish ep{ep+1} loss {total/n_b:.4f}", flush=True)
    
    saved = [p.detach().clone() for p in train_params]
    with torch.no_grad():
        for p_, e_ in zip(train_params, ema_params): p_.copy_(e_)
    m = evaluate_falcon()
    print(f"  → Falcon mAP@10 = {m:.4f}", flush=True)
    with torch.no_grad():
        for p_, s_ in zip(train_params, saved): p_.copy_(s_)
    if m > best_map:
        best_map = m
        torch.save({"lora": lora_save, "head": head.state_dict(),
            "blocks": {str(li): backbone.model.model.layer[li].state_dict() for li in sorted(LLR)},
            "epoch": ep, "mAP": m}, OUT / f"{EXP}_finish.pt")
        print(f"  → saved {EXP}_finish.pt")

print(f"\nFinal best Falcon mAP@10 = {best_map:.4f}", flush=True)
