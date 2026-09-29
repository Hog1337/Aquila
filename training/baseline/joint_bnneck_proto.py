#!/usr/bin/env python3
"""Joint DN+Falcon — BNNeck + Prototype Contrastive + SupCon.
CE через BN, SupCon+ProtoCon на сыром embedding."""
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
SPLITS = BASE / "reid" / "splits"
IMGS = BASE / "data" / "reid" / "images"
DEVICE = "cuda:0"
SIZE = 320
EPOCHS = 40
LAYERS = [16, 18, 20, 22, 23, 24]
P = 16; K = 4
LR_LORA = 3e-5; LR_HEAD = 3e-4; WD = 1e-4; SEED = 42; LS = 0.1
EXP = "JDNF_BNNECK_PROTO"
FALCON_SHARE = 0.30
LAMBDA_SC = 0.3   # SupCon weight
LAMBDA_PC = 0.15  # ProtoCon weight
TEMP = 0.07
MOMENTUM = 0.99   # Prototype momentum

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
    """Option C head + BNNeck: CE через BN, SupCon/ProtoCon на сырых фичах."""
    def __init__(self, tok_dim, proj_dim, n_layers, n_cls_falcon=0, n_cls_dn=0):
        super().__init__()
        self.n_layers = n_layers
        self.queries = nn.Parameter(torch.randn(n_layers, tok_dim) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(tok_dim) for _ in range(n_layers)])
        self.proj = nn.Linear(tok_dim * n_layers, proj_dim)
        # BNNeck: BN перед CE классификатором
        self.bn = nn.BatchNorm1d(proj_dim)
        self.cls_falcon = nn.Linear(proj_dim, n_cls_falcon)
        self.cls_dnreid = nn.Linear(proj_dim, n_cls_dn)
    
    def embed(self, tokens_list):
        """Возвращает raw embedding (до BN) — для SupCon, ProtoCon и inference."""
        pooled = []
        for i, tokens in enumerate(tokens_list):
            t = self.norms[i](tokens)
            t = F.normalize(t, dim=-1)
            q = F.normalize(self.queries[i], dim=-1)
            w = torch.softmax(t @ q / (t.shape[-1] ** 0.5), dim=1)
            pooled.append((t * w.unsqueeze(-1)).sum(1))
        fused = torch.cat(pooled, dim=-1)
        raw = F.normalize(self.proj(fused), dim=-1)
        return raw
    
    def embed_ce(self, tokens_list):
        """Возвращает embedding через BN — для CE."""
        raw = self.embed(tokens_list)
        return self.bn(raw)

# ==== SupCon Loss ====
def supcon_loss(embeddings, labels, temperature=0.07):
    B = embeddings.shape[0]
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T / temperature
    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask = pos_mask - torch.eye(B, device=embeddings.device)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim)
    sum_exp = exp_sim.sum(dim=1, keepdim=True)
    pos_exp = exp_sim * pos_mask
    pos_sum = pos_exp.sum(dim=1)
    n_pos = pos_mask.sum(dim=1)
    loss = -(pos_sum / (sum_exp.squeeze() + 1e-8)).log()
    loss = loss * (n_pos > 0).float()
    return loss.sum() / max((n_pos > 0).sum(), 1)

# ==== Prototype Contrastive Loss ====
class ProtoBank:
    """Momentum prototype bank for all classes."""
    def __init__(self, n_classes, dim, momentum=0.99, device="cuda:0"):
        self.n_classes = n_classes
        self.dim = dim
        self.momentum = momentum
        self.device = device
        # Initialize prototypes as zeros (will be warmed up)
        self.protos = torch.zeros(n_classes, dim, device=device)
        self.counts = torch.zeros(n_classes, device=device)
        self.initialized = torch.zeros(n_classes, dtype=torch.bool, device=device)
    
    @torch.no_grad()
    def update(self, embeddings, labels):
        """Update prototypes with momentum."""
        for i in range(labels.shape[0]):
            c = labels[i].item()
            emb = embeddings[i]
            if not self.initialized[c]:
                self.protos[c] = emb
                self.counts[c] = 1.0
                self.initialized[c] = True
            else:
                self.protos[c] = self.momentum * self.protos[c] + (1 - self.momentum) * emb
                self.counts[c] += 1.0
    
    def get_protos(self, labels):
        """Get prototypes for given labels."""
        return self.protos[labels]

def protocon_loss(embeddings, labels, protobank, temperature=0.07):
    """Contrast each sample against class prototypes."""
    embeddings = F.normalize(embeddings, dim=-1)
    protos = F.normalize(protobank.get_protos(labels), dim=-1)
    # Similarity between samples and their prototypes
    pos_sim = (embeddings * protos).sum(-1) / temperature  # [B]
    
    # Similarity between samples and ALL prototypes
    all_protos = F.normalize(protobank.protos, dim=-1)  # [C, D]
    neg_sim = embeddings @ all_protos.T / temperature  # [B, C]
    
    # Loss: push pos_sim up, neg_sim down
    # For each anchor: log(exp(pos) / sum(exp(all)))
    # Mask out the positive prototype
    labels_oh = F.one_hot(labels, num_classes=protobank.n_classes).float()
    neg_sim_masked = neg_sim - labels_oh * 1e9  # Remove positive prototype from denominator
    
    # Numerator: exp(pos_sim) 
    # Denominator: exp(pos_sim) + sum(exp(neg_sim_masked))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, C+1]
    # Positive is at index 0 for all samples
    loss = F.cross_entropy(logits, torch.zeros(labels.shape[0], dtype=torch.long, device=embeddings.device))
    
    return loss

# ==== TRAINING SETUP ====
# Data loading (same as before)
tr = list(csv.DictReader(open(SPLITS / "train_sub.csv")))
f_iids = [r["image_id"] for r in tr]
f_boxes = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in tr]
f_veh = np.array([int(r["vehicle_id"]) for r in tr])
uniq_f = np.unique(f_veh)
f_v2c = {int(v): i for i,v in enumerate(uniq_f)}
f_y = np.array([f_v2c[int(v)] for v in f_veh])
f_idx_by_cls = {c: np.where(f_y == c)[0] for c in range(len(uniq_f))}
n_f_cls = len(uniq_f)

def load_split(tag):
    rr = list(csv.DictReader(open(SPLITS / f"{tag}.csv")))
    i = [x["image_id"] for x in rr]
    b = [(int(x["x"]), int(x["y"]), int(x["w"]), int(x["h"])) for x in rr]
    v = np.array([int(x["vehicle_id"]) for x in rr])
    c = np.array([int(x["camera_id"]) for x in rr])
    return i, b, v, c
f_qi, f_qb, f_qv, f_qc = load_split("val_query")
f_gi, f_gb, f_gv, f_gc = load_split("val_gallery")

def f_crop(iid, box):
    x,y,w,h = box
    return Image.open(str(IMGS / f"{iid}.jpg")).convert("RGB").crop((x,y,x+w,y+h))

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
def dn_crop(iid):
    return Image.open(str(DN_IMGS / (iid + ".jpg"))).convert("RGB")

print(f"Falcon: {len(f_veh)} img, {n_f_cls} cls | DN: {len(dn_veh)} img, {n_dn_cls} cls", flush=True)

# ==== MODEL ====
backbone = DINOv3ML = type('D', (nn.Module,), {'__init__': lambda s: None})()
# Rebuild properly
from transformers import AutoModel as AM, AutoImageProcessor as AIP

class DINOv3ML(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = AIP.from_pretrained(str(DINO))
        self.model = AM.from_pretrained(str(DINO))
        self.model.requires_grad_(False)
        wrap(self.model, ["q_proj","k_proj","v_proj","o_proj"], 16, 32.0)
        self.targets = set(LAYERS)
        self._captured = {}
    def forward(self, pil_batch):
        self._captured = {}
        inputs = self.proc(images=pil_batch, size=(SIZE,SIZE), return_tensors="pt")
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

backbone = DINOv3ML().to(DEVICE)

# Head with BNNeck
head = HeadBNNeck(1024, 512, len(LAYERS), n_f_cls, n_dn_cls).to(DEVICE)

ck = torch.load(CKPT, map_location="cpu", weights_only=True)
for nm, mod in backbone.model.named_modules():
    if isinstance(mod, LoRALinear):
        a = ck["lora"].get(nm+".lora_A")
        b = ck["lora"].get(nm+".lora_B")
        if a is not None and b is not None:
            with torch.no_grad(): mod.lora_A.copy_(a); mod.lora_B.copy_(b)
if "blocks" in ck:
    for li_s, st in ck["blocks"].items():
        backbone.model.model.layer[int(li_s)].load_state_dict(st)
print(f"Loaded: LoRA + {len(ck.get('blocks',{}))} blocks", flush=True)

vh = ck["head"]
with torch.no_grad():
    for i in range(len(LAYERS)):
        head.queries[i].copy_(vh["queries"][i] if "queries" in vh else vh["query"])
    head.proj.weight.zero_()
    for i in range(len(LAYERS)):
        head.proj.weight[:, i*1024:(i+1)*1024].copy_(vh["proj.weight"] / len(LAYERS))
    head.proj.bias.copy_(vh["proj.bias"])
print("Head init from LLRD6b + BNNeck", flush=True)

LLR = {16:2.5e-6,17:2.5e-6,18:5e-6,19:5e-6,20:1e-5,21:1e-5,22:2e-5,23:2e-5}
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

# Prototype banks
proto_bank_f = ProtoBank(n_f_cls, 512, MOMENTUM, DEVICE)
proto_bank_dn = ProtoBank(n_dn_cls, 512, MOMENTUM, DEVICE)

def sched_fn(epoch):
    if epoch < 3: return 0.1 + 0.9 * (epoch+1)/3
    p = (epoch - 3) / max(1, EPOCHS - 3 - 1)
    return 0.5 * (1 + math.cos(math.pi * p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)

def evaluate_falcon():
    backbone.eval(); head.eval()
    def embed(ii, bb):
        outs = []
        with torch.inference_mode():
            for i in range(0,len(ii),32):
                batch = [f_crop(ii[j],bb[j]) for j in range(i,min(i+32,len(ii)))]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e = head.embed(backbone(batch))
                outs.append(e.float().cpu().numpy())
        return np.concatenate(outs)
    eq = embed(f_qi,f_qb); eg = embed(f_gi,f_gb)
    S = eq@eg.T; order = np.argsort(-S,kind="stable",axis=1)
    aps = []
    for i in range(len(f_qi)):
        qv_i,qc_i = f_qv[i],f_qc[i]
        jk=set(); pos=0
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

# ==== TRAIN ====
n_steps = max(len(uniq_f)//P, len(uniq_dn)//P)
rng = np.random.default_rng(SEED)
best_map = 0.0; t0 = time.time()

for ep in range(EPOCHS):
    total_f, total_dn, total_sc, total_pc, n_f, n_dn, nb = 0.0, 0.0, 0.0, 0.0, 0, 0, 0
    for _ in range(n_steps):
        if rng.random() < FALCON_SHARE:
            sel = rng.choice(len(uniq_f), P, replace=False)
            idx = np.concatenate([rng.choice(f_idx_by_cls[int(c)], K, replace=len(f_idx_by_cls[int(c)])<K) for c in sel])
            batch = [f_crop(f_iids[i], f_boxes[i]) for i in idx]
            y = torch.tensor(f_y[idx], device=DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = head.embed(backbone(batch))
                ce_input = head.bn(raw)
                loss_ce = F.cross_entropy(head.cls_falcon(ce_input), y, label_smoothing=LS)
                loss_sc = supcon_loss(raw, y, TEMP)
                loss_pc = protocon_loss(raw, y, proto_bank_f, TEMP)
                loss = loss_ce + LAMBDA_SC * loss_sc + (LAMBDA_PC * loss_pc if proto_bank_f.initialized.sum() > 10 else 0.0)
            # Update prototypes
            proto_bank_f.update(raw.detach(), y)
            total_f += float(loss_ce); total_sc += float(loss_sc); total_pc += float(loss_pc)
            n_f += 1
        else:
            sel = rng.choice(len(uniq_dn), P, replace=False)
            idx = np.concatenate([rng.choice(dn_idx_by_cls[int(c)], K, replace=len(dn_idx_by_cls[int(c)])<K) for c in sel])
            batch = [dn_crop(dn_iids[i]) for i in idx]
            y = torch.tensor(dn_y[idx], device=DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = head.embed(backbone(batch))
                ce_input = head.bn(raw)
                loss_ce = F.cross_entropy(head.cls_dnreid(ce_input), y, label_smoothing=LS)
                loss_sc = supcon_loss(raw, y, TEMP)
                loss_pc = protocon_loss(raw, y, proto_bank_dn, TEMP)
                loss = loss_ce + LAMBDA_SC * loss_sc + (LAMBDA_PC * loss_pc if proto_bank_f.initialized.sum() > 10 else 0.0)
            proto_bank_dn.update(raw.detach(), y)
            total_dn += float(loss_ce); total_sc += float(loss_sc); total_pc += float(loss_pc)
            n_dn += 1
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        with torch.no_grad():
            for p_, e_ in zip(train_params, ema_params):
                e_.mul_(0.999).add_(p_, alpha=0.001)
        nb += 1
    
    sched.step()
    sc_avg = total_sc / max(1, nb)
    pc_avg = total_pc / max(1, nb)
    print(f"  ep{ep+1:2d} loss_f={total_f/max(1,n_f):.4f} loss_dn={total_dn/max(1,n_dn):.4f} sc={sc_avg:.4f} pc={pc_avg:.4f} t={time.time()-t0:.0f}s", flush=True)
    
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
            torch.save({"lora":lora_save,"head":head.state_dict(),
                "blocks":{str(li):backbone.model.model.layer[li].state_dict() for li in sorted(LLR)},
                "epoch":ep,"mAP":m}, OUT/f"{EXP}_best.pt")
            print(f"  → saved {EXP}_best.pt", flush=True)

print(f"\nDone! Best Falcon mAP@10 = {best_map:.4f}", flush=True)
