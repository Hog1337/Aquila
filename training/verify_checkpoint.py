#!/usr/bin/env python3
"""Verify JDNFV_SUPCON_PROTO_R32_best_ema_fp16.pt mAP@10 on clean val split.
Uses the same eval logic as evaluate.py (official Falcon protocol)."""
import os, sys, csv, time, math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoImageProcessor

BASE = Path("/home/limon/data/university/lct")
DINO = BASE / "reid" / "models" / "dinov3-vitl16"
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "reid" / "baseline" / "out" / "JDNFV_SUPCON_PROTO_R32_best_ema_fp16.pt"
DEVICE = "cuda:0"
SIZE = 320

print(f"Checkpoint: {CKPT}")
print(f"Checkpoint size: {CKPT.stat().st_size / 1e6:.1f} MB", flush=True)

# ======== LoRA ========
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
    def __init__(self, tok_dim, proj_dim, n_layers):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_layers, tok_dim) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(tok_dim) for _ in range(n_layers)])
        self.proj = nn.Linear(tok_dim * n_layers, proj_dim)
        self.bn = nn.BatchNorm1d(proj_dim)
    def embed(self, tl):
        r = []
        for i, t in enumerate(tl):
            n = F.normalize(self.norms[i](t), dim=-1)
            q = F.normalize(self.queries[i], dim=-1)
            w = torch.softmax(n @ q / (n.shape[-1] ** 0.5), dim=1)
            r.append((n * w.unsqueeze(-1)).sum(1))
        return F.normalize(self.proj(torch.cat(r, -1)), dim=-1)

def hook_tokens(m, i, o, store, idx):
    store[idx] = o[0] if isinstance(o, tuple) else o

# ======== Load model ========
print("Loading model...", flush=True)
backbone = AutoModel.from_pretrained(str(DINO), trust_remote_code=True)
backbone.requires_grad_(False)
wrap(backbone, ["q_proj", "k_proj", "v_proj", "o_proj"], 32, 64.0)
backbone = backbone.to(DEVICE)
backbone.eval()

head = HeadBNNeck(1024, 2048, 6).to(DEVICE)
head.eval()

ck = torch.load(CKPT, map_location="cpu", weights_only=True)
print(f"Stored mAP in checkpoint: {ck.get('mAP', 'N/A')}, ema={ck.get('ema', 'N/A')}", flush=True)

# Load LoRA
n_lora = 0
for nm, mod in backbone.named_modules():
    if isinstance(mod, LoRALinear):
        a = ck["lora"].get(nm + ".lora_A")
        b = ck["lora"].get(nm + ".lora_B")
        if a is not None and b is not None:
            with torch.no_grad(): mod.lora_A.copy_(a); mod.lora_B.copy_(b)
            n_lora += 1
print(f"LoRA modules loaded: {n_lora}/96", flush=True)

# Load blocks
for li_s, st in ck["blocks"].items():
    backbone.model.layer[int(li_s)].load_state_dict(st)
print(f"Blocks loaded: {len(ck['blocks'])} ({sorted(ck['blocks'].keys())})", flush=True)

# Load head
head.load_state_dict(ck["head"], strict=False)
print("Head loaded", flush=True)

# Register hooks on layers [0, 4, 8, 12, 16, 20]
tokens = {}
handles = []
for li in range(6):
    layer = backbone.model.layer[li * 4]
    handles.append(layer.register_forward_hook(
        lambda m, i, o, s=li: hook_tokens(m, i, o, tokens, s)))

# ======== Data ========
# Use CLEAN splits (with excluded bad frames)
q_csv = list(csv.DictReader(open(BASE / "reid" / "splits" / "val_query_clean.csv")))
g_csv = list(csv.DictReader(open(BASE / "reid" / "splits" / "val_gallery_clean.csv")))

IMGS = BASE / "data" / "reid" / "images"

print(f"Query: {len(q_csv)} gallery: {len(g_csv)}", flush=True)

qi = [r["image_id"] for r in q_csv]
qb = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in q_csv]
qv = np.array([int(r["vehicle_id"]) for r in q_csv])
qc = np.array([int(r["camera_id"]) for r in q_csv])

gi = [r["image_id"] for r in g_csv]
gb = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in g_csv]
gv = np.array([int(r["vehicle_id"]) for r in g_csv])
gc = np.array([int(r["camera_id"]) for r in g_csv])

import torchvision.transforms as T
transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def embed_batch(ids, boxes, tag):
    outs = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(ids), 64):
            pils = []
            for j in range(i, min(i + 64, len(ids))):
                x, y, w, h = boxes[j]
                pil = Image.open(IMGS / f"{ids[j]}.jpg").convert("RGB").crop((x, y, x + w, y + h))
                pils.append(pil)
            batch = torch.stack([transform(p) for p in pils]).to(DEVICE)
            tokens.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = backbone(pixel_values=batch)
                r = head.embed([tokens[k] for k in range(6)])
            outs.append(r.float().cpu().numpy())
    e = np.concatenate(outs)
    print(f"{tag}: {e.shape} in {time.time()-t0:.0f}s", flush=True)
    return e

eq = embed_batch(qi, qb, "query")
eg = embed_batch(gi, gb, "gallery")

# L2 normalize
eq = eq / np.clip(np.linalg.norm(eq, axis=1, keepdims=True), 1e-12, None)
eg = eg / np.clip(np.linalg.norm(eg, axis=1, keepdims=True), 1e-12, None)

# ======== Official Falcon mAP@10 ========
# Junk filter: remove same vehicle_id AND same camera_id before top-10
# AP normalized by min(n_pos, 10)
S = eq @ eg.T
order = np.argsort(-S, kind="stable", axis=1)

aps = []
n_no_pair = 0
n_total = len(eq)

for i in range(len(eq)):
    qv_i, qc_i = qv[i], qc[i]
    # junk = same vehicle + same camera (remove from gallery BEFORE top-10)
    junk_mask = (gv == qv_i) & (gc == qc_i)
    # valid positives = same vehicle, different camera
    pos_mask = (gv == qv_i) & (gc != qc_i)
    n_pos = int(pos_mask.sum())
    
    if n_pos == 0:
        n_no_pair += 1
        continue  # open-set query, excluded from mAP
    
    # Build ranking without junk
    seq = [j for j in order[i] if not junk_mask[j]]
    top10 = seq[:10]
    rel = np.array([1 if gv[j] == qv_i else 0 for j in top10])
    
    if rel.sum() == 0:
        aps.append(0.0)
    else:
        cum = np.cumsum(rel)
        prec = cum / (np.arange(len(top10)) + 1)
        ap = float((prec * rel).sum() / min(n_pos, 10))
        aps.append(ap)

mAP = np.mean(aps) if aps else 0.0
print(f"\n========== РЕЗУЛЬТАТ ==========")
print(f"Total queries: {n_total}")
print(f"With pair (scored): {len(aps)}")
print(f"Without pair (open-set): {n_no_pair}")
print(f"mAP@10 (official): {mAP:.4f}")
print(f"Checkpoint stored mAP: {ck.get('mAP', 'N/A'):.4f}")
print(f"Delta: {mAP - ck.get('mAP', 0):+.4f}")
print(f"================================", flush=True)

# Also compute Rank-1
r1s = []
for i in range(len(eq)):
    qv_i, qc_i = qv[i], qc[i]
    junk_mask = (gv == qv_i) & (gc == qc_i)
    pos_mask = (gv == qv_i) & (gc != qc_i)
    n_pos = int(pos_mask.sum())
    if n_pos == 0: continue
    seq = [j for j in order[i] if not junk_mask[j]]
    r1s.append(1 if len(seq) > 0 and gv[seq[0]] == qv_i else 0)
print(f"Rank-1: {np.mean(r1s):.4f}", flush=True)

# Cleanup
for h in handles: h.remove()
del backbone, head
torch.cuda.empty_cache()
