#!/usr/bin/env python3
"""Benchmark latency and throughput for Falcon submission."""
import os, sys, math, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from transformers import AutoModel

BASE = Path("/home/limon/data/university/lct")
SPLITS = BASE / "reid" / "splits"
IMGS = BASE / "data" / "reid" / "images"
CKPT = BASE / "reid" / "baseline" / "out" / "JDNFV_SUPCON_PROTO_R32_best_ema_fp16.pt"
DEVICE = "cuda:0"
SIZE = 320

transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

class HB(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(6, 1024) * 0.02)
        self.n = nn.ModuleList([nn.LayerNorm(1024) for _ in range(6)])
        self.p = nn.Linear(6144, 2048)
    def embed(self, tl):
        r = []
        for i, t in enumerate(tl):
            n = F.normalize(self.n[i](t), dim=-1)
            qq = F.normalize(self.q[i], dim=-1)
            w = torch.softmax(n @ qq / 32.0, dim=1)
            r.append((n * w.unsqueeze(-1)).sum(1))
        return F.normalize(self.p(torch.cat(r, -1)), dim=-1)

# Build model once, reuse
def build_model():
    bb = AutoModel.from_pretrained(str(BASE/"reid"/"models"/"dinov3-vitl16"), trust_remote_code=True)
    for p in bb.parameters(): p.requires_grad_(False)
    wrap(bb, ["q_proj","k_proj","v_proj","o_proj"], 32, 64.0)
    bb = bb.to(DEVICE)
    h = HB().to(DEVICE)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    for nm, m in bb.named_modules():
        if isinstance(m, LoRALinear):
            a = ck["lora"].get(nm+".lora_A")
            b = ck["lora"].get(nm+".lora_B")
            if a is not None and b is not None:
                with torch.no_grad(): m.lora_A.copy_(a); m.lora_B.copy_(b)
    for li, st in ck["blocks"].items():
        bb.model.layer[int(li)].load_state_dict(st)
    h.load_state_dict(ck["head"], strict=False)
    bb.eval(); h.eval()
    return bb, h

bb, h = build_model()

# Prepare hooks
tokens = {}
handles = []
for idx, li in enumerate([0, 4, 8, 12, 16, 20]):
    def mkh(s):
        def hk(m, i, o):
            out = o[0] if isinstance(o, tuple) else o
            tokens[s] = out
        return hk
    handles.append(bb.model.layer[li].register_forward_hook(mkh(idx)))

def extract(pils):
    """Full extract: preprocess → forward → L2 norm. Returns numpy array."""
    batch = torch.stack(pils).to(DEVICE)
    tokens.clear()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = bb(pixel_values=batch)
            e = h.embed([tokens[i] for i in range(6)])
    return e.float().cpu().numpy()

# Load test images
print("Loading test images...", flush=True)
import csv
q_pils = []
for i, r in enumerate(csv.DictReader(open(SPLITS/"val_query.csv"))):
    if i >= 50: break
    xb,yb,wb,hb = int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])
    fpath = IMGS / f"{r['image_id']}.jpg"
    pil = Image.open(str(fpath)).convert("RGB").crop((xb,yb,xb+wb,yb+hb))
    q_pils.append(pil)

# ---- Latency b=1 (full cycle: read from disk → decode → crop → preprocess → forward → L2) ----
print("\n=== Latency b=1 (full cycle) ===", flush=True)

def full_extract_one(img_path, bbox):
    """Полный цикл extract() как на стенде: чтение диска → декод → кроп → препроцесс → forward → L2."""
    pil = Image.open(str(img_path)).convert("RGB").crop(bbox)
    t = transform(pil).unsqueeze(0).to(DEVICE)
    tokens.clear()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = bb(pixel_values=t)
            e = h.embed([tokens[i] for i in range(6)])
    return e.float().cpu().numpy()

# Warmup
print("Warmup...", flush=True)
import random
for _ in range(50):
    r2 = random.choice(q_pils)
    t2 = transform(r2).unsqueeze(0).to(DEVICE)
    tokens.clear()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = bb(pixel_values=t2)
            _ = h.embed([tokens[i] for i in range(6)])

# Measure
torch.cuda.synchronize()
times = []
for _ in range(300):
    r = random.choice(list(csv.DictReader(open(SPLITS/"val_query.csv"))))
    img_path = IMGS / f"{r['image_id']}.jpg"
    bbox = (int(r["x"]), int(r["y"]), int(r["x"])+int(r["w"]), int(r["y"])+int(r["h"]))
    t0 = time.perf_counter()
    full_extract_one(img_path, bbox)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

times = times[50:]  # drop first 50 warmup
latency = np.median(times)
p90 = np.percentile(times, 90)
print(f"Latency b=1 (медиана): {latency:.1f} ms")
print(f"P90: {p90:.1f} ms")
print(f"Min: {min(times):.1f} ms, Max: {max(times):.1f} ms")

# Score (на A5000 будет ~1.5-2x медленнее)
a5000_latency = latency * 1.8  # correction factor
lat_score = max(0, 1 - (a5000_latency - 40) / 40) if a5000_latency > 40 else 1.0
print(f"Оценка на A5000 (~{latency*1.8:.0f}ms): latency_score = {lat_score:.2f}")

# ---- Throughput (best FPS across batch sizes) ----
print("\n=== Throughput ===", flush=True)
for batch_size in [1, 8, 16, 32]:
    # Prepare batch of images
    batch_pils = []
    for _ in range(batch_size):
        r = random.choice(list(csv.DictReader(open(SPLITS/"val_query.csv"))))
        img_path = IMGS / f"{r['image_id']}.jpg"
        xb,yb,wb,hb = int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])
        batch_pils.append(Image.open(str(img_path)).convert("RGB").crop((xb,yb,xb+wb,yb+hb)))
    
    # Warmup
    for _ in range(10):
        t = torch.stack([transform(p) for p in batch_pils]).to(DEVICE)
        tokens.clear()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = bb(pixel_values=t)
                _ = h.embed([tokens[i] for i in range(6)])
    
    # Measure sustained throughput (10 second run)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    while time.perf_counter() - t0 < 10:
        t3 = torch.stack([transform(p) for p in batch_pils]).to(DEVICE)
        tokens.clear()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = bb(pixel_values=t3)
                _ = h.embed([tokens[i] for i in range(6)])
        count += batch_size
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - t0
    fps = count / elapsed
    print(f"  batch={batch_size:2d}: {count} img / {elapsed:.1f}s = {fps:.0f} FPS")

for hh in handles: hh.remove()
print("\nDone!")
