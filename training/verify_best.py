#!/usr/bin/env python3
"""Independent verification of best checkpoint mAP."""
import sys, csv, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

BASE = Path("/home/a.a.milkevich/temp_exp")
sys.path.insert(0, str(BASE / "reid" / "baseline"))
from embed_v2 import V2Embedder

SPLITS = BASE / "reid" / "splits"
IMGS = BASE / "data" / "reid" / "images"
CKPT = BASE / "reid" / "baseline" / "out" / "JDNFV_MASKED_FT_best.pt"
device = "cuda"

# Load excluded frames (same as training eval)
excluded = set()
with open(BASE / "reid" / "cleaning_results.csv") as f:
    for row in csv.DictReader(f):
        excluded.add((row["vehicle_id"], row["image_id"], row["src"]))

vq = [r for r in csv.DictReader(open(SPLITS / "val_query.csv"))
      if (r["vehicle_id"], r["image_id"], "val_q") not in excluded]
vg = [r for r in csv.DictReader(open(SPLITS / "val_gallery.csv"))
      if (r["vehicle_id"], r["image_id"], "val_g") not in excluded]
print(f"Q: {len(vq)}, G: {len(vg)}", flush=True)

transform = T.Compose([
    T.Resize((320, 320)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def crop_t(r):
    x,y,w,h = int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])
    return transform(Image.open(IMGS / f"{r['image_id']}.jpg").convert("RGB").crop((x,y,x+w,y+h)))

embedder = V2Embedder(str(CKPT))

BS = 64
t0 = time.time()

q_embs = []
for i in range(0, len(vq), BS):
    batch = torch.stack([crop_t(r) for r in vq[i:i+BS]]).to(device)
    embedder.tokens.clear()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _ = embedder.backbone(pixel_values=batch)
        r = embedder.head.embed([embedder.tokens[k] for k in range(6)])
    q_embs.append(r.float().detach().cpu().numpy())
eq = np.concatenate(q_embs)
print(f"Q: {eq.shape} {time.time()-t0:.1f}s", flush=True)

g_embs = []
for i in range(0, len(vg), BS):
    batch = torch.stack([crop_t(r) for r in vg[i:i+BS]]).to(device)
    embedder.tokens.clear()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _ = embedder.backbone(pixel_values=batch)
        r = embedder.head.embed([embedder.tokens[k] for k in range(6)])
    g_embs.append(r.float().detach().cpu().numpy())
eg = np.concatenate(g_embs)
print(f"G: {eg.shape} {time.time()-t0:.1f}s", flush=True)

eq = eq / np.linalg.norm(eq, axis=1, keepdims=True)
eg = eg / np.linalg.norm(eg, axis=1, keepdims=True)

S = eq @ eg.T
order = np.argsort(-S, kind="stable", axis=1)

aps = []
for i in range(len(vq)):
    qv, qc = int(vq[i]["vehicle_id"]), int(vq[i]["camera_id"])
    jk, pos = set(), 0
    for j in range(len(vg)):
        gv, gc = int(vg[j]["vehicle_id"]), int(vg[j]["camera_id"])
        if gv == qv and gc == qc: jk.add(j)
        if gv == qv and gc != qc: pos += 1
    if pos == 0: continue
    seq = [int(j) for j in order[i] if int(j) not in jk][:10]
    gt = np.array([1 if int(vg[j]["vehicle_id"]) == qv else 0 for j in seq])
    n_gt = int(gt.sum())
    if n_gt > 0:
        tp = np.cumsum(gt)
        prec = tp / np.arange(1, len(seq) + 1)
        aps.append(float(np.sum(prec * gt) / min(pos, 10)))
    else:
        aps.append(0.0)

mAP = float(np.mean(aps))
print(f"\n{'='*50}")
print(f"  mAP@10 (verified): {mAP:.4f}")
print(f"  Valid queries: {len(aps)}/{len(vq)}")
print(f"{'='*50}")

embedder.cleanup()
