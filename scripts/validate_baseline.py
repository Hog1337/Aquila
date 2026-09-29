#!/usr/bin/env python3
"""Валидация без маскирования (через embed-by-key-batch) для сравнения."""
import csv, json, sys, time
from pathlib import Path
import httpx, numpy as np

INFERENCE_URL = "http://localhost:8001"
SPLITS_DIR = Path("/home/limon/data/university/lct/reid/splits")
CLEANING_CSV = Path("/home/limon/data/university/lct/reid/cleaning_results.csv")
BATCH_SIZE = 128

def load_split(name):
    rows = list(csv.DictReader(open(SPLITS_DIR / f"{name}.csv")))
    return rows

def load_excluded():
    excluded = set()
    if CLEANING_CSV.exists():
        for row in csv.DictReader(open(CLEANING_CSV)):
            excluded.add(row["image_id"])
    return excluded

def embed_batch(http, specs):
    all_embs = []
    for start in range(0, len(specs), BATCH_SIZE):
        batch = specs[start:start+BATCH_SIZE]
        resp = http.post(f"{INFERENCE_URL}/internal/embed-by-key-batch",
            data={"keys": json.dumps(batch)}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        all_embs.extend(data["embeddings"])
        print(f"  батч {start//BATCH_SIZE+1}/{(len(specs)-1)//BATCH_SIZE+1}: {len(batch)} шт, {data['elapsed_ms']}ms", flush=True)
    return all_embs

def compute_map(q_embs, q_veh, g_embs, g_veh, top_n=10):
    g_np = np.array(g_embs, dtype=np.float32)
    aps = []
    for qe, qv in zip(q_embs, q_veh):
        sim = g_np @ qe
        top_idx = np.argsort(sim)[::-1][:top_n]
        n_pos = sum(1 for v in g_veh if v == qv)
        if n_pos == 0: continue
        ap = 0.0; n_hit = 0
        for rank, idx in enumerate(top_idx, 1):
            if g_veh[idx] == qv:
                n_hit += 1
                ap += n_hit / rank
        aps.append(ap / min(n_pos, top_n))
    return float(np.mean(aps)) if aps else 0.0

def main():
    http = httpx.Client(timeout=300)
    r = http.get(f"{INFERENCE_URL}/health"); r.raise_for_status()
    h = r.json()
    print(f"Модель: {h['model_version']}, YOLO: {h['yolo_loaded']}")

    val_q = load_split("val_query")
    val_g = load_split("val_gallery")
    excluded = load_excluded()
    val_q = [r for r in val_q if r["image_id"] not in excluded]
    val_g = [r for r in val_g if r["image_id"] not in excluded]
    print(f"Query: {len(val_q)}, Gallery: {len(val_g)}")

    q_specs = [{"image_id":r["image_id"],"x":int(r["x"]),"y":int(r["y"]),"w":int(r["w"]),"h":int(r["h"])} for r in val_q]
    g_specs = [{"image_id":r["image_id"],"x":int(r["x"]),"y":int(r["y"]),"w":int(r["w"]),"h":int(r["h"])} for r in val_g]

    print("\n🚀 Query...")
    t0 = time.time()
    q_embs = embed_batch(http, q_specs)
    print(f"   done in {time.time()-t0:.1f}s")

    print("\n🚀 Gallery...")
    t0 = time.time()
    g_embs = embed_batch(http, g_specs)
    print(f"   done in {time.time()-t0:.1f}s")

    q_np = [np.array(e, dtype=np.float32) for e in q_embs]
    g_np = [np.array(e, dtype=np.float32) for e in g_embs]
    q_veh = [r["vehicle_id"] for r in val_q]
    g_veh = [r["vehicle_id"] for r in val_g]

    map10 = compute_map(q_np, q_veh, g_np, g_veh, top_n=10)
    print(f"\n{'='*50}")
    print(f"  🏆 mAP@10 (crop, без маски) = {map10:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
