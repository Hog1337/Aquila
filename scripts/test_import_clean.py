#!/usr/bin/env python3
"""Импорт чистых (clean) версий val_gallery + val_query через API, запуск метрик."""
import sys, time, json, csv
from pathlib import Path
import httpx

API = "http://localhost:8000/api/v1"
REID_SPLITS = Path("/home/limon/data/university/lct/reid/splits")
IMAGES_DIR = Path("/home/limon/data/university/lct/data/reid/images")

client = httpx.Client(timeout=120)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def create_import_session(csv_path, role):
    log(f"Создаю сессию: {csv_path.name} role={role}")
    resp = client.post(f"{API}/gallery/import-sessions",
        files={"annotations": (csv_path.name, csv_path.read_bytes(), "text/csv")},
        data={"role": role})
    if resp.status_code != 201:
        raise RuntimeError(f"create_import_session: {resp.status_code} {resp.text}")
    s = resp.json()
    log(f"  session_id={s['id']}, total={s['total']}, images={len(s['image_ids'])}")
    return s

def upload_images(session_id, image_ids):
    uploaded = 0
    for i, img_id in enumerate(image_ids):
        img_path = IMAGES_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        data = img_path.read_bytes()
        resp = client.post(f"{API}/gallery/import-sessions/{session_id}/images",
            data={"image_id": img_id},
            files={"image": (f"{img_id}.jpg", data, "image/jpeg")})
        if resp.status_code != 204:
            raise RuntimeError(f"upload {img_id}: {resp.status_code} {resp.text}")
        uploaded += 1
        if (i+1) % 300 == 0 or i == len(image_ids)-1:
            log(f"  images: {i+1}/{len(image_ids)} (uploaded={uploaded})")
    return uploaded

def start_and_poll(session_id):
    log("  Запускаю обработку...")
    resp = client.post(f"{API}/gallery/import-sessions/{session_id}/start")
    if resp.status_code != 202:
        raise RuntimeError(f"start: {resp.status_code} {resp.text}")
    job = resp.json()
    log(f"  job_id={job['id']}")
    deadline = time.time() + 1800
    while time.time() < deadline:
        resp = client.get(f"{API}/gallery/imports/{job['id']}")
        j = resp.json()
        if j["status"] in ("done", "failed", "cancelled"):
            log(f"  статус: {j['status']}, error={j.get('error')}")
            return j
        if j.get("progress"):
            log(f"  прогресс: {j['progress']}%")
        time.sleep(5)
    raise TimeoutError("Import timeout")

def run_metrics():
    log("="*60)
    log("Запускаю метрики...")
    resp = client.post(f"{API}/metrics/runs")
    if resp.status_code != 202:
        raise RuntimeError(f"start_metrics: {resp.status_code} {resp.text}")
    job = resp.json()
    log(f"  metric_job_id={job['id']}")
    deadline = time.time() + 3600
    while time.time() < deadline:
        resp = client.get(f"{API}/metrics/runs/{job['id']}")
        d = resp.json()
        if d["status"] == "done":
            return d.get("run", {})
        elif d["status"] == "failed":
            raise RuntimeError(f"Metrics failed: {d.get('error')}")
        log(f"  прогресс метрик: {d.get('progress',0)}%")
        time.sleep(5)
    raise TimeoutError("Metrics timeout")

t_start = time.time()

# 1. Import clean val_gallery
log("="*60)
log("ШАГ 1: val_gallery_clean (role=val_gallery)")
s1 = create_import_session(REID_SPLITS / "val_gallery_clean.csv", "val_gallery")
u1 = upload_images(s1["id"], s1["image_ids"])
r1 = start_and_poll(s1["id"])

# 2. Import clean val_query
log("="*60)
log("ШАГ 2: val_query_clean (role=val_query)")
s2 = create_import_session(REID_SPLITS / "val_query_clean.csv", "val_query")
u2 = upload_images(s2["id"], s2["image_ids"])
r2 = start_and_poll(s2["id"])

# 3. Status
log("="*60)
resp = client.get(f"{API}/status")
st = resp.json()
log(f"Статус: {st['gallery']}")

# 4. Metrics
metrics = run_metrics()

elapsed = time.time() - t_start
log("="*60)
log(f"ИТОГО: {elapsed:.0f}с ({elapsed/60:.1f} мин)")

print()
print("="*70)
print("РЕЗУЛЬТАТЫ МЕТРИК НА ЧИСТЫХ ДАННЫХ (CLEAN)")
print("="*70)
if metrics:
    print(f"  mAP@10:      {metrics['map']:.4f}")
    print(f"  Rank-1:      {metrics['rank1']:.4f}")
    print(f"  Rank-5:      {metrics['rank5']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  TNR:         {metrics['tnr']:.4f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  Порог τ:     {metrics['threshold']:.2f}")
    print(f"  Запросов:    {metrics['n_queries']}")
    print(f"  Детали:      {json.dumps(metrics.get('details',{}))}")
print("="*70)
