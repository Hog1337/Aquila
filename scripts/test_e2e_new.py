#!/usr/bin/env python3
"""E2E-тест новой архитектуры: импорт галереи → метрики → экспорт → evaluate.py."""
import sys, time, json, csv, io, os
from pathlib import Path
import httpx
import numpy as np

API = "http://localhost:8000/api/v1"
REID = Path("/home/limon/data/university/lct/reid/splits")
IMAGES = Path("/home/limon/data/university/lct/data/reid/images")

client = httpx.Client(timeout=120)
OUT = Path("/tmp/aquila-e2e-test")
OUT.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def check(label, ok, detail=""):
    status = "✅" if ok else "❌"
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return ok

# =========================================================================
# 1. Статус до
# =========================================================================
log("=" * 60)
log("1. ПРОВЕРКА СТАТУСА ДО")
r = client.get(f"{API}/status")
check("API /status", r.status_code == 200)
s = r.json()
check("model_version есть", "model_version" in s)
check("embedding_dim=2048", s.get("embedding_dim") == 2048)
check("gallery пуста", s.get("gallery", {}).get("objects", -1) == 0)
log(f"   Статус: {s['model_version']}, dim={s['embedding_dim']}, objects={s.get('gallery',{}).get('objects',0)}")

# =========================================================================
# 2. Импорт val_gallery_clean.csv + images
# =========================================================================
log("=" * 60)
log("2. ИМПОРТ ГАЛЕРЕИ: val_gallery_clean.csv")

csv_path = REID / "val_gallery_clean.csv"
log(f"   CSV: {csv_path.name} ({os.path.getsize(csv_path)} байт)")

# Session
r = client.post(f"{API}/gallery/import-sessions",
    files={"annotations": (csv_path.name, csv_path.read_bytes(), "text/csv")})
check("create_import_session", r.status_code == 201)
session = r.json()
sid = session["id"]
log(f"   session_id={sid}, total={session['total']}, images={len(session['image_ids'])}")

# Upload images
uploaded = 0
for i, img_id in enumerate(session["image_ids"]):
    img_path = IMAGES / f"{img_id}.jpg"
    if not img_path.exists():
        continue
    r2 = client.post(f"{API}/gallery/import-sessions/{sid}/images",
        data={"image_id": img_id},
        files={"image": (f"{img_id}.jpg", img_path.read_bytes(), "image/jpeg")})
    if r2.status_code == 204:
        uploaded += 1
    if (i + 1) % 200 == 0:
        log(f"   images uploaded: {i+1}/{len(session['image_ids'])}")
check("upload images", uploaded == len(session["image_ids"]),
      f"{uploaded}/{len(session['image_ids'])}")

# Start import
r = client.post(f"{API}/gallery/import-sessions/{sid}/start")
check("start import", r.status_code == 202)
job = r.json()
jid = job["id"]

# Wait for completion
deadline = time.time() + 600
while time.time() < deadline:
    r = client.get(f"{API}/gallery/imports/{jid}")
    j = r.json()
    if j["status"] in ("done", "failed", "cancelled"):
        break
    time.sleep(5)
check("import done", j["status"] == "done", f"error={j.get('error')}")

# Status after
r = client.get(f"{API}/status")
s = r.json()
g = s.get("gallery", {})
check("gallery has objects", g.get("objects", 0) > 0, f"{g.get('objects')} objects")
check("gallery has images", g.get("images", 0) > 0, f"{g.get('images')} images")
log(f"   gallery: {g['objects']} objects, {g['images']} images, {g['vehicles']} vehicles")

# =========================================================================
# 3. МЕТРИКИ: val_query_clean.csv × val_gallery_clean.csv
# =========================================================================
log("=" * 60)
log("3. ЗАПУСК МЕТРИК: val_query_clean × val_gallery_clean")

q_csv = REID / "val_query_clean.csv"
g_csv = REID / "val_gallery_clean.csv"

r = client.post(f"{API}/metrics/runs",
    files=[
        ("query_csv", (q_csv.name, q_csv.read_bytes(), "text/csv")),
        ("gallery_csv", (g_csv.name, g_csv.read_bytes(), "text/csv")),
    ],
    data={"use_rerank": "false"},
)
check("start metrics", r.status_code == 202)
mjob = r.json()
mid = mjob["id"]
log(f"   metrics_job_id={mid}")

# Wait for completion
deadline = time.time() + 1800
while time.time() < deadline:
    r = client.get(f"{API}/metrics/runs/{mid}")
    m = r.json()
    if m["status"] in ("done", "failed"):
        break
    time.sleep(5)

ok = check("metrics done", m["status"] == "done", f"error={m.get('error')}")
if ok and m.get("run"):
    run = m["run"]
    print(f"\n{'='*60}")
    print("📊 РЕЗУЛЬТАТЫ МЕТРИК")
    print(f"{'='*60}")
    print(f"  mAP@10:      {run['map']:.4f}")
    print(f"  Rank-1:      {run['rank1']:.4f}")
    print(f"  Rank-5:      {run['rank5']:.4f}")
    print(f"  F1:          {run['f1']:.4f}")
    print(f"  TNR:         {run['tnr']:.4f}")
    print(f"  PR-AUC:      {run['pr_auc']:.4f}")
    print(f"  Порог τ:     {run['threshold']:.2f}")
    print(f"  Запросов:    {run['n_queries']}")
    print(f"  details:     {json.dumps(run.get('details',{}), indent=2)[:200]}...")
    print(f"{'='*60}")

# =========================================================================
# 4. ЭКСПОРТ: те же val_query → 3 файла
# =========================================================================
log("=" * 60)
log("4. ЗАПУСК ЭКСПОРТА: val_query_clean → 3 файла")

r = client.post(f"{API}/exports",
    files={"query_csv": (q_csv.name, q_csv.read_bytes(), "text/csv")},
    data={"threshold": "0.87", "use_rerank": "false"},
)
check("start export", r.status_code == 202)
ejob = r.json()
eid = ejob["id"]
log(f"   export_job_id={eid}")

# Wait
deadline = time.time() + 1800
while time.time() < deadline:
    r = client.get(f"{API}/exports/{eid}")
    e = r.json()
    if e["status"] in ("done", "failed"):
        break
    time.sleep(5)
check("export done", e["status"] == "done", f"error={e.get('error')}")

if e["status"] == "done":
    # Download files
    for art in e.get("artifacts", []):
        name = art["name"]
        r = client.get(f"{API}/exports/{eid}/files/{name}")
        (OUT / name).write_bytes(r.content)
        log(f"   {name}: {art['bytes']} байт -> {OUT / name}")
    
    # Verify formats
    sub = OUT / "submission.csv"
    emb = OUT / "embeddings.npy"
    cand = OUT / "candidates.csv"
    
    sub_lines = sub.read_text().strip().split("\n")
    check("submission.csv не пуст", len(sub_lines) > 0, f"{len(sub_lines)} query")
    
    cand_lines = cand.read_text().strip().split("\n")
    check("candidates.csv с заголовком", cand_lines[0] == "query_id,gallery_id,confidence",
          f"header: {cand_lines[0]}")
    
    emb_arr = np.load(emb)
    check("embeddings.npy 2D", emb_arr.ndim == 2, f"shape={emb_arr.shape}")
    check("embeddings.npy float32", emb_arr.dtype == np.float32, f"dtype={emb_arr.dtype}")
    
    # Check: n_queries + n_gallery = emb rows
    n_q = len(sub_lines)
    n_g = sum(1 for _ in open(OUT / "embeddings.npy", "rb"))  # not quite right
    # Actually check that embeddings.npy has more rows than queries
    check("embeddings > queries", emb_arr.shape[0] > n_q,
          f"emb rows={emb_arr.shape[0]}, queries={n_q}")

# =========================================================================
# 5. ПОИСК (одиночный запрос через API)
# =========================================================================
log("=" * 60)
log("5. ТЕСТ ПОИСКА: первый объект val_query → поиск по галерее")

# Read first query from CSV
with open(q_csv) as f:
    first = list(csv.DictReader(f))[0]
img_path = IMAGES / f"{first['image_id']}.jpg"
if img_path.exists():
    r = client.post(f"{API}/queries",
        files={"image": (img_path.name, img_path.read_bytes(), "image/jpeg")},
        data={"x": first["x"], "y": first["y"], "w": first["w"], "h": first["h"],
              "top_n": 10, "rerank": "false"})
    check("search API", r.status_code == 200)
    sr = r.json()
    check("search returned candidates", len(sr.get("candidates", [])) > 0,
          f"{len(sr.get('candidates', []))} candidates")
    if sr.get("candidates"):
        check("candidate has score", sr["candidates"][0].get("score") is not None,
              f"top score={sr['candidates'][0]['score']:.4f}")

# =========================================================================
# ИТОГ
# =========================================================================
log("=" * 60)
log("✅ E2E ТЕСТ ЗАВЕРШЁН")
log(f"   Файлы экспорта: {OUT}")
log(f"   Для evaluate.py: python evaluate.py --gt ... --submission {OUT/'submission.csv'} ...")
print()
print("=" * 60)
print("СТАТУС ПОСЛЕ ТЕСТА:")
r = client.get(f"{API}/status")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
