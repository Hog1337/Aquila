#!/usr/bin/env python3
"""Валидация модели на val сплите: вычисление mAP@10.

Запуск:
  cd /home/limon/data/university/lct/aquila && python3 scripts/validate.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

INFERENCE_URL = "http://localhost:8001"
SPLITS_DIR = Path("/home/limon/data/university/lct/reid/splits")
CLEANING_CSV = Path("/home/limon/data/university/lct/reid/cleaning_results.csv")

BATCH_SIZE = 64  # размер одного батча для embed-masked-by-key-batch


def load_split(name: str) -> list[dict]:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        print(f"❌ Сплит не найден: {path}")
        sys.exit(1)
    rows = list(csv.DictReader(open(path)))
    print(f"📄 {name}: {len(rows)} строк")
    return rows


def load_excluded() -> set[str]:
    """Загружает список image_id, которые нужно исключить."""
    excluded = set()
    if CLEANING_CSV.exists():
        for row in csv.DictReader(open(CLEANING_CSV)):
            excluded.add(row["image_id"])
        print(f"🚫 Исключаем {len(excluded)} мусорных кадров")
    return excluded


def embed_batch(http: httpx.Client, specs: list[dict]) -> list[list[float]]:
    """Получить эмбеддинги для списка {image_id, x, y, w, h}."""
    all_embs = []
    for start in range(0, len(specs), BATCH_SIZE):
        batch = specs[start:start + BATCH_SIZE]
        resp = http.post(
            f"{INFERENCE_URL}/internal/embed-by-key-batch",
            data={"keys": json.dumps(batch)},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        all_embs.extend(data["embeddings"])
        elapsed = data["elapsed_ms"]
        print(f"  батч {start // BATCH_SIZE + 1}/{(len(specs)-1)//BATCH_SIZE+1}: "
              f"{len(batch)} шт, {elapsed}ms", flush=True)
    return all_embs


def compute_map(
    query_embs: list[np.ndarray],
    query_ids: list[str],
    gallery_embs: list[np.ndarray],
    gallery_ids: list[str],
    gallery_vehs: list[str],
    top_n: int = 10,
) -> float:
    """mAP@10: mean Average Precision @ top-N."""
    gallery_np = np.array(gallery_embs, dtype=np.float32)  # (G, D)
    aps = []

    for q_emb, q_veh in zip(query_embs, query_ids):
        # Нормализация (эмбеддинги уже L2-norm)
        sim = gallery_np @ q_emb  # (G,)
        top_idx = np.argsort(sim)[::-1][:top_n]

        # Ищем позитивы (тот же vehicle_id)
        n_pos = sum(1 for v in gallery_vehs if v == q_veh)
        if n_pos == 0:
            continue

        ap = 0.0
        n_hit = 0
        for rank, idx in enumerate(top_idx, 1):
            if gallery_vehs[idx] == q_veh:
                n_hit += 1
                ap += n_hit / rank
        ap /= min(n_pos, top_n)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def main():
    print("🔌 Подключение к inference...")
    http = httpx.Client(timeout=600)
    r = http.get(f"{INFERENCE_URL}/health")
    r.raise_for_status()
    health = r.json()
    print(f"   Статус: {health['status']}, модель: {health['model_version']}, "
          f"YOLO: {health['yolo_loaded']}")

    # Загружаем сплиты
    val_q = load_split("val_query")
    val_g = load_split("val_gallery")

    # Исключаем мусорные кадры
    excluded = load_excluded()
    val_q = [r for r in val_q if r["image_id"] not in excluded]
    val_g = [r for r in val_g if r["image_id"] not in excluded]
    print(f"   После фильтрации: {len(val_q)} query, {len(val_g)} gallery")

    if not val_q or not val_g:
        print("❌ Один из сплитов пуст после фильтрации")
        sys.exit(1)

    # Собираем specs для инференса
    q_specs = [
        {"image_id": r["image_id"], "x": int(r["x"]), "y": int(r["y"]),
         "w": int(r["w"]), "h": int(r["h"])}
        for r in val_q
    ]
    g_specs = [
        {"image_id": r["image_id"], "x": int(r["x"]), "y": int(r["y"]),
         "w": int(r["w"]), "h": int(r["h"])}
        for r in val_g
    ]

    # Инференс
    print(f"\n🚀 Инференс {len(q_specs)} query...")
    t0 = time.time()
    q_embs = embed_batch(http, q_specs)
    print(f"   query done in {time.time()-t0:.1f}s")

    print(f"\n🚀 Инференс {len(g_specs)} gallery...")
    t0 = time.time()
    g_embs = embed_batch(http, g_specs)
    print(f"   gallery done in {time.time()-t0:.1f}s")

    # mAP
    q_np = [np.array(e, dtype=np.float32) for e in q_embs]
    g_np = [np.array(e, dtype=np.float32) for e in g_embs]
    q_veh = [r["vehicle_id"] for r in val_q]
    g_veh = [r["vehicle_id"] for r in val_g]

    print("\n📊 Вычисление mAP@10...")
    t0 = time.time()
    map10 = compute_map(q_np, q_veh, g_np, g_veh, g_veh, top_n=10)
    print(f"   вычислено за {time.time()-t0:.1f}s")

    print(f"\n{'='*50}")
    print(f"  🏆 mAP@10 = {map10:.4f}")
    print(f"  Модель: {health['model_version']}")
    print(f"  Query: {len(q_specs)}, Gallery: {len(g_specs)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
