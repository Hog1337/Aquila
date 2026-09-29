#!/usr/bin/env python3
"""Тестовый скрипт: импорт val_gallery + val_query через API, запуск метрик."""
import sys
import time
import json
from pathlib import Path

import httpx

API = "http://localhost:8000/api/v1"

REID_SPLITS = Path("/home/limon/data/university/lct/reid/splits")
IMAGES_DIR = Path("/home/limon/data/university/lct/data/reid/images")

client = httpx.Client(timeout=120)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def create_import_session(csv_path: Path, role: str) -> dict:
    """Шаг 1: загружаем CSV, получаем сессию."""
    log(f"Создаю сессию импорта: {csv_path.name} role={role}")
    resp = client.post(
        f"{API}/gallery/import-sessions",
        files={"annotations": (csv_path.name, csv_path.read_bytes(), "text/csv")},
        data={"role": role},
    )
    if resp.status_code != 201:
        raise RuntimeError(f"create_import_session: {resp.status_code} {resp.text}")
    session = resp.json()
    log(f"  session_id={session['id']}, total={session['total']}, images={len(session['image_ids'])}")
    return session


def upload_images(session_id: str, image_ids: list[str]):
    """Шаг 2: загружаем кадры по одному. Пропускаем отсутствующие."""
    missing_local = 0
    uploaded = 0
    for i, img_id in enumerate(image_ids):
        img_path = IMAGES_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            if missing_local == 0:
                log(f"  WARNING: {img_id}.jpg не найден в {IMAGES_DIR}")
            missing_local += 1
            continue
        data = img_path.read_bytes()
        resp = client.post(
            f"{API}/gallery/import-sessions/{session_id}/images",
            data={"image_id": img_id},
            files={"image": (f"{img_id}.jpg", data, "image/jpeg")},
        )
        if resp.status_code != 204:
            raise RuntimeError(f"upload {img_id}: {resp.status_code} {resp.text}")
        uploaded += 1
        if (i + 1) % 200 == 0 or i == len(image_ids) - 1:
            log(f"  images uploaded: {i + 1}/{len(image_ids)} (ok={uploaded}, missing={missing_local})")
    log(f"  Всего загружено: {uploaded}, пропущено (нет локально): {missing_local}")
    return uploaded, missing_local


def start_import_job(session_id: str) -> dict:
    """Шаг 3: запускаем обработку (инференс + запись в Qdrant+PG)."""
    log("  Запускаю обработку...")
    resp = client.post(f"{API}/gallery/import-sessions/{session_id}/start")
    if resp.status_code != 202:
        raise RuntimeError(f"start_import: {resp.status_code} {resp.text}")
    job = resp.json()
    log(f"  job_id={job['id']}, status={job['status']}")
    return job


def poll_import_job(job_id: str, timeout_s: int = 1800) -> dict:
    """Ждём завершения задания импорта."""
    log(f"  Ожидание завершения импорта (timeout={timeout_s}s)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"{API}/gallery/imports/{job_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"get_import_job: {resp.status_code} {resp.text}")
        job = resp.json()
        if job["status"] in ("done", "failed", "cancelled"):
            log(f"  Статус: {job['status']}, progress={job.get('progress')}%, error={job.get('error')}")
            return job
        if job.get("progress") and job["progress"] >= 0:
            log(f"  Прогресс: {job['progress']}% ({job.get('processed')}/{job.get('total')})")
        time.sleep(5)
    raise TimeoutError(f"Импорт {job_id} не завершился за {timeout_s}с")


def run_batch_import(csv_name: str, role: str) -> tuple[str, int]:
    """Полный цикл импорта одного CSV."""
    csv_path = REID_SPLITS / csv_name
    session = create_import_session(csv_path, role)
    total_uploaded, missing = upload_images(session["id"], session["image_ids"])
    job = start_import_job(session["id"])
    result = poll_import_job(job["id"])
    if result["status"] != "done":
        raise RuntimeError(f"Импорт {csv_name} завершился со статусом {result['status']}: {result.get('error')}")
    return result["id"], total_uploaded


def run_metrics() -> dict:
    """Запускаем оценку метрик и ждём результат."""
    log("=" * 60)
    log("Запускаю оценку метрик...")
    resp = client.post(f"{API}/metrics/runs")
    if resp.status_code != 202:
        raise RuntimeError(f"start_metrics: {resp.status_code} {resp.text}")
    job = resp.json()
    log(f"  metric_job_id={job['id']}")

    # Ждём завершения
    deadline = time.time() + 3600
    while time.time() < deadline:
        resp = client.get(f"{API}/metrics/runs/{job['id']}")
        if resp.status_code != 200:
            raise RuntimeError(f"get_metric_job: {resp.status_code} {resp.text}")
        data = resp.json()
        status = data["status"]
        if status == "done":
            run_data = data.get("run")
            log(f"  Метрики готовы!")
            if run_data:
                log(f"  mAP@10:   {run_data['map']:.4f}")
                log(f"  Rank-1:   {run_data['rank1']:.4f}")
                log(f"  Rank-5:   {run_data['rank5']:.4f}")
                log(f"  F1:       {run_data['f1']:.4f}")
                log(f"  TNR:      {run_data['tnr']:.4f}")
                log(f"  PR-AUC:   {run_data['pr_auc']:.4f}")
                log(f"  Порог τ: {run_data['threshold']:.2f}")
                log(f"  Запросов: {run_data['n_queries']}")
            return run_data or {}
        elif status == "failed":
            raise RuntimeError(f"Метрики упали: {data.get('error')}")
        progress = data.get("progress", 0)
        log(f"  Прогресс метрик: {progress}%")
        time.sleep(5)

    raise TimeoutError("Метрики не завершились за 3600с")


def verify_status():
    """Проверяем статус системы."""
    log("=" * 60)
    log("Проверка статуса...")
    resp = client.get(f"{API}/status")
    s = resp.json()
    log(f"  model: {s['model_version']}")
    log(f"  embedding_dim: {s['embedding_dim']}")
    log(f"  gallery: {s['gallery']}")
    return s


def main():
    t_start = time.time()

    # Проверка статуса перед началом
    status_before = verify_status()
    log(f"  gallery до загрузки: {status_before['gallery']}")

    # 1. Импортируем val_gallery
    log("=" * 60)
    log("ШАГ 1: Импорт val_gallery (role=val_gallery)")
    batch_id_gallery, uploaded_gallery = run_batch_import("val_gallery.csv", "val_gallery")

    # 2. Импортируем val_query
    log("=" * 60)
    log("ШАГ 2: Импорт val_query (role=val_query)")
    batch_id_query, uploaded_query = run_batch_import("val_query.csv", "val_query")

    # 3. Проверка статуса после загрузки
    log("=" * 60)
    status_after = verify_status()
    log(f"  gallery после загрузки: {status_after['gallery']}")

    # 4. Запуск метрик
    log("=" * 60)
    metrics = run_metrics()

    elapsed = time.time() - t_start
    log("=" * 60)
    log(f"ИТОГО: {elapsed:.0f}с ({elapsed/60:.1f} мин)")
    log(f"Загружено gallery: {uploaded_gallery} images / {status_after['gallery']['objects']} объектов")
    log(f"Загружено query:   {uploaded_query} images")

    if metrics:
        print()
        print("=" * 60)
        print("РЕЗУЛЬТАТЫ МЕТРИК НА VALIDATION:")
        print("=" * 60)
        print(f"  mAP@10:      {metrics['map']:.4f}")
        print(f"  Rank-1:      {metrics['rank1']:.4f}")
        print(f"  Rank-5:      {metrics['rank5']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        print(f"  TNR:         {metrics['tnr']:.4f}")
        print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
        print(f"  Порог τ:     {metrics['threshold']:.2f}")
        print(f"  Запросов:    {metrics['n_queries']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
