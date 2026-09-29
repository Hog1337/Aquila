import csv
import io
import uuid
import asyncio
from collections import Counter

from fastapi import APIRouter, HTTPException, File, Form, UploadFile

import numpy as np

from .. import config
from ..schemas.models import MetricJob, MetricRun
from ..services.database import db
from ..services.storage import storage
from ..services.searcher import searcher
from ..services.inference_client import inference

def _dedup_by_image_id(hits: list[dict]) -> list[dict]:
    """Оставляет один результат (с максимальным score) на image_id."""
    best = {}
    for h in hits:
        iid = h["image_id"]
        if iid not in best or h["score"] > best[iid]["score"]:
            best[iid] = h
    return list(best.values())


    print(f"[gallery] Создано {created}/{total} gallery-объектов", flush=True)
    


router = APIRouter()

_active_metrics: dict[str, asyncio.Task] = {}
TOP_K = 10
REQUIRED_COLUMNS = ("image_id", "x", "y", "w", "h")


def _parse_csv(data: bytes) -> list[dict]:
    """Парсит CSV с bbox. Возвращает список {image_id, x, y, w, h, vehicle_id?, camera_id?}."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "CSV должен быть в кодировке UTF-8")
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise HTTPException(422, f"В CSV нет колонок: {', '.join(missing)}")
    rows = []
    for n, row in enumerate(reader, start=2):
        image_id = (row.get("image_id") or "").strip()
        try:
            x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            raise HTTPException(422, f"CSV, строка {n}: x, y, w, h должны быть целыми числами")
        rows.append({
            "image_id": image_id, "x": x, "y": y, "w": w, "h": h,
        })
    if not rows:
        raise HTTPException(422, "В CSV нет строк с аннотациями")
    return rows


def _parse_gt_csv(data: bytes) -> tuple[dict, dict]:
    """Парсит GT CSV в формате evaluate.py: image_id,vehicle_id,camera_id,split.
    Возвращает (query_gt, gallery_gt) — словари {image_id: {vehicle_id, camera_id}}."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "GT CSV должен быть в кодировке UTF-8")
    reader = csv.DictReader(io.StringIO(text))
    for col in ("image_id", "vehicle_id", "camera_id", "split"):
        if col not in (reader.fieldnames or []):
            raise HTTPException(422, f"В GT CSV нет колонки '{col}'")
    query_gt, gallery_gt = {}, {}
    for n, row in enumerate(reader, start=2):
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            continue
        split = (row.get("split") or "").strip().lower()
        if split not in ("query", "gallery"):
            raise HTTPException(422, f"GT CSV, строка {n}: split должен быть 'query' или 'gallery', получено '{split}'")
        record = {
            "vehicle_id": (row.get("vehicle_id") or "").strip(),
            "camera_id": (row.get("camera_id") or "").strip(),
        }
        if split == "query":
            query_gt[image_id] = record
        else:
            gallery_gt[image_id] = record
    if not query_gt or not gallery_gt:
        raise HTTPException(422, "GT CSV: нужны и query, и gallery (проверьте колонку split)")
    return query_gt, gallery_gt


def _check_images_sync(storage, image_ids: list[str]) -> list[str]:
    """Синхронная проверка: какие image_id отсутствуют в хранилище."""
    missing = []
    for image_id in image_ids:
        key = f"images/{image_id}.jpg"
        data = storage.get_image(key)
        if data is None:
            data = storage.get_image(f"images/{image_id}.png")
        if data is None:
            missing.append(image_id)
    return missing


@router.post("/metrics/runs", status_code=202, response_model=MetricJob)
async def start_metrics_run(
    query_csv: UploadFile = File(...),
    gallery_csv: UploadFile = File(...),
    gt_csv: UploadFile = File(...),
    use_rerank: bool = Form(False),
):
    """Запускает оценку качества.

    Принимает:
      query_csv  — CSV с запросами (image_id, x, y, w, h) — для инференса
      gallery_csv — CSV с галереей (image_id, x, y, w, h) — для инференса
      gt_csv     — CSV в формате evaluate.py (image_id, vehicle_id, camera_id, split)
      use_rerank — Query Expansion

    Если каких-то image_id нет в хранилище → 409 с их списком.
    """
    q_data = await query_csv.read()
    g_data = await gallery_csv.read()
    gt_data = await gt_csv.read()

    query_rows = _parse_csv(q_data)
    gallery_rows = _parse_csv(g_data)
    query_gt, gallery_gt = _parse_gt_csv(gt_data)

    # Проверяем, какие image_id нужны для инференса
    all_image_ids = list(set(
        [r["image_id"] for r in query_rows] +
        [r["image_id"] for r in gallery_rows]
    ))
    missing = await asyncio.to_thread(_check_images_sync, storage, all_image_ids)
    if missing:
        raise HTTPException(409, detail={
            "message": f"Не хватает {len(missing)} изображений",
            "missing": missing,
            "total": len(all_image_ids),
        })

    job_id = str(uuid.uuid4())
    await db.create_metric_job(job_id)
    task = asyncio.create_task(_run_metrics(job_id, query_rows, gallery_rows, query_gt, gallery_gt, use_rerank))
    _active_metrics[job_id] = task
    return MetricJob(id=job_id, status="queued")





async def _run_metrics(
    job_id: str,
    query_rows: list[dict],
    gallery_rows: list[dict],
    query_gt: dict,
    gallery_gt: dict,
    use_rerank: bool,
):
    """Считает mAP@10, Rank-1/5, PR-кривую, F1, TNR, PR-AUC.
    Протокол: как в evaluate.py организаторов.
    """
    try:
        await db.update_metric_job(job_id, status="running")

        n_queries = len(query_rows)
        gallery_image_ids = [r["image_id"] for r in gallery_rows]

        # Запускаем обработку незавершённых изображений
        from ..services.processor import process_all_pending
        await process_all_pending()

        # Ждём завершения обработки всех необходимых изображений
        all_needed = set(r["image_id"] for r in query_rows) | set(r["image_id"] for r in gallery_rows)
        for attempt in range(120):
            ready = set()
            for img_id in all_needed:
                row = await db.get_image_status(img_id)
                if row and row.get("processing_status") == "ready":
                    ready.add(img_id)
            if ready == all_needed:
                break
            await asyncio.sleep(2)
        else:
            # Таймаут — проверяем, какие не готовы
            failed = []
            for img_id in all_needed:
                row = await db.get_image_status(img_id)
                if not row or row.get("processing_status") == "failed":
                    failed.append(img_id)
            if failed:
                raise RuntimeError(f"Не удалось обработать изображения: {failed[:10]}...")
            # Остальные всё ещё в очереди — продолжаем (надеемся на лучшее)
        # Обновляем прогресс после ожидания
        if job_id:
            await db.update_metric_job(job_id, progress=5)

        # Ground truth для галереи
        gal_vid = {}
        gal_cam = {}
        for gid, info in gallery_gt.items():
            gal_vid[gid] = info["vehicle_id"]
            gal_cam[gid] = info["camera_id"]

        vehicle_counts = Counter(gal_vid.values())

        aps, r1s, r5s = [], [], []
        best_scores, has_matches, top_corrects = [], [], []

        # Пробуем забрать эмбеддинги запросов из Qdrant (они уже посчитаны при загрузке)
        # Если вектора нет — сделаем инференс как fallback
        from ..services.processor import gallery_object_id as make_obj_id
        query_obj_ids = [make_obj_id(q["image_id"], q["x"], q["y"], q["w"], q["h"]) for q in query_rows]
        cached_embs: dict[str, list[float]] = {}
        try:
            cached_embs = await asyncio.to_thread(searcher.retrieve_vectors, query_obj_ids)
        except Exception as e:
            print(f"[metrics] не удалось прочитать эмбеддинги из Qdrant: {e}", flush=True)

        for qi, q in enumerate(query_rows):
            image_id = q["image_id"]
            x, y, w, h = q["x"], q["y"], q["w"], q["h"]

            # GT для запроса
            q_info = query_gt.get(image_id, {})
            q_vid = q_info.get("vehicle_id", "")
            q_cam = q_info.get("camera_id", "")

            # Количество позитивов ПОСЛЕ junk-фильтра
            n_valid_pos = 0
            for gid, gvid in gal_vid.items():
                if gvid == q_vid and gal_cam.get(gid) != q_cam:
                    n_valid_pos += 1
            has_match = n_valid_pos > 0
            has_matches.append(has_match)

            # Берём эмбеддинг: из Qdrant (если есть) или через инференс
            obj_id = query_obj_ids[qi]
            emb = cached_embs.get(obj_id)
            if emb is None:
                # Fallback: кадр из S3 + инференс
                image_data = await asyncio.to_thread(storage.get_image, f"images/{image_id}.jpg")
                if image_data is None:
                    image_data = await asyncio.to_thread(storage.get_image, f"images/{image_id}.png")
                if image_data is None:
                    raise ValueError(f"Кадр {image_id} не найден в хранилище")
                emb, _ = await asyncio.to_thread(inference.embed, image_data, x, y, w, h)

            # Поиск по галерее
            if use_rerank:
                hits = await asyncio.to_thread(searcher.search_by_image_ids, emb, gallery_image_ids, TOP_K * 3)
            else:
                hits = await asyncio.to_thread(searcher.search_by_image_ids, emb, gallery_image_ids, TOP_K * 3)

            # Дедуплицируем по image_id (full-frame + crop → один с макс. score)
            hits = _dedup_by_image_id(hits)

            # Junk-фильтр
            clean = []
            for h in hits:
                gid = h["image_id"]
                if gal_vid.get(gid) == q_vid and gal_cam.get(gid) == q_cam:
                    continue
                clean.append(h)
            clean = clean[:TOP_K]

            best_scores.append(clean[0]["score"] if clean else float("-inf"))
            top_corrects.append(bool(clean) and gal_vid.get(clean[0]["image_id"]) == q_vid)

            if has_match:
                rel = np.array([1 if gal_vid.get(h["image_id"]) == q_vid else 0 for h in clean], dtype=bool)
                if rel.any():
                    cum = np.cumsum(rel)
                    prec = cum / (np.arange(len(rel)) + 1)
                    aps.append(float((prec * rel).sum() / min(n_valid_pos, TOP_K)))
                else:
                    aps.append(0.0)
                r1s.append(bool(rel[:1].any()))
                r5s.append(bool(rel[:5].any()))

            if qi % 25 == 0 or qi == n_queries - 1:
                await db.update_metric_job(job_id, progress=int((qi + 1) / n_queries * 80))

        mAP = float(np.mean(aps)) if aps else 0.0
        rank1 = float(np.mean(r1s)) if r1s else 0.0
        rank5 = float(np.mean(r5s)) if r5s else 0.0
        pr_auc = _pr_auc(np.array(best_scores), np.array(has_matches, dtype=int))

        # PR-кривая по порогам
        thresholds = np.arange(0.30, 0.96, 0.01)
        curve = []
        best_f1 = best_prec = best_rec = best_tnr = 0.0
        best_th = 0.7

        for th in thresholds:
            tp = fp = fn = tn = fp_openset = 0
            for score, has_match, correct in zip(best_scores, has_matches, top_corrects):
                accepted = score >= th
                if accepted and has_match and correct:
                    tp += 1
                elif accepted:
                    fp += 1
                    if not has_match:
                        fp_openset += 1
                elif has_match:
                    fn += 1
                else:
                    tn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            tnr = tn / (tn + fp_openset) if (tn + fp_openset) > 0 else 1.0

            curve.append({
                "threshold": round(float(th), 2),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "tnr": round(float(tnr), 4),
            })

            if f1 > best_f1 and tnr >= 0.9:
                best_f1 = f1
                best_th = float(th)
                best_prec = float(prec)
                best_rec = float(rec)
                best_tnr = float(tnr)

        run_id = await db.save_metrics_run({
            "model_version": config.MODEL_VERSION,
            "threshold": best_th,
            "map": round(float(mAP), 4),
            "rank1": round(float(rank1), 4),
            "rank5": round(float(rank5), 4),
            "f1": round(float(best_f1), 4),
            "tnr": round(float(best_tnr), 4),
            "pr_auc": round(float(pr_auc), 4),
            "n_queries": n_queries,
            "details": {
                "curve": curve,
                "n_scored": len(aps),
                "n_openset_excluded": n_queries - len(aps),
                "query_image_ids": [r["image_id"] for r in query_rows],
                "gallery_image_ids": [r["image_id"] for r in gallery_rows],
            },
        })

        await db.update_metric_job(job_id, status="done", progress=100, run_id=run_id)
    except Exception as e:
        await db.update_metric_job(job_id, status="failed", error=str(e))
    finally:
        _active_metrics.pop(job_id, None)


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision по кривой Precision-Recall."""
    finite = np.isfinite(scores)
    if labels.sum() == 0 or not finite.any():
        return 0.0
    s = np.where(finite, scores, np.min(scores[finite]) - 1.0)
    order = np.argsort(-s, kind="stable")
    y = labels[order]
    cum_tp = np.cumsum(y)
    prec = cum_tp / (np.arange(len(y)) + 1)
    rec = cum_tp / labels.sum()
    ap, prev_rec = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_rec)
        prev_rec = r
    return float(ap)


@router.get("/metrics/runs/latest", response_model=MetricRun)
async def get_latest_metrics():
    run = await db.get_latest_metrics_run()
    if not run:
        raise HTTPException(404, "Оценка ещё не запускалась")
    return MetricRun(**run)


@router.get("/metrics/runs/{job_id}", response_model=MetricJob)
async def get_metrics_job(job_id: str):
    job = await db.get_metric_job(job_id)
    if not job:
        raise HTTPException(404, "Задание не найдено")
    run = None
    if job.get("run_id"):
        run_data = await db.get_latest_metrics_run()
        if run_data:
            run = MetricRun(**run_data)
    return MetricJob(id=job["id"], status=job["status"], progress=job.get("progress", 0),
                     error=job.get("error"), run=run)


@router.post("/metrics/runs/{job_id}/cleanup", status_code=200)
async def cleanup_metrics_run(job_id: str, scope: str = "all"):
    """Удаляет из галереи объекты, загруженные для этого прогона метрик.
    scope: 'query' — только query, 'gallery' — только gallery, 'all' — и query, и gallery.
    Передаётся как query-параметр. Чистит PG, Qdrant и S3."""
    from ..services.searcher import searcher
    from ..services.storage import storage
    import boto3
    from botocore.config import Config

    job = await db.get_metric_job(job_id)
    if not job or not job.get("run_id"):
        raise HTTPException(404, "Прогон не найден")

    # Получаем данные конкретного прогона (не latest!)
    run_data = await db.get_metrics_run_by_id(job["run_id"])
    if not run_data:
        raise HTTPException(404, "Данные прогона не найдены")

    details = run_data.get("details", {}) or {}
    query_ids = details.get("query_image_ids", [])
    gallery_ids = details.get("gallery_image_ids", [])

    to_delete = list(set(
        (query_ids if scope in ("query", "all") else []) +
        (gallery_ids if scope in ("gallery", "all") else [])
    ))
    if not to_delete:
        return {"status": "ok", "deleted_pg": 0, "deleted_qdrant": 0, "deleted_s3": 0, "scope": scope}

    # Удаляем из PG (gallery_objects + images)
    deleted_pg = await db.delete_objects_by_image_ids(to_delete)
    deleted_images = await db.delete_images_by_ids(to_delete)

    # Удаляем из Qdrant одной командой (filter-based)
    deleted_qdrant = await asyncio.to_thread(searcher.delete_by_image_ids, to_delete)

    # Удаляем из S3 (batch delete, до 1000 объектов за раз)
    deleted_s3 = 0
    try:
        s3_client = boto3.client("s3",
            endpoint_url=config.S3_ENDPOINT,
            aws_access_key_id=config.S3_ACCESS_KEY,
            aws_secret_access_key=config.S3_SECRET_KEY,
            config=Config(signature_version="s3v4"), region_name="us-east-1")
        batch = []
        for image_id in to_delete:
            batch.append({"Key": f"images/{image_id}.jpg"})
            if len(batch) == 1000:
                resp = await asyncio.to_thread(s3_client.delete_objects,
                    Bucket="gallery", Delete={"Objects": batch, "Quiet": True})
                deleted_s3 += len(batch) - len(resp.get("Errors", []))
                batch = []
        if batch:
            resp = await asyncio.to_thread(s3_client.delete_objects,
                Bucket="gallery", Delete={"Objects": batch, "Quiet": True})
            deleted_s3 += len(batch) - len(resp.get("Errors", []))
    except Exception as e:
        print(f"[cleanup] S3 batch delete error: {e}", flush=True)

    return {"status": "ok", "deleted_pg": deleted_pg, "deleted_qdrant": deleted_qdrant, "deleted_s3": deleted_s3, "deleted_images": deleted_images, "scope": scope}


@router.delete("/metrics/runs", status_code=200)
async def clear_metrics():
    """Удаляет всю историю прогонов метрик (metric_jobs + metric_runs)."""
    return await db.clear_metrics()
