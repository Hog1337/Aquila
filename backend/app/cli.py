"""Пакетный запуск решения на закрытом тесте (ТЗ п.8): организаторы передают каталог
изображений и CSV целиком, решение обрабатывает весь набор и возвращает submission.csv,
embeddings.npy, candidates.csv.

Запуск внутри контейнера backend (том --input смонтирован read-only, --output — на запись):
    python -m app.cli submit --input /input --output /output

По умолчанию ожидает в --input каталог images/ и файлы test_gallery.csv, test_query.csv.
Импортирует галерею в Qdrant, запросы обрабатывает на лету (без сохранения в Qdrant).
"""
import argparse
import asyncio
import csv
import io
import re
from pathlib import Path

from .services.database import db
from .services.storage import storage
from .services.export import build_export_artifacts

REQUIRED_COLUMNS = ("image_id", "x", "y", "w", "h")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


def _parse_csv(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise SystemExit(f"{path}: в CSV нет колонок: {', '.join(missing)}")
    rows = []
    for n, row in enumerate(reader, start=2):
        image_id = (row.get("image_id") or "").strip()
        if not IMAGE_ID_RE.match(image_id):
            raise SystemExit(f"{path}, строка {n}: некорректный image_id")
        try:
            x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            raise SystemExit(f"{path}, строка {n}: x, y, w, h должны быть целыми числами")
        rows.append({
            "image_id": image_id, "x": x, "y": y, "w": w, "h": h,
            "vehicle_id": (row.get("vehicle_id") or "").strip() or None,
            "camera_id": (row.get("camera_id") or "").strip() or None,
        })
    if not rows:
        raise SystemExit(f"{path}: в CSV нет строк с аннотациями")
    return rows


def _find_image(images_dir: Path, image_id: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


async def cmd_submit(args: argparse.Namespace):
    """Полный цикл: импорт галереи + запросов (как веб-флоу) → экспорт 3 файлов.
    Использует тот же механизм, что и веб-интерфейс:
    uploadSingleImage → process_one (асинхронно) → ожидание → build_export_artifacts."""
    await db.connect()
    try:
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = input_dir / "images"
        gallery_csv = input_dir / args.gallery_csv
        query_csv = input_dir / args.query_csv
        for p in (images_dir, gallery_csv, query_csv):
            if not p.exists():
                raise SystemExit(f"Не найдено: {p}")

        threshold = args.threshold
        if threshold is None:
            threshold = (await db.get_current_threshold())["value"]

        gallery_rows = _parse_csv(gallery_csv)
        query_rows = _parse_csv(query_csv)
        all_rows = gallery_rows + query_rows
        all_image_ids = [r["image_id"] for r in all_rows]

        print(f"[submit] загрузка {len(all_rows)} изображений: "
              f"{len(gallery_rows)} gallery + {len(query_rows)} query")

        # 1. Параллельная загрузка всех кадров в S3 + запуск process_one (как веб)
        # Используем воркер-пул (как runImages на фронте) вместо asyncio.gather
        from PIL import Image as PILImage
        from .services.processor import process_one as fire_process_one, flush_qdrant_batch

        _UPLOAD_WORKERS = 8
        uploaded = 0
        upload_failures = []

        async def upload_worker():
            nonlocal uploaded
            while True:
                row = await upload_queue.get()
                if row is None:
                    upload_queue.task_done()
                    return
                try:
                    image_id = row["image_id"]
                    img_path = _find_image(images_dir, image_id)
                    if img_path is None:
                        raise SystemExit(f"Кадр {image_id} не найден в {images_dir}")
                    data = img_path.read_bytes()
                    await asyncio.to_thread(storage.upload_image, f"images/{image_id}.jpg", data)
                    try:
                        w_img, h_img = PILImage.open(io.BytesIO(data)).size
                    except Exception:
                        w_img, h_img = 1920, 1080
                    await db.save_image(image_id, w_img, h_img, status="uploaded")
                    asyncio.create_task(fire_process_one(
                        image_id, row["x"], row["y"], row["w"], row["h"],
                        row.get("vehicle_id"), image_data=data
                    ))
                    uploaded += 1
                    if uploaded % 50 == 0 or uploaded == len(all_rows):
                        print(f"\r[submit] загружено: {uploaded}/{len(all_rows)}", end="", flush=True)
                except Exception as e:
                    upload_failures.append(f"{row['image_id']}: {e}")
                finally:
                    upload_queue.task_done()

        upload_queue = asyncio.Queue()
        for r in all_rows:
            upload_queue.put_nowait(r)

        workers = [asyncio.create_task(upload_worker()) for _ in range(_UPLOAD_WORKERS)]
        await upload_queue.join()
        for _ in workers:
            await upload_queue.put(None)  # сигнал остановки
        await asyncio.gather(*workers)
        print()

        if upload_failures:
            print(f"[submit] Ошибки загрузки ({len(upload_failures)}): {upload_failures[0]}")
            return

        # 2. Ждём завершения векторизации всех изображений (поллинг как на фронте)
        print(f"[submit] ожидание векторизации...")
        for attempt in range(180):
            status = await db.get_processing_status_for_ids(all_image_ids)
            done = status["ready"] + status["failed"]
            tracked = status.get("tracked", status["total"])
            if attempt % 10 == 0 or done >= tracked:
                print(f"\r[submit] векторизация: {done}/{tracked} "
                      f"(ready={status['ready']}, "
                      f"failed={status['failed']}, "
                      f"uploaded={status['uploaded']})", end="", flush=True)
            if tracked > 0 and done >= tracked:
                print()
                break
            await asyncio.sleep(2)
        else:
            print()
            failed_img_ids = []
            for img_id in all_image_ids:
                row = await db.get_image_status(img_id)
                if not row or row.get("processing_status") == "failed":
                    failed_img_ids.append((img_id, (row or {}).get("error", "нет статуса")))
            if failed_img_ids:
                print(f"[submit] ВНИМАНИЕ: {len(failed_img_ids)} изображений не обработаны:", flush=True)
                for fid, ferr in failed_img_ids[:10]:
                    print(f"[submit]   ❌ {fid}: {ferr}", flush=True)
                if len(failed_img_ids) > 10:
                    print(f"[submit]   ... и ещё {len(failed_img_ids) - 10}", flush=True)

        # 3. Сбрасываем буфер Qdrant (чтобы все векторы были доступны для поиска)
        await flush_qdrant_batch()

        # 4. Экспорт (эмбеддинги уже в Qdrant, build_export_artifacts заберёт их оттуда)
        print(f"[submit] экспорт: threshold={threshold:.3f}, rerank={args.rerank}")
        artifacts = await build_export_artifacts(query_rows, threshold, use_rerank=args.rerank, gallery_rows=gallery_rows)
        for name, data in artifacts.items():
            (output_dir / name).write_bytes(data)
            print(f"[submit] {name}: {len(data)} байт -> {output_dir / name}")
    finally:
        await db.close()


async def cmd_evaluate(args: argparse.Namespace):
    """Оценка качества: запускает эталонный расчёт метрик (как evaluate.py организаторов).
    Используется в submit.sh --gt-csv для валидации на известном ground truth."""
    import numpy as np
    import pandas as pd

    gt_path = Path(args.gt)
    submission_path = Path(args.submission)
    if not gt_path.exists():
        raise SystemExit(f"GT CSV не найден: {gt_path}")
    if not submission_path.exists():
        raise SystemExit(f"submission.csv не найден: {submission_path}")

    candidates_path = Path(args.candidates) if args.candidates else None
    embeddings_path = Path(args.embeddings) if args.embeddings else None
    query_csv_path = Path(args.query) if args.query else None
    gallery_csv_path = Path(args.gallery) if args.gallery else None

    # Загружаем evaluate.py из директории скриптов (или используем встроенную логику)
    # Ищем evaluate.py рядом с проектом или в известных locations
    eval_script = None
    for candidate in [
        Path("/lct/reid/evaluate.py"),
        Path(__file__).parent.parent.parent.parent / "reid" / "evaluate.py",
        Path("/input/../evaluate.py"),
    ]:
        if candidate.exists():
            eval_script = candidate
            break

    if eval_script:
        # Запускаем evaluate.py как подпроцесс
        import subprocess
        cmd = ["python", str(eval_script), "--gt", str(gt_path), "--submission", str(submission_path)]
        if candidates_path:
            cmd += ["--candidates", str(candidates_path)]
        if embeddings_path and query_csv_path and gallery_csv_path:
            cmd += ["--embeddings", str(embeddings_path), "--query", str(query_csv_path), "--gallery", str(gallery_csv_path)]
        if args.json:
            cmd += ["--json", args.json]
        print(f"[evaluate] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"evaluate.py завершился с ошибкой (код {result.returncode})")
    else:
        # Встроенная мини-оценка, если evaluate.py не найден
        print("[evaluate] evaluate.py не найден, использую встроенную логику")
        await _run_inline_evaluation(gt_path, submission_path, candidates_path, embeddings_path, query_csv_path, gallery_csv_path, args.json)


async def _run_inline_evaluation(gt_path, submission_path, candidates_path, embeddings_path, query_csv_path, gallery_csv_path, json_path):
    """Встроенная мини-оценка (упрощённый аналог evaluate.py)."""
    import numpy as np
    import pandas as pd

    TOP_K = 10

    # Загрузка GT
    gt = pd.read_csv(gt_path, dtype={"image_id": str})
    for col in ("image_id", "vehicle_id", "camera_id", "split"):
        if col not in gt.columns:
            raise SystemExit(f"В GT CSV нет колонки '{col}'")
    query_gt = gt[gt.split == "query"].set_index("image_id")
    gallery_gt = gt[gt.split == "gallery"].set_index("image_id")
    if query_gt.empty or gallery_gt.empty:
        raise SystemExit("В GT пустой query или gallery")

    gal_vid = gallery_gt.vehicle_id.to_dict()
    gal_cam = gallery_gt.camera_id.to_dict()

    # Загрузка submission
    ranked = {}
    with open(submission_path, "r", encoding="utf-8-sig", newline="") as f:
        import csv
        reader = csv.reader(f)
        for vals in reader:
            vals = [v.strip() for v in vals if v and v.strip()]
            if not vals:
                continue
            qid, preds = vals[0], vals[1:]
            ranked[qid] = [p for p in preds if p in gal_vid][:TOP_K]

    n_scored = 0
    aps, r1s, r5s = [], [], []
    missing_q = 0

    for qid, row in query_gt.iterrows():
        same_vid = (gallery_gt.vehicle_id == row.vehicle_id)
        same_cam = (gallery_gt.camera_id == row.camera_id)
        n_pos = int((same_vid & ~same_cam).sum())
        if n_pos == 0:
            continue
        n_scored += 1

        if qid not in ranked:
            aps.append(0.0); r1s.append(False); r5s.append(False); continue

        clean = []
        for gid in ranked[qid]:
            if gid in gal_vid and not (gal_vid[gid] == row.vehicle_id and gal_cam[gid] == row.camera_id):
                clean.append(gid)
        clean = clean[:TOP_K]

        rel = np.array([gal_vid.get(g) == row.vehicle_id for g in clean], dtype=bool)
        if rel.any():
            cum = np.cumsum(rel)
            prec = cum / (np.arange(len(rel)) + 1)
            aps.append(float((prec * rel).sum() / min(n_pos, TOP_K)))
        else:
            aps.append(0.0)
        r1s.append(bool(rel[:1].any()))
        r5s.append(bool(rel[:5].any()))

    mAP = float(np.mean(aps)) if aps else 0.0
    rank1 = float(np.mean(r1s)) if r1s else 0.0
    rank5 = float(np.mean(r5s)) if r5s else 0.0

    print(f"\n=== Результаты оценки ===")
    print(f"Запросов в зачёте: {n_scored}")
    print(f"mAP@{TOP_K}: {mAP:.4f}")
    print(f"Rank-1: {rank1:.4f}")
    print(f"Rank-5: {rank5:.4f}")

    if json_path:
        import json
        report = {"ranking": {"n_scored": n_scored, f"mAP@{TOP_K}": mAP, "Rank-1": rank1, "Rank-5": rank5}}
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Отчёт -> {json_path}")


async def cmd_clear(args: argparse.Namespace):
    """Очищает галерею: удаляет все объекты из PG и Qdrant."""
    await db.connect()
    try:
        # Qdrant: scroll all points and delete
        from .services.searcher import searcher
        all_points = await asyncio.to_thread(searcher.scroll_all)
        point_ids = [p["id"] for p in all_points]
        if point_ids:
            for start in range(0, len(point_ids), 500):
                chunk = point_ids[start:start + 500]
                for pid in chunk:
                    await asyncio.to_thread(searcher.delete_point, pid)
            print(f"[clear] Qdrant: удалено {len(point_ids)} точек")
        else:
            print("[clear] Qdrant: пусто")

        # PG: удаляем все данные
        async with db.pool.acquire() as conn:
            obj = await conn.fetchval("SELECT COUNT(*) FROM reid.gallery_objects")
            await conn.execute("DELETE FROM reid.gallery_objects")
            await conn.execute("DELETE FROM reid.images")
            print(f"[clear] PostgreSQL: удалено {obj} объектов, images очищены")

        print("[clear] Галерея очищена")
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Импортировать папку с images/+CSV и выполнить экспорт (ТЗ п.8)")
    p_submit.add_argument("--input", required=True, help="Каталог с images/ и CSV галереи/запросов")
    p_submit.add_argument("--output", required=True, help="Каталог для submission.csv/embeddings.npy/candidates.csv")
    p_submit.add_argument("--gallery-csv", default="test_gallery.csv", help="Имя CSV галереи внутри --input")
    p_submit.add_argument("--query-csv", default="test_query.csv", help="Имя CSV запросов внутри --input")
    p_submit.add_argument("--threshold", type=float, default=None, help="Порог отказа (по умолчанию — текущий в системе)")
    p_submit.add_argument("--rerank", action="store_true", help="Использовать Query Expansion")
    p_submit.set_defaults(func=cmd_submit)

    p_clear = sub.add_parser("clear", help="Очистить галерею: удалить все объекты из PG и Qdrant")
    p_clear.set_defaults(func=cmd_clear)

    p_eval = sub.add_parser("evaluate", help="Оценить качество: mAP/Rank по GT и submission.csv")
    p_eval.add_argument("--gt", required=True, help="ground truth CSV (image_id,vehicle_id,camera_id,split)")
    p_eval.add_argument("--submission", required=True, help="submission.csv")
    p_eval.add_argument("--candidates", default=None, help="candidates.csv (опционально)")
    p_eval.add_argument("--embeddings", default=None, help="embeddings.npy (опционально)")
    p_eval.add_argument("--query", default=None, help="test_query.csv (для --embeddings)")
    p_eval.add_argument("--gallery", default=None, help="test_gallery.csv (для --embeddings)")
    p_eval.add_argument("--json", default=None, help="сохранить JSON-отчёт")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
