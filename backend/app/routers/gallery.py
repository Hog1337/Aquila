import asyncio
import csv
import hashlib
import io
import uuid

from fastapi import APIRouter, HTTPException, Query, File, Form, UploadFile, Body
from PIL import Image, UnidentifiedImageError

from ..schemas.models import GalleryListResponse, GalleryObject
from ..services.database import db
from ..services.storage import storage
from ..services.searcher import searcher
from ..services.inference_client import inference

router = APIRouter()


@router.delete("/gallery/objects/{object_id}", status_code=200)
async def delete_gallery_object(object_id: str):
    """Удаляет один объект из галереи: из Qdrant, PostgreSQL и, если кадр больше не используется, из S3."""
    from ..services.searcher import searcher
    from ..services.storage import storage

    # Удаляем из Qdrant (если есть)
    try:
        searcher.delete_point(object_id)
    except Exception:
        pass

    # Удаляем из PG
    try:
        result = await db.delete_gallery_object(object_id)
    except Exception as e:
        raise HTTPException(400, f"Некорректный ID: {e}")

    if not result["image_id"]:
        raise HTTPException(404, f"Объект {object_id} не найден")

    # Удаляем кадр из S3 если он больше не используется
    if result["deleted_image"]:
        try:
            storage.delete_image(f"images/{result['image_id']}.jpg")
        except Exception:
            pass

    return {"status": "ok", "object_id": object_id, "image_id": result["image_id"], "deleted_image": result["deleted_image"]}


@router.delete("/gallery", status_code=200)
async def clear_gallery():
    """Очищает всю галерею: удаляет объекты из Qdrant, PostgreSQL и SeaweedFS."""
    from ..services.searcher import searcher
    from ..services.storage import storage
    import boto3
    from botocore.config import Config
    from .. import config

    # Сначала читаем image_id из БД (до очистки PG)
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("SELECT image_id FROM reid.images")
        image_ids = [r["image_id"] for r in rows]

    # Удаляем из Qdrant одной командой
    if image_ids:
        await asyncio.to_thread(searcher.delete_by_image_ids, image_ids)
    else:
        # Если images пуста — удаляем всё через scroll_all
        all_points = await asyncio.to_thread(searcher.scroll_all)
        for p in all_points:
            await asyncio.to_thread(searcher.delete_point, p["id"])

    # Удаляем из PG
    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM reid.gallery_objects")
        await conn.execute("DELETE FROM reid.images")

    # Удаляем из S3 (batch delete)
    deleted_s3 = 0
    if image_ids:
        try:
            import boto3
            from botocore.config import Config
            s3_client = boto3.client("s3",
                endpoint_url=config.S3_ENDPOINT,
                aws_access_key_id=config.S3_ACCESS_KEY,
                aws_secret_access_key=config.S3_SECRET_KEY,
                config=Config(signature_version="s3v4"), region_name="us-east-1")
            batch = []
            for image_id in image_ids:
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
            print(f"[clear] S3 batch delete error: {e}", flush=True)

    # Если image_ids были получены из БД — первый batch-delete уже удалил всё.
    # Если БД была пуста, но в S3 что-то есть — чистим по листингу.
    if not image_ids:
        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=config.S3_ENDPOINT,
                aws_access_key_id=config.S3_ACCESS_KEY,
                aws_secret_access_key=config.S3_SECRET_KEY,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
            continuation = None
            while True:
                kwargs = {"Bucket": "gallery", "Prefix": "images/"}
                if continuation:
                    kwargs["ContinuationToken"] = continuation
                resp = await asyncio.to_thread(s3_client.list_objects_v2, **kwargs)
                to_delete = [{"Key": obj["Key"]} for obj in resp.get("Contents", [])]
                if to_delete:
                    await asyncio.to_thread(s3_client.delete_objects,
                        Bucket="gallery",
                        Delete={"Objects": to_delete},
                    )
                    deleted_s3 += len(to_delete)
                if not resp.get("IsTruncated"):
                    break
                continuation = resp.get("NextContinuationToken")
        except Exception as e:
            print(f"[clear_gallery] S3 listing cleanup error: {e}", flush=True)

    return {"status": "ok", "deleted_s3": deleted_s3}


@router.get("/gallery/objects", response_model=GalleryListResponse)


@router.get("/gallery/objects", response_model=GalleryListResponse)
async def list_gallery_objects(
    vehicle_id: str | None = Query(None),
    limit: int = Query(24),
    offset: int = Query(0),
):
    """Возвращает страницу объектов галереи с фильтром по vehicle_id."""
    items, total = await db.get_gallery_objects(vehicle_id=vehicle_id, limit=limit, offset=offset)
    return GalleryListResponse(total=total, items=[GalleryObject(**i) for i in items])


@router.post("/gallery/objects", status_code=201)
async def add_gallery_object(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    vehicle_id: str | None = Form(None),
):
    """Добавляет один объект в галерею: сохраняет кадр, считает эмбеддинг и индексирует."""
    contents = await image.read()
    try:
        width, height = Image.open(io.BytesIO(contents)).size
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "Файл не является изображением")
    image_id = hashlib.sha256(contents).hexdigest()[:16]
    obj_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{image_id}:{x}:{y}:{w}:{h}"))

    emb, _ = inference.embed(contents, x, y, w, h)

    storage.upload_image(f"images/{image_id}.jpg", contents)

    await db.save_image(image_id, width, height)
    await db.save_gallery_object(obj_id, image_id, {"x": x, "y": y, "w": w, "h": h}, vehicle_id, None)

    searcher.upsert(obj_id, emb, {
        "image_id": image_id,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "vehicle_id": vehicle_id,
        "batch_id": None,
    })

    return {"id": obj_id, "image_id": image_id, "vehicle_id": vehicle_id, "bbox": {"x": x, "y": y, "w": w, "h": h}}


@router.post("/gallery/upload-image", status_code=200)
async def upload_single_image(
    image_id: str = Form(...),
    file: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    w: int = Form(0),
    h: int = Form(0),
    vehicle_id: str | None = Form(None),
):
    """Загружает одно изображение в хранилище и запускает фоновую обработку
    (crop → YOLO mask → embed → gallery_object + Qdrant).
    Ответ — сразу после сохранения в S3. Processing идёт в фоне.
    vehicle_id — опциональный идентификатор ТС из CSV."""
    from ..services.processor import process_one

    contents = await file.read()
    if not contents:
        raise HTTPException(422, "Пустой файл")

    key = f"images/{image_id}.jpg"
    existing = await asyncio.to_thread(storage.get_image, key)
    if existing is not None:
        return {"status": "skipped", "image_id": image_id, "reason": "already exists"}

    await asyncio.to_thread(storage.upload_image, key, contents)

    try:
        w_img, h_img = PILImage.open(io.BytesIO(contents)).size
    except Exception:
        w_img, h_img = 1920, 1080
    await db.save_image(image_id, w_img, h_img, status="uploaded")

    # Запускаем фоновую обработку, передаём image_data чтобы избежать S3 re-read
    asyncio.create_task(process_one(image_id, x, y, w, h, vehicle_id, image_data=contents))

    return {"status": "ok", "image_id": image_id}


@router.post("/gallery/check-images", status_code=200)
async def check_missing_images(
    query_csv: UploadFile = File(...),
    gallery_csv: UploadFile | None = File(None),
):
    """Проверяет, какие image_id из CSV отсутствуют в хранилище.
    Возвращает {all_present: bool, missing: list, total: int}."""
    from ..services.storage import storage

    q_data = await query_csv.read()
    g_data = await gallery_csv.read() if gallery_csv else b""

    all_ids = set()
    for data in [q_data, g_data]:
        if not data:
            continue
        text = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                all_ids.add(image_id)

    missing = []
    missing_bbox = {}
    missing_vehicle_id = {}
    in_gallery = 0
    # Собираем bbox из CSV для возврата (чтобы фронт мог передать при upload)
    csv_bbox = {}
    csv_vehicle_ids = {}
    for data in [q_data, g_data]:
        if not data:
            continue
        text = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            iid = (row.get("image_id") or "").strip()
            if iid:
                try:
                    csv_bbox[iid] = {"x": int(row["x"]), "y": int(row["y"]), "w": int(row["w"]), "h": int(row["h"])}
                except (TypeError, ValueError, KeyError):
                    pass
                vid = (row.get("vehicle_id") or "").strip()
                if vid:
                    csv_vehicle_ids[iid] = vid

    # Семафор: не более 8 одновременных S3-запросов, чтобы не перегрузить хранилище
    _check_sem = asyncio.Semaphore(8)

    async def _check_one(image_id: str) -> tuple[str, bool, bool]:
        """Проверяет одно изображение: (image_id, missing_in_s3, has_gallery_object)"""
        async with _check_sem:
            key = f"images/{image_id}.jpg"
            data = await asyncio.to_thread(storage.get_image, key)
            if data is None:
                data = await asyncio.to_thread(storage.get_image, f"images/{image_id}.png")
            obj = await db.get_object_by_image_id(image_id)
            return image_id, data is None, obj is not None

    results = await asyncio.gather(*[_check_one(iid) for iid in sorted(all_ids)])
    for image_id, is_missing, has_obj in results:
        if is_missing:
            missing.append(image_id)
            if image_id in csv_bbox:
                missing_bbox[image_id] = csv_bbox[image_id]
            if image_id in csv_vehicle_ids:
                missing_vehicle_id[image_id] = csv_vehicle_ids[image_id]
        if has_obj:
            in_gallery += 1

    return {
        "all_present": len(missing) == 0,
        "missing": missing,
        "missing_bbox": missing_bbox,
        "missing_vehicle_id": missing_vehicle_id,
        "all_bbox": csv_bbox,
        "all_vehicle_ids": csv_vehicle_ids,
        "total": len(all_ids),
        "with_vectors": in_gallery,
    }



@router.post("/gallery/process-pending", status_code=200)
async def process_pending():
    """Запускает обработку всех изображений со статусом 'uploaded'."""
    from ..services.processor import process_all_pending
    n = await process_all_pending()
    return {"status": "ok", "started": n}


@router.post("/gallery/processing-status", status_code=200)
async def get_processing_status(image_ids: list[str] = Body(..., embed=True)):
    """Принимает список image_id, возвращает агрегированный статус обработки.
    Используется фронтом для отслеживания прогресса векторизации батча."""
    return await db.get_processing_status_for_ids(image_ids)
