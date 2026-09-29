import csv
import io
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, File, Form, Query, UploadFile

from ..schemas.models import ExportJob, ExportJobListResponse
from ..services.database import db
from ..services.storage import storage
from ..services.export import build_export_artifacts

router = APIRouter()

_active_exports: dict[str, asyncio.Task] = {}
REQUIRED_COLUMNS = ("image_id", "x", "y", "w", "h")


def _parse_query_csv(data: bytes) -> list[dict]:
    """Парсит CSV с запросами. Возвращает список {image_id, x, y, w, h, vehicle_id?}."""
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
            "vehicle_id": (row.get("vehicle_id") or "").strip() or None,
        })
    if not rows:
        raise HTTPException(422, "В CSV нет строк с аннотациями")
    return rows


@router.post("/exports", status_code=202, response_model=ExportJob)
async def start_export(
    query_csv: UploadFile = File(...),
    gallery_csv: UploadFile | None = File(None),
    threshold: float = Form(0.87),
    use_rerank: bool = Form(False),
):
    """Запускает формирование файлов экспорта (submission.csv, embeddings.npy, candidates.csv).

    Принимает: query_csv — CSV с запросами, gallery_csv — опционально CSV с подмножеством галереи
    (если не указан — поиск по всей галерее), threshold — порог, use_rerank — Query Expansion.

    Если каких-то image_id нет в хранилище → 409 с их списком."""
    q_data = await query_csv.read()
    query_rows = _parse_query_csv(q_data)
    if not query_rows:
        raise HTTPException(422, "CSV с запросами пуст")

    gallery_rows = None
    if gallery_csv:
        g_data = await gallery_csv.read()
        gallery_rows = _parse_query_csv(g_data)

    # Проверяем наличие изображений
    all_image_ids = list(set(
        [r["image_id"] for r in query_rows] +
        ([r["image_id"] for r in gallery_rows] if gallery_rows else [])
    ))
    # Собираем bbox из CSV для возврата
    bbox_map = {}
    for row in query_rows + (gallery_rows or []):
        bbox_map[row["image_id"]] = {"x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]}

    missing = []
    for image_id in all_image_ids:
        data = storage.get_image(f"images/{image_id}.jpg")
        if data is None:
            data = storage.get_image(f"images/{image_id}.png")
        if data is None:
            missing.append(image_id)
    if missing:
        missing_bbox = {iid: bbox_map[iid] for iid in missing if iid in bbox_map}
        raise HTTPException(409, detail={
            "message": f"Не хватает {len(missing)} изображений",
            "missing": missing,
            "missing_bbox": missing_bbox,
            "total": len(all_image_ids),
        })

    job_id = str(uuid.uuid4())
    await db.create_export_job(job_id, threshold)
    storage.put(f"exports/{job_id}/query.csv", q_data)
    await db.update_export_job(job_id, row_count=len(query_rows))

    task = asyncio.create_task(_run_export(job_id, query_rows, threshold, use_rerank, gallery_rows))
    _active_exports[job_id] = task
    return await get_export_job(job_id)


async def _run_export(job_id: str, query_rows: list[dict], threshold: float, use_rerank: bool, gallery_rows: list[dict] | None = None):
    try:
        await db.update_export_job(job_id, status="running", progress=5)

        async def report(done: int, total: int):
            await db.update_export_job(job_id, progress=5 + int(done / total * 90), row_count=total)

        artifacts_bytes = await build_export_artifacts(query_rows, threshold, use_rerank, on_progress=report, gallery_rows=gallery_rows)

        artifacts = []
        for name, data in artifacts_bytes.items():
            storage.upload_image(f"exports/{job_id}/{name}", data)
            artifacts.append({"name": name, "bytes": len(data)})

        await db.update_export_job(job_id, status="done", progress=100, artifacts=artifacts)
    except Exception as e:
        await db.update_export_job(job_id, status="failed", error=str(e))
    finally:
        _active_exports.pop(job_id, None)


@router.get("/exports/{job_id}", response_model=ExportJob)
async def get_export_job(job_id: str):
    job = await db.get_export_job(job_id)
    if not job:
        raise HTTPException(404, "Задание не найдено")
    return ExportJob(**job)


@router.get("/exports", response_model=ExportJobListResponse)
async def list_exports(limit: int = Query(1)):
    items = await db.list_export_jobs(limit=limit)
    return ExportJobListResponse(items=[ExportJob(**i) for i in items])


@router.get("/exports/{job_id}/files/{file_name}")
async def get_export_file(job_id: str, file_name: str):
    """Отдаёт файл-артефакт экспорта на скачивание."""
    data = storage.get_image(f"exports/{job_id}/{file_name}")
    if not data:
        raise HTTPException(404, "Файл не найден")
    from fastapi.responses import Response
    media = "text/csv" if file_name.endswith(".csv") else "application/octet-stream"
    return Response(content=data, media_type=media, headers={"Content-Disposition": f"attachment; filename={file_name}"})
