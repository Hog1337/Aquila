import asyncio
import csv
import io
import json
import re
import uuid

from fastapi import APIRouter, HTTPException, File, Form, Query, UploadFile
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError

from ..schemas.models import ImportJob, ImportJobListResponse, ImportSession
from ..services.database import db
from ..services.storage import storage
from ..services.ingest import ingest_row

router = APIRouter()

REQUIRED_COLUMNS = ("image_id", "x", "y", "w", "h")
MAX_CSV_MB = 64
MAX_IMAGE_MB = 50
MAX_CONSECUTIVE_ROW_ERRORS = 20
STORAGE_PATIENCE_S = (1, 2, 4, 8, 15, 15, 15)
IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")

_active_imports: dict[str, asyncio.Task] = {}
_known_sessions: set[str] = set()


def _key(session_id: str, name: str) -> str:
    return f"imports/{session_id}/{name}"


def _session_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise HTTPException(404, "Сессия импорта не найдена")


def _parse_rows(csv_bytes: bytes) -> list[dict]:
    try:
        text = csv_bytes.decode("utf-8-sig")
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
        if not IMAGE_ID_RE.match(image_id):
            raise HTTPException(422, f"CSV, строка {n}: некорректный image_id")
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise HTTPException(422, f"CSV, строка {n}: x, y не меньше 0, w и h больше 0")
        rows.append({
            "image_id": image_id, "x": x, "y": y, "w": w, "h": h,
            "vehicle_id": (row.get("vehicle_id") or "").strip() or None,
        })
    if not rows:
        raise HTTPException(422, "В CSV нет строк с аннотациями")
    return rows


async def _load_session(session_id: str) -> tuple[dict, list[dict]]:
    meta = await asyncio.to_thread(storage.get, _key(session_id, "meta.json"))
    raw = await asyncio.to_thread(storage.get, _key(session_id, "annotations.csv"))
    if meta is None or raw is None:
        raise HTTPException(404, "Сессия импорта не найдена")
    return json.loads(meta), _parse_rows(raw)


@router.post("/gallery/import-sessions", status_code=201, response_model=ImportSession)
async def create_import_session(annotations: UploadFile = File(...)):
    """Создаёт сессию импорта: принимает CSV с аннотациями, возвращает список нужных image_id."""
    name = annotations.filename or ""
    if not name.lower().endswith(".csv"):
        raise HTTPException(422, "annotations должен быть CSV-файлом")

    data = await annotations.read()
    if len(data) > MAX_CSV_MB * 1024 * 1024:
        raise HTTPException(413, f"CSV больше {MAX_CSV_MB} МБ")
    rows = _parse_rows(data)

    session_id = str(uuid.uuid4())
    meta = {"source_name": name, "total": len(rows)}
    await asyncio.to_thread(storage.put, _key(session_id, "annotations.csv"), data)
    await asyncio.to_thread(storage.put, _key(session_id, "meta.json"), json.dumps(meta).encode("utf-8"))
    _known_sessions.add(session_id)
    return ImportSession(
        id=session_id, source_name=name, total=len(rows),
        image_ids=sorted({r["image_id"] for r in rows}),
    )


@router.post("/gallery/import-sessions/{session_id}/images", status_code=204)
async def upload_import_image(session_id: str, image_id: str = Form(...), image: UploadFile = File(...)):
    """Загружает один кадр в сессию импорта."""
    sid = _session_id(session_id)
    if not IMAGE_ID_RE.match(image_id):
        raise HTTPException(422, "Некорректный image_id")
    if sid not in _known_sessions:
        if await asyncio.to_thread(storage.get, _key(sid, "meta.json")) is None:
            raise HTTPException(404, "Сессия импорта не найдена")
        _known_sessions.add(sid)
    data = await image.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Кадр больше {MAX_IMAGE_MB} МБ")
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(422, f"{image_id}: файл не является изображением")
    try:
        await asyncio.to_thread(storage.upload_image, f"images/{image_id}.jpg", data)
    except (BotoCoreError, ClientError, OSError) as e:
        raise HTTPException(502, f"Хранилище не приняло кадр {image_id}: {e}")


@router.post("/gallery/import-sessions/{session_id}/start", status_code=202, response_model=ImportJob)
async def start_import(session_id: str):
    """Запускает обработку сессии: инференс + запись в Qdrant + PG."""
    sid = _session_id(session_id)
    meta, rows = await _load_session(sid)
    job_id = str(uuid.uuid4())
    await db.create_import_job(job_id, meta["source_name"], len(rows))
    _active_imports[job_id] = asyncio.create_task(_run_import(job_id, rows))
    return ImportJob(id=job_id, status="queued", total=len(rows), source_name=meta["source_name"])


class StorageDown(RuntimeError):
    """Хранилище не отвечает дольше терпения."""


async def _read_frame(key: str) -> bytes | None:
    for pause in (*STORAGE_PATIENCE_S, None):
        try:
            return await asyncio.to_thread(storage.get_image, key, 1)
        except (BotoCoreError, ClientError, OSError) as e:
            if pause is None:
                raise StorageDown(f"Хранилище не отвечает: {e}")
            await asyncio.sleep(pause)


async def _run_import(job_id: str, rows: list[dict]):
    try:
        await db.update_import_job(job_id, status="running")
        missing = 0
        failed = 0
        first_error = None
        consecutive = 0
        last_id, image_data = None, None
        for i, row in enumerate(rows):
            image_id = row["image_id"]
            try:
                if image_id != last_id:
                    last_id, image_data = None, None
                    image_data = await _read_frame(f"images/{image_id}.jpg")
                    last_id = image_id
                if image_data is None:
                    missing += 1
                    print(f"[import] {image_id}: кадр не найден в S3", flush=True)
                else:
                    await ingest_row(row, image_data, job_id, row_no=i)
                consecutive = 0
            except StorageDown:
                raise
            except Exception as e:
                failed += 1
                consecutive += 1
                err_msg = f"{image_id}: {e}"
                first_error = first_error or err_msg
                import traceback
                print(f"[import] ОШИБКА {err_msg}", flush=True)
                traceback.print_exc()
                last_id = None
                if consecutive >= MAX_CONSECUTIVE_ROW_ERRORS:
                    raise RuntimeError(f"{consecutive} строк подряд с ошибкой, последняя: {image_id}: {e}")
            await db.update_import_job(job_id, progress=int((i + 1) / len(rows) * 100), processed=i + 1)

        if failed == len(rows):
            raise RuntimeError(f"Ни одна строка не обработана. Первая ошибка: {first_error}")
        notes = []
        if missing:
            notes.append(f"Строк без кадра пропущено: {missing}")
        if failed:
            notes.append(f"Строк с ошибкой пропущено: {failed} (первая: {first_error})")
        await db.update_import_job(job_id, status="done", progress=100, processed=len(rows), error="; ".join(notes) or None)
    except Exception as e:
        await db.update_import_job(job_id, status="failed", error=str(e))
    finally:
        _active_imports.pop(job_id, None)


@router.get("/gallery/imports/{job_id}", response_model=ImportJob)
async def get_import_job(job_id: str):
    """Возвращает состояние задания импорта."""
    job = await db.get_import_job(job_id)
    if not job:
        raise HTTPException(404, "Задание не найдено")
    return ImportJob(**job)


@router.get("/gallery/imports", response_model=ImportJobListResponse)
async def list_import_jobs(limit: int = Query(1), status: str | None = Query(None)):
    """Возвращает список последних импортов, опционально по статусу."""
    items = await db.list_import_jobs(limit=limit, status=status)
    return ImportJobListResponse(items=[ImportJob(**i) for i in items])


@router.delete("/gallery/imports/{job_id}", response_model=ImportJob)
async def cancel_import_job(job_id: str):
    """Отменяет задание импорта."""
    task = _active_imports.get(job_id)
    if task:
        task.cancel()
    await db.cancel_import_job(job_id)
    job = await db.get_import_job(job_id)
    return ImportJob(**job)
