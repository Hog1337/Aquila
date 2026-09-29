"""Асинхронная обработка изображений: crop → YOLO mask → embed → gallery_object + Qdrant.

Используется:
  - после upload-single-image (фоновый task)
  - при process-pending (догоняет незавершённые)
"""
import asyncio
import base64
import time
import uuid
from io import BytesIO

from PIL import Image as PILImage

from ..services.database import db
from ..services.inference_client import inference
from ..services.searcher import searcher
from ..services.storage import storage

# Максимум параллельных задач обработки
_PROCESS_SEM = asyncio.Semaphore(8)
# Регистрируем активные задачи, чтобы не запустить дубли
_pending: set[str] = set()

# Флаг: использовать батч-инференс (буферизация запросов)
_USE_BATCH_INFERENCE = False

# Буфер батч-инференса: (image_id, x, y, w, h, image_base64)
_INF_BUF: list[tuple[str, int, int, int, int, str]] = []
_INF_RESULTS: dict[str, list[float]] = {}
_INF_LOCK = asyncio.Lock()
_INF_FLUSH_INTERVAL = 0.15  # сек
_INF_BATCH_SIZE = 8
_INF_EVENT = asyncio.Event()  # сигнал флашеру: буфер непустой

# Буфер для батчевого Qdrant upsert
_QDRANT_BATCH: list[tuple[str, list[float], dict]] = []
_QDRANT_LOCK = asyncio.Lock()
_QDRANT_BATCH_SIZE = 200
_QDRANT_LAST_FLUSH = 0.0  # time.monotonic() последнего сброса
_QDRANT_EVENT = asyncio.Event()  # сигнал флашеру: буфер непустой
_QDRANT_FLUSH_INTERVAL = 2.0  # фоновая проверка каждые 2с


def gallery_object_id(image_id: str, x: int, y: int, w: int, h: int) -> str:
    """Стабильный ID gallery_object для пары (image_id, bbox)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{image_id}:{x}:{y}:{w}:{h}"))


def set_batch_inference(enabled: bool = True):
    """Включает/выключает батч-инференс (буферизация запросов)."""
    global _USE_BATCH_INFERENCE
    _USE_BATCH_INFERENCE = enabled


def start_inference_flusher():
    """Запускает фоновый сброс буфера батч-инференса."""
    asyncio.create_task(_inference_flusher_loop())


async def flush_inference_batch():
    """Принудительный сброс буфера. Вызывается перед экспортом."""
    batch = None
    async with _INF_LOCK:
        if _INF_BUF:
            batch = _INF_BUF[:]
            _INF_BUF.clear()
    if not batch:
        return 0
    try:
        embs = await asyncio.to_thread(inference.embed_batch_full, batch)
        async with _INF_LOCK:
            for (iid, _, _, _, _, _), emb in zip(batch, embs):
                _INF_RESULTS[iid] = emb
        return len(batch)
    except Exception as e:
        print(f"[processor] batch inference error: {e}", flush=True)
        async with _INF_LOCK:
            _INF_BUF.extend(batch)
        return 0


async def _inference_flusher_loop():
    """Фоновая задача: каждые 0.5с сбрасывает буфер."""
    print('[processor] inference flusher started', flush=True)
    while _USE_BATCH_INFERENCE:
        # Ждём сигнал или таймаут (0.15s)
        try:
            await asyncio.wait_for(_INF_EVENT.wait(), timeout=_INF_FLUSH_INTERVAL)
        except asyncio.TimeoutError:
            pass
        _INF_EVENT.clear()
        async with _INF_LOCK:
            if not _INF_BUF:
                continue
            batch = _INF_BUF[:]
            _INF_BUF.clear()
        if batch:
            try:
                embs = await asyncio.to_thread(inference.embed_batch_full, batch)
                async with _INF_LOCK:
                    for (iid, _, _, _, _, _), emb in zip(batch, embs):
                        _INF_RESULTS[iid] = emb
            except Exception as e:
                import traceback
                print(f"[processor] inference flush error (batch_size={len(batch)}): {e}", flush=True)
                traceback.print_exc()
                async with _INF_LOCK:
                    _INF_BUF.extend(batch)
                await asyncio.sleep(1.0)


async def _inference_add(image_id: str, x: int, y: int, w: int, h: int, image_data: bytes) -> bool:
    """Добавляет изображение в буфер батч-инференса.
    Возвращает True, если результат уже готов (был в кэше флаша).
    """
    img_b64 = base64.b64encode(image_data).decode('ascii')
    async with _INF_LOCK:
        if image_id in _INF_RESULTS:
            return True  # уже готов
        _INF_BUF.append((image_id, x, y, w, h, img_b64))
    _INF_EVENT.set()  # будить флашер
    return False


async def _inference_wait(image_id: str, timeout: float = 120.0) -> list[float]:
    """Ожидает появления эмбеддинга в результатах."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with _INF_LOCK:
            if image_id in _INF_RESULTS:
                return _INF_RESULTS.pop(image_id)
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Inference timeout for {image_id} after {timeout:.0f}s")


async def _qdrant_flush_now() -> int:
    """Принудительный сброс буфера Qdrant. Возвращает количество сброшенных точек."""
    global _QDRANT_BATCH, _QDRANT_LAST_FLUSH
    batch = None
    async with _QDRANT_LOCK:
        if _QDRANT_BATCH:
            batch = _QDRANT_BATCH[:]
            _QDRANT_BATCH = []
            _QDRANT_LAST_FLUSH = time.monotonic()
    if batch:
        await asyncio.to_thread(searcher.upsert_batch, batch)
        return len(batch)
    return 0


async def _qdrant_flusher_loop():
    """Фоновая задача: каждые 2 с сбрасывает буфер Qdrant."""
    print('[processor] qdrant flusher started', flush=True)
    while True:
        try:
            await asyncio.wait_for(_QDRANT_EVENT.wait(), timeout=_QDRANT_FLUSH_INTERVAL)
        except asyncio.TimeoutError:
            pass
        _QDRANT_EVENT.clear()
        n = await _qdrant_flush_now()
        if n:
            print(f'[processor] qdrant flusher: flushed {n} points', flush=True)


async def _qdrant_upsert(point_id: str, embedding: list[float], payload: dict):
    """Добавляет точку в буфер батчевого upsert.
    Сигналит фоновому флашеру."""
    global _QDRANT_BATCH, _QDRANT_LAST_FLUSH
    async with _QDRANT_LOCK:
        _QDRANT_BATCH.append((point_id, embedding, payload))
    _QDRANT_EVENT.set()  # будить флашер


async def flush_qdrant_batch():
    """Сбрасывает оставшиеся точки из буфера в Qdrant.
    Вызывается перед экспортом, чтобы все векторы были доступны для поиска."""
    return await _qdrant_flush_now()


def start_qdrant_flusher():
    """Запускает фоновый флашер Qdrant."""
    asyncio.create_task(_qdrant_flusher_loop())


async def process_one(image_id: str, x: int = 0, y: int = 0, w: int = 0, h: int = 0,
                      vehicle_id: str | None = None,
                      image_data: bytes | None = None):
    """Обрабатывает одно изображение: embed, сохраняет gallery_object + Qdrant.

    Вызывается как фоновая задача (asyncio.create_task).
    Если bbox = (0,0,0,0) — используется full-frame bbox из размеров кадра.
    vehicle_id — опциональный идентификатор ТС из CSV.
    image_data — если передан, S3 чтение пропускается (используется в CLI).
    """
    if image_id in _pending:
        return  # уже в обработке
    _pending.add(image_id)

    async with _PROCESS_SEM:
        try:
            print(f"[processor] start {image_id} (bbox={x},{y},{w},{h})", flush=True)
            await db.update_image_status(image_id, "processing")

            # Загружаем кадр: из переданных bytes (CLI) или из S3 (веб)
            if image_data is not None:
                data = image_data
            else:
                data = await asyncio.to_thread(storage.get_image, f"images/{image_id}.jpg")
                if data is None:
                    data = await asyncio.to_thread(storage.get_image, f"images/{image_id}.png")
                if data is None:
                    await db.update_image_status(image_id, "failed", "Изображение не найдено в S3")
                    return

            # Определяем размеры и bbox
            try:
                pil = PILImage.open(BytesIO(data))
                w_img, h_img = pil.size
            except Exception:
                w_img, h_img = 1920, 1080

            if w <= 0 or h <= 0:
                # bbox не передан — используем full-frame
                bx, by, bw, bh = 0, 0, w_img, h_img
            else:
                bx, by, bw, bh = x, y, w, h

            # Embed: батч-буфер или индивидуальный
            if _USE_BATCH_INFERENCE:
                ready = await _inference_add(image_id, bx, by, bw, bh, data)
                if not ready:
                    emb = await _inference_wait(image_id)
                else:
                    async with _INF_LOCK:
                        emb = _INF_RESULTS.pop(image_id)
            else:
                emb, _ = await asyncio.to_thread(inference.embed, data, bx, by, bw, bh)

            # Регистрируем image (если ещё нет)
            await db.save_image(image_id, w_img, h_img, status="processing")

            # Создаём gallery_object и добавляем в буфер Qdrant
            obj_id = gallery_object_id(image_id, bx, by, bw, bh)
            bbox = {"x": bx, "y": by, "w": bw, "h": bh}
            await db.save_gallery_object(obj_id, image_id, bbox, vehicle_id, None)
            await _qdrant_upsert(obj_id, emb, {
                "image_id": image_id,
                "bbox": bbox,
                "vehicle_id": vehicle_id,
                "batch_id": None,
            })

            await db.update_image_status(image_id, "ready")
            print(f"[processor] done {image_id}", flush=True)
        except Exception as e:
            print(f"[processor] error for {image_id}: {e}", flush=True)
            await db.update_image_status(image_id, "failed", str(e))
        finally:
            _pending.discard(image_id)


async def process_all_pending(max_concurrent: int = 4) -> int:
    """Запускает обработку для всех изображений со статусом 'uploaded'.
    Возвращает количество запущенных задач."""
    pending = await db.get_images_by_status("uploaded")
    if not pending:
        return 0

    print(f"[processor] Запускаю обработку {len(pending)} изображений...", flush=True)
    tasks = []
    for img in pending:
        task = asyncio.create_task(process_one(img["image_id"]))
        tasks.append(task)
        # Небольшая задержка, чтобы не завалить S3/inference
        await asyncio.sleep(0.05)

    # Дожидаемся завершения всех задач и сбрасываем буфер Qdrant
    if tasks:
        await asyncio.gather(*tasks)
        await flush_qdrant_batch()
    return len(pending)
