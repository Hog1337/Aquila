import asyncio
import csv
import io
from typing import Awaitable, Callable

import numpy as np
from PIL import Image

from .database import db
from .searcher import searcher
from .storage import storage
from .inference_client import inference

TOP_K = 10


async def build_export_artifacts(
    query_rows: list[dict],
    threshold: float,
    use_rerank: bool = False,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    gallery_rows: list[dict] | None = None,
) -> dict[str, bytes]:
    """Формирует байты submission.csv, embeddings.npy и candidates.csv (ТЗ п.8).
    Каждый запрос обрабатывается независимо: кадр из S3 → inference → поиск по галерее.
    Эмбеддинги запросов НЕ сохраняются в Qdrant — только в embeddings.npy.
    Принимает:
        query_rows — список словарей с ключами image_id, x, y, w, h, vehicle_id?;
        threshold — порог принятия совпадений;
        use_rerank — использовать Query Expansion;
        on_progress — колбэк прогресса (обработано, всего);
        gallery_rows — опционально: список {image_id, ...} для поиска только по этим объектам.
            Если не указан — поиск по всей галерее.
    """
    if not query_rows:
        raise ValueError("Нет запросов для экспорта")

    # Определяем объекты галереи: все или только указанные в gallery_rows
    if gallery_rows:
        gallery_image_ids = [r["image_id"] for r in gallery_rows]
        gallery_objects = await db.get_all_gallery_objects()
        # Фильтруем только те, что есть в gallery_rows
        gallery_objects = [o for o in gallery_objects if o["image_id"] in gallery_image_ids]
        # Сортируем в порядке gallery_rows
        id_order = {img_id: i for i, img_id in enumerate(gallery_image_ids)}
        gallery_objects.sort(key=lambda o: id_order.get(o["image_id"], 999999))
    else:
        gallery_objects = await db.get_all_gallery_objects()

    if not gallery_objects:
        raise ValueError("Галерея пуста")

    gallery_ids = [o["id"] for o in gallery_objects]
    gallery_vectors = await asyncio.to_thread(searcher.retrieve_vectors, gallery_ids)
    missing_g = [o["image_id"] for o in gallery_objects if o["id"] not in gallery_vectors]
    if missing_g:
        raise ValueError(f"В Qdrant нет векторов для {len(missing_g)} объектов галереи")

    # Пробуем забрать эмбеддинги запросов из Qdrant (уже посчитаны при загрузке)
    # Если вектора нет — делаем инференс как fallback
    from .processor import gallery_object_id as make_obj_id
    query_obj_ids = [make_obj_id(q["image_id"], q["x"], q["y"], q["w"], q["h"]) for q in query_rows]
    cached_embs: dict[str, list[float]] = {}
    try:
        cached_embs = await asyncio.to_thread(searcher.retrieve_vectors, query_obj_ids)
    except Exception as e:
        print(f"[export] не удалось прочитать эмбеддинги из Qdrant: {e}", flush=True)

    # Строим словари vehicle_id/camera_id для junk-фильтра (если есть в CSV)
    # Junk = тот же vehicle_id И та же camera_id
    q_vid = {r["image_id"]: r.get("vehicle_id") for r in query_rows if r.get("vehicle_id")}
    q_cam = {r["image_id"]: r.get("camera_id") for r in query_rows if r.get("camera_id")}
    g_vid = {r["image_id"]: r.get("vehicle_id") for r in (gallery_rows or []) if r.get("vehicle_id")}
    g_cam = {r["image_id"]: r.get("camera_id") for r in (gallery_rows or []) if r.get("camera_id")}
    have_junk_data = bool(q_vid and g_vid)  # можно фильтровать junk только когда есть vehicle_id

    # Обрабатываем запросы: эмбеддинг из Qdrant (если есть) или inference → поиск
    query_embeddings = []
    sub_lines = []
    cand_lines = ["query_id,gallery_id,confidence\n"]

    for i, q in enumerate(query_rows):
        image_id = q["image_id"]
        x, y, w, h = q["x"], q["y"], q["w"], q["h"]

        # Берём эмбеддинг: из Qdrant (если есть) или через инференс
        obj_id = query_obj_ids[i]
        emb = cached_embs.get(obj_id)
        if emb is None:
            # Fallback: кадр из S3 + инференс
            image_data = await asyncio.to_thread(storage.get_image, f"images/{image_id}.jpg")
            if image_data is None:
                raise ValueError(f"Кадр {image_id} не найден в хранилище")
            emb, _ = await asyncio.to_thread(inference.embed, image_data, x, y, w, h)
        query_embeddings.append(emb)

        # Поиск: по всей галерее или по указанному подмножеству
        if gallery_rows:
            # Ищем 30 кандидатов, чтобы после junk-фильтра осталось 10
            hits = await asyncio.to_thread(searcher.search_by_image_ids, emb, gallery_image_ids, TOP_K * 3)
        elif use_rerank:
            hits = await asyncio.to_thread(searcher.search_rerank, emb, TOP_K * 3)
        else:
            hits = await asyncio.to_thread(searcher.search, emb, TOP_K * 3)

        # Junk-фильтр: удаляем те же vehicle_id + camera_id (если данные есть)
        if have_junk_data:
            q_vehicle_id = q_vid.get(image_id)
            q_camera_id = q_cam.get(image_id)
            clean = []
            for h in hits:
                gid = h["image_id"]
                if q_vehicle_id and g_vid.get(gid) == q_vehicle_id and g_cam.get(gid) == q_camera_id:
                    continue  # junk
                clean.append(h)
                if len(clean) >= TOP_K:
                    break
            hits = clean
        else:
            # Без vehicle_id — просто топ 10 как есть
            hits = hits[:TOP_K]

        top_ids = [h["image_id"] for h in hits[:TOP_K]]
        padded = top_ids + [""] * (TOP_K - len(top_ids))
        sub_lines.append(",".join([image_id, *padded]) + "\n")

        for h in hits:
            if h["score"] >= threshold:
                cand_lines.append(f'{image_id},{h["image_id"]},{h["score"]:.6f}\n')

        if on_progress and (i % 25 == 0 or i == len(query_rows) - 1):
            await on_progress(i + 1, len(query_rows))

    # Собираем embeddings.npy: сначала все query (в порядке CSV), потом вся галерея (по row_no)
    gallery_emb_list = [gallery_vectors[oid] for oid in gallery_ids]
    emb_matrix = np.array(query_embeddings + gallery_emb_list, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, emb_matrix)

    return {
        "submission.csv": "".join(sub_lines).encode(),
        "embeddings.npy": buf.getvalue(),
        "candidates.csv": "".join(cand_lines).encode(),
    }
