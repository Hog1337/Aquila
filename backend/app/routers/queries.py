from __future__ import annotations

import uuid
import time
import io

import httpx
from fastapi import APIRouter, File, Form, Query, UploadFile, HTTPException
from fastapi.responses import Response, PlainTextResponse
from PIL import Image

from .. import config
from ..schemas.models import SearchResult, Candidate, Timing, QueryListResponse, QueryListItem, QuerySummary
from ..services.database import db
from ..services.storage import storage
from ..services.searcher import searcher
from ..services.inference_client import inference

router = APIRouter()


@router.post("/queries", response_model=SearchResult)
async def create_query(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    top_n: int = Form(10),
    threshold: float | None = Form(None),
    rerank: bool = Form(True),
):
    """Выполняет поисковый запрос: считает эмбеддинг объекта, ищет кандидатов и сохраняет результат.
    Принимает: image — файл кадра, x, y, w, h — bbox объекта, top_n — число кандидатов,
    threshold — порог принятия (по умолчанию текущий), rerank — включить Query Expansion."""
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(422, "Допустимы только JPEG и PNG")
    contents = await image.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(422, "Файл больше 50 МБ")

    img = Image.open(io.BytesIO(contents))
    if x + w > img.width or y + h > img.height:
        raise HTTPException(422, "Рамка выходит за пределы кадра")

    query_id = str(uuid.uuid4())
    query_image_key = f"queries/{query_id}.jpg"
    storage.upload_image(query_image_key, contents)

    if threshold is None:
        th = await db.get_current_threshold()
        threshold = th["value"]

    t0 = time.time()
    emb, inference_ms = inference.embed(contents, x, y, w, h)

    t1 = time.time()
    if rerank:
        hits = searcher.search_rerank(emb, top_n)
    else:
        hits = searcher.search(emb, top_n)
    search_ms = int((time.time() - t1) * 1000)

    gallery_size = searcher.count()

    candidates = []
    best_score = None
    accepted_count = 0
    for i, hit in enumerate(hits):
        score = hit["score"]
        if best_score is None or score > best_score:
            best_score = score
        accepted = score >= threshold
        if accepted:
            accepted_count += 1
        candidates.append(Candidate(
            rank=i + 1,
            object_id=hit["id"],
            image_id=hit.get("image_id", ""),
            vehicle_id=hit.get("vehicle_id"),
            score=score,
            accepted=accepted,
        ))

    elapsed_ms = int((time.time() - t0) * 1000)

    await db.save_query(
        query_id=query_id,
        bbox={"x": x, "y": y, "w": w, "h": h},
        threshold=threshold,
        top_n=top_n,
        best_score=best_score,
        accepted_count=accepted_count,
        elapsed_ms=elapsed_ms,
        model_version=config.MODEL_VERSION,
        embedding=emb,
        query_image_key=query_image_key,
    )
    if candidates:
        await db.save_results(query_id, [c.model_dump() for c in candidates])

    return SearchResult(
        query_id=query_id,
        model_version=config.MODEL_VERSION,
        threshold=threshold,
        top_n=top_n,
        bbox={"x": x, "y": y, "w": w, "h": h},
        best_score=best_score,
        elapsed_ms=elapsed_ms,
        timing=Timing(inference_ms=inference_ms, search_ms=search_ms),
        gallery_size=gallery_size,
        candidates=candidates,
        reranked=rerank,
    )


@router.get("/queries", response_model=QueryListResponse)
async def list_queries(
    refused: str | None = Query(None),
    limit: int = Query(20),
    offset: int = Query(0),
):
    """Возвращает страницу истории запросов со сводкой за сегодня.
    Принимает: refused — фильтр "только отклонённые" ("true"/иначе все), limit — размер страницы, offset — смещение."""
    refused_only = refused == "true"
    items, total, summary = await db.list_queries(limit=limit, offset=offset, refused_only=refused_only)
    return QueryListResponse(
        total=total,
        summary=QuerySummary(**summary),
        items=[QueryListItem(**i) for i in items],
    )


@router.get("/queries/history.csv")
async def get_history_csv(refused: str | None = Query(None)):
    """Отдаёт историю запросов в формате CSV. Принимает: refused — фильтр "только отклонённые" ("true"/иначе все)."""
    refused_only = refused == "true"
    items, _, _ = await db.list_queries(limit=10000, offset=0, refused_only=refused_only)
    lines = ["id,created_at,best_score,accepted_count,refused,elapsed_ms,threshold\n"]
    for i in items:
        lines.append(f'{i["id"]},{i["created_at"]},{i.get("best_score","")},{i["accepted_count"]},{i["refused"]},{i["elapsed_ms"]},{i["threshold"]}\n')
    return PlainTextResponse("".join(lines), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=history.csv"})


@router.get("/queries/{query_id}", response_model=SearchResult)
async def get_query(query_id: str):
    """Возвращает сохранённый результат запроса по его id. Принимает: query_id — id запроса."""
    result = await db.get_query(query_id)
    if not result:
        raise HTTPException(404, "Запрос не найден")
    return SearchResult(**result)


@router.get("/queries/{query_id}/image")
async def get_query_image(query_id: str):
    """Возвращает исходный кадр запроса. Принимает: query_id — id запроса."""
    key = await db.get_query_image_key(query_id)
    if not key:
        raise HTTPException(404, "Кадр не найден")
    data = storage.get_image(key)
    if not data:
        raise HTTPException(404, "Кадр не найден")
    return Response(content=data, media_type="image/jpeg")


@router.get("/queries/{query_id}/crop")
async def get_query_crop(query_id: str, size: int | None = Query(None)):
    """Возвращает кроп объекта запроса по его bbox.
    Принимает: query_id — id запроса, size — сторона миниатюры в пикселях (опционально)."""
    key = await db.get_query_image_key(query_id)
    if not key:
        raise HTTPException(404, "Кадр не найден")
    data = storage.get_image(key)
    if not data:
        raise HTTPException(404, "Кадр не найден")
    bbox = await db.get_query_bbox(query_id)
    if not bbox:
        raise HTTPException(404, "BBox не найден")
    img = Image.open(io.BytesIO(data)).crop((bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]))
    if size:
        img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.get("/queries/{query_id}/candidates.csv")
async def get_candidates_csv(query_id: str, threshold: float | None = Query(None)):
    """Отдаёт кандидатов запроса в формате CSV.
    Принимает: query_id — id запроса, threshold — порог принятия для колонки accepted (по умолчанию порог запроса)."""
    result = await db.get_query(query_id)
    if not result:
        raise HTTPException(404, "Запрос не найден")
    th = threshold if threshold is not None else result["threshold"]
    lines = ["rank,image_id,vehicle_id,score,accepted\n"]
    for c in result["candidates"]:
        accepted = c["score"] >= th
        lines.append(f'{c["rank"]},{c["image_id"]},{c.get("vehicle_id","")},{c["score"]:.4f},{accepted}\n')
    return PlainTextResponse("".join(lines), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=candidates.csv"})


@router.get("/queries/{query_id}/candidates/{rank:int}/gradcam")
async def get_gradcam(query_id: str, rank: int):
    """Строит Grad-CAP (градиентную карту релевантности) для пары запрос-кандидат.
    Показывает, изменение каких пикселей наиболее сильно повлияло бы на сходство.
    Принимает: query_id — id запроса, rank — позиция кандидата в списке результатов."""
    q_key = await db.get_query_image_key(query_id)
    if not q_key:
        raise HTTPException(404, "Запрос не найден")
    q_data = storage.get_image(q_key)
    if not q_data:
        raise HTTPException(404, "Кадр запроса не найден")

    q_bbox = await db.get_query_bbox(query_id)
    if not q_bbox:
        raise HTTPException(404, "BBox не найден")

    result = await db.get_query(query_id)
    if not result:
        raise HTTPException(404, "Результат не найден")
    cands = result.get("candidates", [])
    if rank < 1 or rank > len(cands):
        raise HTTPException(404, f"Кандидат #{rank} не найден")
    cand = cands[rank - 1]
    cand_obj = await db.get_object_by_id(cand["object_id"]) if cand.get("object_id") else None
    if not cand_obj:
        raise HTTPException(404, "Объект кандидата не найден")

    cand_data = storage.get_image(f"images/{cand_obj['image_id']}.jpg")
    if not cand_data:
        raise HTTPException(404, "Кадр кандидата не найден")

    q_img = Image.open(io.BytesIO(q_data))
    q_crop = q_img.crop((q_bbox["x"], q_bbox["y"], q_bbox["x"] + q_bbox["w"], q_bbox["y"] + q_bbox["h"]))
    c_bbox = cand_obj["bbox"]
    c_img = Image.open(io.BytesIO(cand_data))
    c_crop = c_img.crop((c_bbox["x"], c_bbox["y"], c_bbox["x"] + c_bbox["w"], c_bbox["y"] + c_bbox["h"]))

    # Градиентная карта релевантности через /internal/compare
    buf_q = io.BytesIO()
    q_crop.save(buf_q, "JPEG", quality=95)
    buf_c = io.BytesIO()
    c_crop.save(buf_c, "JPEG", quality=95)

    resp = httpx.post(
        f"{config.INFERENCE_URL}/internal/compare",
        files={
            "query_image": ("q.jpg", buf_q.getvalue(), "image/jpeg"),
            "candidate_image": ("c.jpg", buf_c.getvalue(), "image/jpeg"),
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "query_overlay_url": data["query_overlay_url"],
        "candidate_overlay_url": data["candidate_overlay_url"],
        "similarity": data["similarity"],
        "regions": data.get("regions", []),
        "note": "Grad-ATTN: градиент косинусного сходства, показывающий, "
                "изменение каких пикселей наиболее сильно повлияло бы на "
                "близость эмбеддингов.",
    }
