import asyncio
import io
import uuid

from PIL import Image

from .database import db
from .searcher import searcher
from .inference_client import inference


async def ingest_row(row: dict, image_data: bytes, batch_id: str, row_no: int) -> tuple[int, int]:
    """Считает эмбеддинг одного объекта и сохраняет его в PG и Qdrant.
    Кадр должен уже лежать в хранилище (storage.upload_image).
    Принимает: row — {image_id, x, y, w, h, vehicle_id?}, image_data — байты кадра,
    batch_id — id партии импорта, row_no — номер строки в CSV.
    Возвращает: (width, height) кадра."""
    image_id, x, y, w, h = row["image_id"], row["x"], row["y"], row["w"], row["h"]
    vehicle_id = row.get("vehicle_id")
    obj_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{image_id}:{x}:{y}:{w}:{h}"))
    size = Image.open(io.BytesIO(image_data)).size

    emb, _ = await asyncio.to_thread(inference.embed, image_data, x, y, w, h)

    await db.save_image(image_id, size[0], size[1])
    await db.save_gallery_object(obj_id, image_id, {"x": x, "y": y, "w": w, "h": h}, vehicle_id, batch_id, row_no=row_no)
    await asyncio.to_thread(searcher.upsert, obj_id, emb, {
        "image_id": image_id,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "vehicle_id": vehicle_id,
        "batch_id": batch_id,
    })
    return size
