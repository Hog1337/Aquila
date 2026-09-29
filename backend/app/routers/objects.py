import io

from fastapi import APIRouter, HTTPException, Query
from PIL import Image

from ..services.database import db
from ..services.storage import storage

router = APIRouter()


@router.get("/objects/{object_id}/crop")
async def get_object_crop(object_id: str, size: int | None = Query(None)):
    """Возвращает кроп объекта галереи по его bbox.
    Принимает: object_id — id объекта, size — сторона миниатюры в пикселях (опционально)."""
    obj = await db.get_object_by_id(object_id)
    if not obj:
        raise HTTPException(404, "Объект не найден")
    data = storage.get_image(f"images/{obj['image_id']}.jpg")
    if not data:
        raise HTTPException(404, "Кадр не найден")
    bbox = obj["bbox"]
    img = Image.open(io.BytesIO(data)).crop((bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]))
    if size:
        img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/jpeg")
