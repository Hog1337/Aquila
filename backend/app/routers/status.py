from fastapi import APIRouter

from .. import config
from ..schemas.models import StatusResponse
from ..services.database import db
from ..services.searcher import searcher

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Возвращает состояние сервиса: версию модели, размерность эмбеддинга и счётчики галереи."""
    counts = await db.get_gallery_counts()
    processing = await db.get_processing_counts()
    try:
        vectorized = searcher.count()
    except Exception as e:
        print(f"[status] searcher.count error: {type(e).__name__}: {e}", flush=True)
        vectorized = 0

    return StatusResponse(
        model_version=config.MODEL_VERSION,
        embedding_dim=config.EMBEDDING_DIM,
        gallery={
            "objects": counts["objects"],
            "vectorized": vectorized,
            "vehicles": counts.get("vehicles", 0),
            "processing_pending": processing.get("uploaded", 0),
            "processing_active": processing.get("processing", 0),
            "processing_failed": processing.get("failed", 0),
        },
    )
