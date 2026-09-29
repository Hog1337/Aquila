from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .services.database import db
from .routers import status, threshold, queries, objects, gallery, imports, metrics, exports


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    print(f"[backend] connected to postgres, qdrant={config.QDRANT_HOST}:{config.QDRANT_PORT}, s3={config.S3_ENDPOINT}, inference={config.INFERENCE_URL}")
    # Включаем батч-инференс (TV.Resize antialias=True совместим с HuggingFace)
    from .services.processor import set_batch_inference, start_inference_flusher, start_qdrant_flusher
    set_batch_inference(True)
    start_inference_flusher()
    start_qdrant_flusher()
    print(f"[backend] batch inference + qdrant flusher enabled")
    yield
    await db.close()


app = FastAPI(title="Falcon ReID Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(threshold.router, prefix="/api/v1", tags=["threshold"])
app.include_router(queries.router, prefix="/api/v1", tags=["queries"])
app.include_router(objects.router, prefix="/api/v1", tags=["objects"])
app.include_router(gallery.router, prefix="/api/v1", tags=["gallery"])
app.include_router(imports.router, prefix="/api/v1", tags=["imports"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
app.include_router(exports.router, prefix="/api/v1", tags=["exports"])
