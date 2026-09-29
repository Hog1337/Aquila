from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Candidate(BaseModel):
    rank: int
    object_id: str | None = None
    image_id: str
    vehicle_id: str | None = None
    score: float
    accepted: bool


class Timing(BaseModel):
    inference_ms: int = 0
    search_ms: int = 0


class SearchResult(BaseModel):
    query_id: str
    created_at: datetime | None = None
    model_version: str = "FALCON_FT_best"
    threshold: float
    top_n: int = 10
    bbox: BBox
    best_score: float | None = None
    elapsed_ms: int = 0
    timing: Timing = Timing()
    gallery_size: int = 0
    candidates: list[Candidate] = []
    reranked: bool = False


class QueryListItem(BaseModel):
    id: str
    created_at: datetime
    best_score: float | None = None
    accepted_count: int = 0
    refused: bool = False
    elapsed_ms: int = 0
    threshold: float


class QuerySummary(BaseModel):
    today_count: int = 0
    refused_share: float = 0.0
    median_ms: int = 0


class QueryListResponse(BaseModel):
    total: int
    summary: QuerySummary
    items: list[QueryListItem]


class ThresholdOut(BaseModel):
    value: float
    reason: str | None = None
    changed_by: str | None = None
    created_at: datetime | None = None


class ThresholdIn(BaseModel):
    value: float = Field(..., ge=-1, le=1)
    reason: str = ""


class GalleryObject(BaseModel):
    id: str
    image_id: str
    vehicle_id: str | None = None
    bbox: BBox


class GalleryListResponse(BaseModel):
    total: int
    items: list[GalleryObject]


class StatusResponse(BaseModel):
    model_version: str = "FALCON_FT_best"
    embedding_dim: int = 2048
    gallery: dict[str, int] = {}


class ImportJob(BaseModel):
    id: str
    status: str = "queued"
    progress: int = 0
    processed: int = 0
    total: int = 0
    source_name: str | None = None
    created_at: datetime | None = None
    error: str | None = None


class ImportSession(BaseModel):
    id: str
    source_name: str
    total: int
    image_ids: list[str]


class ImportJobListResponse(BaseModel):
    items: list[ImportJob]


class CurvePoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    tnr: float


class MetricRun(BaseModel):
    id: int | str
    created_at: datetime | None = None
    model_version: str = "FALCON_FT_best"
    threshold: float | None = None
    map: float | None = None
    rank1: float | None = None
    rank5: float | None = None
    f1: float | None = None
    tnr: float | None = None
    pr_auc: float | None = None
    n_queries: int = 0
    details: dict[str, Any] = {}


class MetricJob(BaseModel):
    id: str
    status: str = "queued"
    progress: int = 0
    error: str | None = None
    run: MetricRun | None = None


class ExportArtifact(BaseModel):
    name: str
    bytes: int = 0


class ExportJob(BaseModel):
    id: str
    status: str = "queued"
    progress: int = 0
    query_batch_id: str
    query_source: str | None = None
    threshold: float = 0.7
    row_count: int | None = None
    artifacts: list[ExportArtifact] = []
    error: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None


class ExportJobListResponse(BaseModel):
    items: list[ExportJob]


class ExportIn(BaseModel):
    threshold: float = 0.7
    query_batch_id: str
