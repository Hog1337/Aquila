from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import asyncpg

from .. import config


async def _init_connection(conn: asyncpg.Connection):
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


class Database:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(config.POSTGRES_DSN, min_size=2, max_size=10, init=_init_connection)

    async def close(self):
        if self.pool:
            await self.pool.close()

    # ---- Threshold ----

    async def get_current_threshold(self) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT value, reason, changed_by, created_at FROM reid.current_threshold")
            if row:
                return {"value": row["value"], "reason": row["reason"], "changed_by": row["changed_by"], "created_at": row["created_at"].isoformat()}
            return {"value": config.DEFAULT_THRESHOLD, "reason": "default", "changed_by": None, "created_at": None}

    async def save_threshold(self, value: float, reason: str):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO reid.threshold_history (value, reason, changed_by) VALUES ($1, $2, 'backend') RETURNING value, reason, changed_by, created_at",
                value, reason,
            )
            return {"value": row["value"], "reason": row["reason"], "changed_by": row["changed_by"], "created_at": row["created_at"].isoformat()}

    # ---- Gallery (единая галерея, без ролей) ----

    async def get_gallery_counts(self) -> dict:
        async with self.pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM reid.gallery_objects")
            img = await conn.fetchval("SELECT COUNT(*) FROM reid.images")
            veh = await conn.fetchval("SELECT COUNT(DISTINCT vehicle_id) FROM reid.gallery_objects WHERE vehicle_id IS NOT NULL")
            return {"objects": total, "images": img, "vehicles": veh}

    async def get_gallery_objects(self, vehicle_id: str | None = None, limit: int = 24, offset: int = 0) -> tuple[list[dict], int]:
        async with self.pool.acquire() as conn:
            if vehicle_id:
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM reid.gallery_objects WHERE vehicle_id ILIKE $1",
                    f"%{vehicle_id}%",
                )
                rows = await conn.fetch(
                    "SELECT id, image_id, vehicle_id, bbox_x, bbox_y, bbox_w, bbox_h "
                    "FROM reid.gallery_objects WHERE vehicle_id ILIKE $1 "
                    "ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                    f"%{vehicle_id}%", limit, offset,
                )
            else:
                total = await conn.fetchval("SELECT COUNT(*) FROM reid.gallery_objects")
                rows = await conn.fetch(
                    "SELECT id, image_id, vehicle_id, bbox_x, bbox_y, bbox_w, bbox_h "
                    "FROM reid.gallery_objects ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset,
                )
            items = [
                {"id": str(r["id"]), "image_id": r["image_id"], "vehicle_id": r["vehicle_id"],
                 "bbox": {"x": r["bbox_x"], "y": r["bbox_y"], "w": r["bbox_w"], "h": r["bbox_h"]}}
                for r in rows
            ]
            return items, total

    async def get_object_by_id(self, object_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT image_id, bbox_x, bbox_y, bbox_w, bbox_h FROM reid.gallery_objects WHERE id = $1",
                uuid.UUID(object_id),
            )
            if row:
                return {"image_id": row["image_id"], "bbox": {"x": row["bbox_x"], "y": row["bbox_y"], "w": row["bbox_w"], "h": row["bbox_h"]}}
            return None

    async def get_object_by_image_id(self, image_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, bbox_x, bbox_y, bbox_w, bbox_h FROM reid.gallery_objects WHERE image_id = $1 LIMIT 1",
                image_id,
            )
            if row:
                return {"id": str(row["id"]), "bbox": {"x": row["bbox_x"], "y": row["bbox_y"], "w": row["bbox_w"], "h": row["bbox_h"]}}
            return None

    async def save_image(self, image_id: str, width: int = 0, height: int = 0, status: str = 'ready'):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO reid.images (image_id, width, height, processing_status) VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (image_id) DO UPDATE SET processing_status = $4",
                image_id, width, height, status,
            )

    async def update_image_status(self, image_id: str, status: str, error: str | None = None):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE reid.images SET processing_status = $2, processing_error = $3 WHERE image_id = $1",
                image_id, status, error,
            )

    async def get_images_by_status(self, status: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT i.image_id, i.width, i.height
                   FROM reid.images i
                   LEFT JOIN reid.gallery_objects g ON g.image_id = i.image_id
                   WHERE i.processing_status = $1 AND g.id IS NULL""",
                status,
            )
            return [dict(r) for r in rows]

    async def get_image_status(self, image_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT image_id, processing_status, processing_error FROM reid.images WHERE image_id = $1",
                image_id,
            )
            return dict(row) if row else None

    async def get_processing_counts(self) -> dict:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT i.processing_status, COUNT(*)::int AS cnt
                   FROM reid.images i
                   LEFT JOIN reid.gallery_objects g ON g.image_id = i.image_id
                   WHERE g.id IS NULL
                   GROUP BY i.processing_status"""
            )
            counts = {r["processing_status"]: r["cnt"] for r in rows}
            return {
                "uploaded": counts.get("uploaded", 0),
                "processing": counts.get("processing", 0),
                "ready": counts.get("ready", 0),
                "failed": counts.get("failed", 0),
            }

    async def save_gallery_object(self, id: str, image_id: str, bbox: dict, vehicle_id: str | None, batch_id: str | None, row_no: int | None = None):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO reid.gallery_objects (id, image_id, bbox_x, bbox_y, bbox_w, bbox_h, vehicle_id, batch_id, row_no)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                   ON CONFLICT (id) DO UPDATE SET vehicle_id = $7, batch_id = $8, row_no = $9""",
                uuid.UUID(id), image_id, bbox["x"], bbox["y"], bbox["w"], bbox["h"], vehicle_id,
                uuid.UUID(batch_id) if batch_id else None, row_no,
            )

    # ---- Search queries ----

    async def save_query(self, query_id: str, bbox: dict, threshold: float, top_n: int, best_score: float | None,
                         accepted_count: int, elapsed_ms: int, model_version: str, embedding: list[float] | None, query_image_key: str | None) -> str:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO reid.search_queries (id, bbox_x, bbox_y, bbox_w, bbox_h, threshold, top_n, best_score, accepted_count, elapsed_ms, model_version, embedding, query_image_key)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)""",
                uuid.UUID(query_id), bbox["x"], bbox["y"], bbox["w"], bbox["h"], threshold, top_n, best_score, accepted_count, elapsed_ms, model_version, embedding, query_image_key,
            )
            return query_id

    async def save_results(self, query_id: str, candidates: list[dict]):
        async with self.pool.acquire() as conn:
            for c in candidates:
                oid = uuid.UUID(c["object_id"]) if c.get("object_id") else None
                await conn.execute(
                    "INSERT INTO reid.search_results (query_id, rank, object_id, image_id, vehicle_id, score, accepted) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    uuid.UUID(query_id), c["rank"], oid, c["image_id"], c.get("vehicle_id"), c["score"], c["accepted"],
                )

    async def get_query(self, query_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reid.search_queries WHERE id = $1", uuid.UUID(query_id))
            if not row:
                return None
            candidates = []
            for r in await conn.fetch("SELECT * FROM reid.search_results WHERE query_id = $1 ORDER BY rank", uuid.UUID(query_id)):
                candidates.append({
                    "rank": r["rank"], "object_id": str(r["object_id"]) if r["object_id"] else None,
                    "image_id": r["image_id"], "vehicle_id": r["vehicle_id"], "score": r["score"], "accepted": r["accepted"],
                })
            return {
                "query_id": str(row["id"]),
                "created_at": row["created_at"].isoformat(),
                "model_version": row["model_version"],
                "threshold": row["threshold"],
                "top_n": row["top_n"],
                "bbox": {"x": row["bbox_x"], "y": row["bbox_y"], "w": row["bbox_w"], "h": row["bbox_h"]},
                "best_score": row["best_score"],
                "elapsed_ms": row["elapsed_ms"],
                "candidates": candidates,
            }

    async def list_queries(self, limit: int = 20, offset: int = 0, refused_only: bool = False) -> tuple[list[dict], int, dict]:
        async with self.pool.acquire() as conn:
            w = "refused = TRUE" if refused_only else "TRUE"
            total = await conn.fetchval(f"SELECT COUNT(*) FROM reid.search_queries WHERE {w}")
            rows = await conn.fetch(
                f"SELECT id, created_at, best_score, accepted_count, refused, elapsed_ms, threshold FROM reid.search_queries WHERE {w} ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit, offset,
            )
            items = [
                {"id": str(r["id"]), "created_at": r["created_at"].isoformat(), "best_score": r["best_score"],
                 "accepted_count": r["accepted_count"], "refused": r["refused"], "elapsed_ms": r["elapsed_ms"], "threshold": r["threshold"]}
                for r in rows
            ]
            today = await conn.fetchval(
                "SELECT COUNT(*) FROM reid.search_queries WHERE created_at >= $1",
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
            )
            refused_today = await conn.fetchval(
                "SELECT COUNT(*) FROM reid.search_queries WHERE created_at >= $1 AND refused = TRUE",
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
            )
            median = await conn.fetchval(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms) FROM reid.search_queries WHERE created_at >= $1 AND elapsed_ms IS NOT NULL",
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
            )
            summary = {"today_count": today, "refused_share": refused_today / today if today else 0.0, "median_ms": median or 0}
            return items, total, summary

    async def get_query_image_key(self, query_id: str) -> str | None:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT query_image_key FROM reid.search_queries WHERE id = $1", uuid.UUID(query_id))

    async def get_query_bbox(self, query_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT bbox_x, bbox_y, bbox_w, bbox_h FROM reid.search_queries WHERE id = $1", uuid.UUID(query_id))
            if row:
                return {"x": row["bbox_x"], "y": row["bbox_y"], "w": row["bbox_w"], "h": row["bbox_h"]}
            return None

    # ---- Import jobs ----

    async def create_import_job(self, job_id: str, source_name: str, total: int) -> str:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO reid.import_jobs (id, source_name, total) VALUES ($1, $2, $3)",
                uuid.UUID(job_id), source_name, total,
            )
            return job_id

    async def update_import_job(self, job_id: str, status: str | None = None, progress: int | None = None, processed: int | None = None, error: str | None = None):
        async with self.pool.acquire() as conn:
            sets = []
            params = []
            idx = 1
            if status is not None:
                sets.append(f"status = ${idx}"); params.append(status); idx += 1
            if progress is not None:
                sets.append(f"progress = ${idx}"); params.append(progress); idx += 1
            if processed is not None:
                sets.append(f"processed = ${idx}"); params.append(processed); idx += 1
            if error is not None:
                sets.append(f"error = ${idx}"); params.append(error); idx += 1
            if sets:
                params.append(uuid.UUID(job_id))
                await conn.execute(f"UPDATE reid.import_jobs SET {', '.join(sets)} WHERE id = ${idx}", *params)

    async def get_import_job(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reid.import_jobs WHERE id = $1", uuid.UUID(job_id))
            if row:
                return {"id": str(row["id"]), "status": row["status"], "progress": row["progress"],
                        "processed": row["processed"], "total": row["total"], "source_name": row["source_name"],
                        "created_at": row["created_at"].isoformat(), "error": row["error"]}
            return None

    async def list_import_jobs(self, limit: int = 1, status: str | None = None) -> list[dict]:
        async with self.pool.acquire() as conn:
            where = []
            params = []
            idx = 1
            if status:
                where.append(f"status = ${idx}"); params.append(status); idx += 1
            w = " AND ".join(where) if where else "TRUE"
            params.append(limit)
            rows = await conn.fetch(f"SELECT * FROM reid.import_jobs WHERE {w} ORDER BY created_at DESC LIMIT ${idx}", *params)
            return [{"id": str(r["id"]), "status": r["status"], "progress": r["progress"],
                     "processed": r["processed"], "total": r["total"], "source_name": r["source_name"],
                     "created_at": r["created_at"].isoformat(), "error": r["error"]} for r in rows]

    async def cancel_import_job(self, job_id: str):
        await self.update_import_job(job_id, status="cancelled")

    # ---- Export jobs ----

    async def create_export_job(self, job_id: str, threshold: float, query_batch_id: str | None = None) -> str:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO reid.export_jobs (id, threshold, query_batch_id) VALUES ($1, $2, $3)",
                uuid.UUID(job_id), threshold, uuid.UUID(query_batch_id) if query_batch_id else None,
            )
            return job_id

    async def update_export_job(self, job_id: str, status: str | None = None, progress: int | None = None, row_count: int | None = None, artifacts: list | None = None, error: str | None = None):
        async with self.pool.acquire() as conn:
            sets = []
            params = []
            idx = 1
            if status is not None:
                sets.append(f"status = ${idx}"); params.append(status); idx += 1
                if status in ("done", "failed"):
                    sets.append("finished_at = NOW()")
            if progress is not None:
                sets.append(f"progress = ${idx}"); params.append(progress); idx += 1
            if row_count is not None:
                sets.append(f"row_count = ${idx}"); params.append(row_count); idx += 1
            if artifacts is not None:
                sets.append(f"artifacts = ${idx}::jsonb"); params.append(artifacts); idx += 1
            if error is not None:
                sets.append(f"error = ${idx}"); params.append(error); idx += 1
            if sets:
                params.append(uuid.UUID(job_id))
                await conn.execute(f"UPDATE reid.export_jobs SET {', '.join(sets)} WHERE id = ${idx}", *params)

    _EXPORT_JOB_SELECT = """
        SELECT e.*, qb.source_name AS query_source
        FROM reid.export_jobs e
        LEFT JOIN reid.import_jobs qb ON qb.id = e.query_batch_id
    """

    @staticmethod
    def _export_job_row(row) -> dict:
        return {
            "id": str(row["id"]), "status": row["status"], "progress": row["progress"],
            "query_batch_id": str(row["query_batch_id"]) if row["query_batch_id"] else "", "query_source": row["query_source"] or "CSV",
            "threshold": row["threshold"], "row_count": row["row_count"],
            "artifacts": row["artifacts"], "error": row["error"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        }

    async def get_export_job(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(f"{self._EXPORT_JOB_SELECT} WHERE e.id = $1", uuid.UUID(job_id))
            return self._export_job_row(row) if row else None

    async def list_export_jobs(self, limit: int = 1) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"{self._EXPORT_JOB_SELECT} ORDER BY e.created_at DESC LIMIT $1", limit)
            return [self._export_job_row(r) for r in rows]

    # ---- Metric jobs (оценка качества) ----

    async def create_metric_job(self, job_id: str) -> str:
        async with self.pool.acquire() as conn:
            await conn.execute("INSERT INTO reid.metric_jobs (id) VALUES ($1)", uuid.UUID(job_id))
            return job_id

    async def update_metric_job(self, job_id: str, status: str | None = None, progress: int | None = None, error: str | None = None, run_id: int | None = None):
        async with self.pool.acquire() as conn:
            sets = []
            params = []
            idx = 1
            if status is not None:
                sets.append(f"status = ${idx}"); params.append(status); idx += 1
            if progress is not None:
                sets.append(f"progress = ${idx}"); params.append(progress); idx += 1
            if error is not None:
                sets.append(f"error = ${idx}"); params.append(error); idx += 1
            if run_id is not None:
                sets.append(f"run_id = ${idx}"); params.append(run_id); idx += 1
            if sets:
                params.append(uuid.UUID(job_id))
                await conn.execute(f"UPDATE reid.metric_jobs SET {', '.join(sets)} WHERE id = ${idx}", *params)

    async def get_metric_job(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reid.metric_jobs WHERE id = $1", uuid.UUID(job_id))
            if row:
                return {"id": str(row["id"]), "status": row["status"], "progress": row["progress"], "error": row["error"], "run_id": row["run_id"]}
            return None

    async def save_metrics_run(self, data: dict) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO reid.metric_runs (model_version, threshold, map, rank1, rank5, f1, tnr, pr_auc, n_queries, details)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id""",
                data.get("model_version"), data.get("threshold"), data.get("map"),
                data.get("rank1"), data.get("rank5"), data.get("f1"), data.get("tnr"), data.get("pr_auc"),
                data.get("n_queries"), data.get("details", {}),
            )
            return row["id"]

    async def get_latest_metrics_run(self) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reid.metric_runs ORDER BY created_at DESC LIMIT 1")
            if row:
                return {
                    "id": row["id"], "created_at": row["created_at"].isoformat(),
                    "model_version": row["model_version"], "threshold": row["threshold"],
                    "map": row["map"], "rank1": row["rank1"], "rank5": row["rank5"],
                    "f1": row["f1"], "tnr": row["tnr"], "pr_auc": row["pr_auc"],
                    "n_queries": row["n_queries"], "details": row["details"],
                }
            return None

    async def get_metrics_run_by_id(self, run_id: int) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reid.metric_runs WHERE id = $1", run_id)
            if row:
                return {
                    "id": row["id"], "created_at": row["created_at"].isoformat(),
                    "model_version": row["model_version"], "threshold": row["threshold"],
                    "map": row["map"], "rank1": row["rank1"], "rank5": row["rank5"],
                    "f1": row["f1"], "tnr": row["tnr"], "pr_auc": row["pr_auc"],
                    "n_queries": row["n_queries"], "details": row["details"],
                }
            return None

    # ---- Batches ----

    async def get_objects_by_batch(self, batch_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, image_id, vehicle_id, bbox_x, bbox_y, bbox_w, bbox_h "
                "FROM reid.gallery_objects WHERE batch_id = $1 "
                "ORDER BY row_no NULLS LAST, created_at, id",
                uuid.UUID(batch_id),
            )
            return [
                {"id": str(r["id"]), "image_id": r["image_id"], "vehicle_id": r["vehicle_id"],
                 "bbox": {"x": r["bbox_x"], "y": r["bbox_y"], "w": r["bbox_w"], "h": r["bbox_h"]}}
                for r in rows
            ]

    async def get_all_gallery_objects(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, image_id, vehicle_id, bbox_x, bbox_y, bbox_w, bbox_h "
                "FROM reid.gallery_objects ORDER BY row_no NULLS LAST, created_at, id",
            )
            return [
                {"id": str(r["id"]), "image_id": r["image_id"], "vehicle_id": r["vehicle_id"],
                 "bbox": {"x": r["bbox_x"], "y": r["bbox_y"], "w": r["bbox_w"], "h": r["bbox_h"]}}
                for r in rows
            ]

    async def delete_gallery_object(self, object_id: str) -> dict:
        """Удаляет один объект галереи: запись из gallery_objects, если кадр больше не используется — из images и S3.
        Возвращает {image_id, deleted_image}."""
        async with self.pool.acquire() as conn:
            # Получаем image_id объекта
            row = await conn.fetchrow(
                "SELECT image_id FROM reid.gallery_objects WHERE id = $1",
                object_id,
            )
            if not row:
                return {"image_id": None, "deleted_image": False}
            image_id = row["image_id"]

            # Удаляем запись из gallery_objects
            await conn.execute(
                "DELETE FROM reid.gallery_objects WHERE id = $1",
                object_id,
            )

            # Проверяем, ссылается ли ещё кто-то на этот image_id
            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM reid.gallery_objects WHERE image_id = $1",
                image_id,
            )

            deleted_image = False
            if remaining == 0:
                # Никто больше не использует этот кадр — удаляем из images и storage
                await conn.execute(
                    "DELETE FROM reid.images WHERE image_id = $1",
                    image_id,
                )
                deleted_image = True

            return {"image_id": image_id, "deleted_image": deleted_image}

    async def delete_objects_by_image_ids(self, image_ids: list[str]) -> int:
        """Удаляет все gallery_objects с указанными image_id. Возвращает количество удалённых."""
        if not image_ids:
            return 0
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM reid.gallery_objects WHERE image_id = ANY($1::text[])",
                image_ids,
            )
            return int(result.split()[1]) if result else 0

    async def delete_images_by_ids(self, image_ids: list[str]) -> int:
        """Удаляет записи из таблицы images по image_id. Возвращает количество удалённых."""
        if not image_ids:
            return 0
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM reid.images WHERE image_id = ANY($1::text[])",
                image_ids,
            )
            return int(result.split()[1]) if result else 0

    async def clear_metrics(self) -> dict:
        """Удаляет все записи metric_jobs и metric_runs. Возвращает количество удалённых."""
        async with self.pool.acquire() as conn:
            jobs = await conn.fetchval("SELECT COUNT(*) FROM reid.metric_jobs")
            runs = await conn.fetchval("SELECT COUNT(*) FROM reid.metric_runs")
            await conn.execute("DELETE FROM reid.metric_jobs")
            await conn.execute("DELETE FROM reid.metric_runs")
            return {"deleted_jobs": jobs, "deleted_runs": runs}

    async def get_processing_status_for_ids(self, image_ids: list[str]) -> dict:
        """Возвращает агрегированный статус обработки для списка image_id.
        Считает только те image_id, что есть в таблице images.
        Возвращает {total, tracked, uploaded, processing, ready, failed}.
        total — количество запрошенных image_id.
        tracked — сколько из них есть в таблице images (остальные — только в S3, без векторизации).
        """
        if not image_ids:
            return {"total": 0, "tracked": 0, "uploaded": 0, "processing": 0, "ready": 0, "failed": 0}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT processing_status, COUNT(*)::int AS cnt "
                "FROM reid.images "
                "WHERE image_id = ANY($1::text[]) "
                "GROUP BY processing_status",
                image_ids,
            )
            counts = {r["processing_status"]: r["cnt"] for r in rows}
            tracked = sum(counts.values())
            return {
                "total": len(image_ids),
                "tracked": tracked,
                "uploaded": counts.get("uploaded", 0),
                "processing": counts.get("processing", 0),
                "ready": counts.get("ready", 0),
                "failed": counts.get("failed", 0),
            }


db = Database()
