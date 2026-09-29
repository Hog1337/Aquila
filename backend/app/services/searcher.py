from qdrant_client import QdrantClient
from qdrant_client.http import models

from .. import config


class Searcher:
    def __init__(self):
        """Создаёт клиент Qdrant для поиска и хранения эмбеддингов. Поиск идёт по всей коллекции — без фильтра по ролям."""
        self.client = QdrantClient(
            url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}",
            api_key=config.QDRANT_API_KEY or None,
        )
        self.collection = config.QDRANT_COLLECTION

    def search(self, embedding: list[float], top_n: int = 10) -> list[dict]:
        """Ищет ближайшие объекты во всей галерее по косинусному сходству."""
        hits = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_n,
            with_payload=True,
        )
        return [
            {
                "id": str(pt.id),
                "score": pt.score,
                "image_id": pt.payload.get("image_id"),
                "vehicle_id": pt.payload.get("vehicle_id"),
                "bbox": pt.payload.get("bbox"),
            }
            for pt in hits.points
        ]

    def search_rerank(self, embedding: list[float], top_n: int = 10) -> list[dict]:
        """Ищет кандидатов с Query Expansion: усредняет запрос с топ-5 результатами первого прохода и ищет повторно."""
        first_k = max(top_n * 5, 50)

        hits = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=first_k,
            with_payload=True,
            with_vectors=True,
        )

        if not hits.points:
            return []

        qe_top = min(5, len(hits.points))
        import numpy as np

        avg = np.array(embedding, dtype=np.float32)
        for pt in hits.points[:qe_top]:
            avg += np.array(pt.vector, dtype=np.float32)
        avg /= (1 + qe_top)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm

        rerun = self.client.query_points(
            collection_name=self.collection,
            query=avg.tolist(),
            limit=top_n,
            with_payload=True,
        )
        return [
            {
                "id": str(pt.id),
                "score": pt.score,
                "image_id": pt.payload.get("image_id"),
                "vehicle_id": pt.payload.get("vehicle_id"),
                "bbox": pt.payload.get("bbox"),
            }
            for pt in rerun.points
        ]

    def retrieve_vectors(self, ids: list[str]) -> dict[str, list[float]]:
        """Возвращает векторы точек по их id, батчами."""
        out: dict[str, list[float]] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            points = self.client.retrieve(collection_name=self.collection, ids=chunk, with_vectors=True, with_payload=False)
            for p in points:
                out[str(p.id)] = p.vector
        return out

    def upsert(self, point_id: str, embedding: list[float], payload: dict):
        """Добавляет или обновляет точку в индексе."""
        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=point_id, vector=embedding, payload=payload)],
        )

    def upsert_batch(self, points: list[tuple[str, list[float], dict]]) -> int:
        """Добавляет пачку точек одной командой. Возвращает количество точек.
        Разбивает на под-батчи по 500 точек, чтобы не превысить лимит Qdrant (32 MB)."""
        BATCH_SIZE = 500
        total = 0
        for start in range(0, len(points), BATCH_SIZE):
            chunk = points[start:start + BATCH_SIZE]
            structs = [models.PointStruct(id=pid, vector=emb, payload=pload) for pid, emb, pload in chunk]
            self.client.upsert(
                collection_name=self.collection,
                points=structs,
                wait=True,
            )
            total += len(chunk)
        return total

    def scroll_all(self) -> list[dict]:
        """Возвращает все точки индекса постранично."""
        all_points = []
        next_offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            points, next_offset = result
            for p in points:
                all_points.append({
                    "id": str(p.id),
                    "vector": p.vector,
                    "payload": p.payload,
                })
            if next_offset is None:
                break
        return all_points

    def count(self) -> int:
        """Возвращает размер всей галереи."""
        return self.client.count(collection_name=self.collection).count

    def search_by_image_ids(self, embedding: list[float], image_ids: list[str], top_n: int = 10) -> list[dict]:
        """Ищет ближайшие объекты среди заданного набора image_id (для метрик/экспорта)."""
        hits = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_n,
            with_payload=True,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="image_id", match=models.MatchAny(any=image_ids))]
            ),
        )
        return [
            {
                "id": str(pt.id),
                "score": pt.score,
                "image_id": pt.payload.get("image_id"),
                "vehicle_id": pt.payload.get("vehicle_id"),
                "bbox": pt.payload.get("bbox"),
            }
            for pt in hits.points
        ]

    def delete_point(self, point_id: str):
        """Удаляет точку из индекса."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )

    def delete_by_image_ids(self, image_ids: list[str]) -> int:
        """Удаляет все точки с указанными image_id одной командой (filter-based).
        Возвращает количество image_ids (Qdrant не возвращает точное число удалённых)."""
        if not image_ids:
            return 0
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="image_id", match=models.MatchAny(any=image_ids))]
                )
            ),
            wait=True,
        )
        return len(image_ids)


searcher = Searcher()
