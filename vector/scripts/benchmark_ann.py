#!/usr/bin/env python3
"""Бенчмарк приближённого поиска (HNSW) на синтетической галерее до 10^6 объектов (ТЗ п.10, «Масштабируемость»).

Создаёт временную коллекцию, заполняет её кластеризованными векторами (по 5 снимков на «автомобиль»,
как в реальной галерее), затем сравнивает HNSW с точным перебором (exact=true):
  recall@K = доля точного топ-K, найденная HNSW (K = --limit); задержка p50/p95, мс; пропускная способность, запросов/с.
Данные синтетические: это замер скорости и полноты индекса, а не качества модели ReID. В отчётах указывайте это явно.

    python benchmark_ann.py --n 1000000 --quantization scalar
    python benchmark_ann.py --n 100000            # быстрая проверка
"""
import argparse
import json
import os
import time

import numpy as np
from qdrant_client import QdrantClient, models

IMAGES_PER_VEHICLE = 5
NOISE = 0.35  # доля шума относительно длины центра: определяет, насколько снимки одного ТС похожи друг на друга


def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def make_chunk(rng, vehicles, dim):
    """vehicles центров-«автомобилей» и по IMAGES_PER_VEHICLE зашумлённых копий каждого."""
    centers = normalize(rng.standard_normal((vehicles, dim), dtype=np.float32))
    copies = np.repeat(centers, IMAGES_PER_VEHICLE, axis=0)
    noise = rng.standard_normal(copies.shape, dtype=np.float32) * (NOISE / np.sqrt(dim))
    return normalize(copies + noise).astype(np.float32)


def search(client, name, vector, limit, params):
    start = time.perf_counter()
    hits = client.query_points(name, query=vector, limit=limit, search_params=params).points
    return {h.id for h in hits}, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1_000_000, help="размер галереи")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--limit", type=int, default=IMAGES_PER_VEHICLE,
                        help="сколько соседей сравнивать; по умолчанию столько, сколько снимков у одного ТС: дальше идут случайные почти равные значения, их порядок не определён")
    parser.add_argument("--hnsw-ef", type=int, default=128, help="ef при поиске: больше = точнее и медленнее")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construct", type=int, default=128)
    parser.add_argument("--quantization", choices=("none", "scalar"), default="none")
    parser.add_argument("--chunk-vehicles", type=int, default=400, help="автомобилей в одном загружаемом блоке")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", action="store_true", help="не удалять временную коллекцию после теста")
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", "reidqdrantdevonly"))
    args = parser.parse_args()

    name = f"bench_ann_{args.n}"
    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=600)
    if client.collection_exists(name):
        client.delete_collection(name)

    scalar = args.quantization == "scalar"
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=args.dim, distance=models.Distance.COSINE, on_disk=scalar),
        hnsw_config=models.HnswConfigDiff(m=args.m, ef_construct=args.ef_construct),
        quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True)) if scalar else None,
        # Индекс строится после загрузки, а не по ходу неё: так быстрее на миллионе точек
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    rng = np.random.default_rng(args.seed)
    queries = None
    loaded = 0
    started = time.perf_counter()
    while loaded < args.n:
        vehicles = min(args.chunk_vehicles, -(-(args.n - loaded) // IMAGES_PER_VEHICLE))
        chunk = make_chunk(rng, vehicles, args.dim)[: args.n - loaded]
        client.upload_collection(name, vectors=chunk, ids=range(loaded, loaded + len(chunk)), batch_size=len(chunk))
        if queries is None:
            # Запросы: слегка изменённые копии объектов из первого блока
            base = chunk[rng.choice(len(chunk), size=min(args.queries, len(chunk)), replace=False)]
            queries = normalize(base + rng.standard_normal(base.shape, dtype=np.float32) * (0.2 / np.sqrt(args.dim)))
        loaded += len(chunk)
        if loaded % 100_000 < len(chunk):
            print(f"загружено {loaded}/{args.n}", flush=True)
    load_seconds = time.perf_counter() - started

    # Включаем индексацию и ждём построения HNSW
    client.update_collection(name, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000))
    build_started = time.perf_counter()
    while True:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) > 0:
            break
        time.sleep(2)
    build_seconds = time.perf_counter() - build_started

    exact_params = models.SearchParams(exact=True)
    ann_params = models.SearchParams(
        hnsw_ef=args.hnsw_ef,
        quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0) if scalar else None,
    )

    truth, exact_ms = [], []
    for vector in queries:
        ids, ms = search(client, name, vector.tolist(), args.limit, exact_params)
        truth.append(ids)
        exact_ms.append(ms)

    ann_ms, recalls = [], []
    wall = time.perf_counter()
    for vector, true_ids in zip(queries, truth):
        ids, ms = search(client, name, vector.tolist(), args.limit, ann_params)
        ann_ms.append(ms)
        recalls.append(len(ids & true_ids) / len(true_ids))
    wall = time.perf_counter() - wall

    result = {
        "gallery_size": args.n,
        "dim": args.dim,
        "quantization": args.quantization,
        "hnsw": {"m": args.m, "ef_construct": args.ef_construct, "ef_search": args.hnsw_ef},
        "queries": len(queries),
        "indexed_vectors": info.indexed_vectors_count,
        f"recall_at_{args.limit}": round(float(np.mean(recalls)), 4),
        "ann_latency_ms": {"p50": round(float(np.percentile(ann_ms, 50)), 2),
                           "p95": round(float(np.percentile(ann_ms, 95)), 2)},
        "exact_latency_ms": {"p50": round(float(np.percentile(exact_ms, 50)), 2),
                             "p95": round(float(np.percentile(exact_ms, 95)), 2)},
        "ann_qps_single_client": round(len(queries) / wall, 1),
        "load_seconds": round(load_seconds, 1),
        "index_build_seconds": round(build_seconds, 1),
        "note": "синтетические данные: замер скорости и полноты индекса, не качества ReID",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.keep:
        client.delete_collection(name)


if __name__ == "__main__":
    main()
