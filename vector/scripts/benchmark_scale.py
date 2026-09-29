#!/usr/bin/env python3
"""Бенчмарк времени поиска в Qdrant от объёма данных (ТЗ п.10, «Масштабируемость»).

Создаёт временные коллекции разных размеров (от 10K до ~1M векторов),
замеряет латентность поиска (p50, p95, p99, mean) и строит график.

Данные синтетические: кластеризованные векторы (по 5 снимков на «автомобиль»)
с шумом — имитация реальной галереи ReID. Это замер скорости индекса,
а не качества модели.

Использование:
    python scripts/benchmark_scale.py                         # полный прогон
    python scripts/benchmark_scale.py --max 200000            # быстрая проверка
    python scripts/benchmark_scale.py --sizes 10000 50000 100000 200000
    python scripts/benchmark_scale.py --no-show               # без интерактивного показа
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from qdrant_client import QdrantClient, models

# ── Константы ────────────────────────────────────────────────────────────────

DEFAULT_SIZES = [10_000, 25_000, 50_000, 100_000, 200_000, 350_000, 500_000, 750_000, 1_000_000]
# Для быстрых запусков — умеренный набор
FAST_SIZES = [10_000, 50_000, 100_000, 200_000, 350_000]
IMAGES_PER_VEHICLE = 5      # как в реальной галерее: несколько снимков одного ТС
NOISE = 0.35                # шум относительно длины центра (см. benchmark_ann.py)
WARMUP_QUERIES = 20         # запросов на прогрев перед замерами
MEASURED_QUERIES = 200      # запросов для сбора статистики
CHUNK_VEHICLES = 500        # автомобилей в одном загружаемом блоке


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def make_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Создать n кластеризованных векторов: по IMAGES_PER_VEHICLE на «автомобиль»."""
    vehicles = -(-n // IMAGES_PER_VEHICLE)
    centers = normalize(rng.standard_normal((vehicles, dim), dtype=np.float32))
    copies = np.repeat(centers, IMAGES_PER_VEHICLE, axis=0)[:n]
    noise = rng.standard_normal(copies.shape, dtype=np.float32) * (NOISE / np.sqrt(dim))
    return normalize(copies + noise).astype(np.float32)


def make_queries(rng: np.random.Generator, base_vectors: np.ndarray, n: int, dim: int) -> np.ndarray:
    """Создать запросы — слегка изменённые копии случайных векторов из галереи."""
    chosen = base_vectors[rng.choice(len(base_vectors), size=n, replace=False)]
    return normalize(chosen + rng.standard_normal((n, dim), dtype=np.float32) * (0.2 / np.sqrt(dim)))


def create_temporary_collection(client: QdrantClient, name: str, dim: int,
                                quantization: str, hnsw_m: int, hnsw_ef: int):
    """Создать коллекцию с отложенной индексацией."""
    scalar = quantization == "scalar"
    client.create_collection(
        name,
        vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE, on_disk=scalar,
        ),
        hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef),
        quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True)) if scalar else None,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )


def load_vectors(client: QdrantClient, name: str, vectors: np.ndarray):
    """Загрузить векторы в коллекцию пачками (gRPC-streaming, parallel)."""
    n = len(vectors)
    # Используем parallel=10 для параллельной загрузки через gRPC
    client.upload_collection(
        name,
        vectors=vectors,
        ids=range(n),
        batch_size=256,
        parallel=10,
        max_retries=5,
    )


def build_index(client: QdrantClient, name: str, n_vectors: int, timeout: int = 600) -> float:
    """Запустить построение HNSW-индекса и ждать завершения. Возвращает затраченное время.
    Если за timeout секунд индекс не построился — возвращает -1 (коллекция остаётся для диагностики).
    """
    # Порог должен быть не больше числа векторов, иначе индекс не запустится
    threshold = min(20000, max(1000, n_vectors // 2))
    client.update_collection(name, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=threshold))
    start = time.perf_counter()
    last_indexed = 0
    stall_count = 0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            print(f"\n  ⚠ Таймаут построения индекса ({timeout} с), пропускаю...")
            return -1.0
        time.sleep(2)
        info = client.get_collection(name)
        indexed = info.indexed_vectors_count or 0
        if indexed == n_vectors:
            break
        # Прогресс каждые 10 секунд
        if int(elapsed) % 10 == 0 and indexed != last_indexed:
            print(f"\r    индекс: {indexed}/{n_vectors} ({elapsed:.0f}с)", end="", flush=True)
        if indexed > 0 and indexed == last_indexed:
            stall_count += 1
            if stall_count >= 15:  # 30 секунд без прогресса — форсим
                try:
                    client.update_collection(
                        name,
                        optimizer_config=models.OptimizersConfigDiff(
                            indexing_threshold=threshold,
                            default_segment_number=2,
                        ),
                    )
                except Exception:
                    pass
                stall_count = 0
        else:
            stall_count = 0
        last_indexed = indexed
    print()
    return time.perf_counter() - start


def measure_latency(client: QdrantClient, name: str, queries: np.ndarray,
                    limit: int, hnsw_ef: int, scalar: bool) -> dict:
    """Замерить латентность поиска. Возвращает {p50, p95, p99, mean, min, max, qps} в мс."""
    params = models.SearchParams(
        hnsw_ef=hnsw_ef,
        quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0) if scalar else None,
    )

    # Прогрев
    for v in queries[:WARMUP_QUERIES]:
        client.query_points(name, query=v.tolist(), limit=limit, search_params=params)

    # Замеры
    times_ms = []
    for v in queries[WARMUP_QUERIES:]:
        start = time.perf_counter()
        client.query_points(name, query=v.tolist(), limit=limit, search_params=params)
        times_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(times_ms)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "qps": round(len(times_ms) / (np.sum(arr) / 1000), 1),
        "std": round(float(np.std(arr)), 2),
    }


def print_latency(latency: dict):
    print(f"  ┌──────────┬──────────┐")
    print(f"  │ Метрика  │  Знач.   │")
    print(f"  ├──────────┼──────────┤")
    print(f"  │ p50, мс  │ {latency['p50']:>8.2f} │")
    print(f"  │ p95, мс  │ {latency['p95']:>8.2f} │")
    print(f"  │ p99, мс  │ {latency['p99']:>8.2f} │")
    print(f"  │ mean, мс │ {latency['mean']:>8.2f} │")
    print(f"  │ qps      │ {latency['qps']:>8.1f} │")
    print(f"  └──────────┴──────────┘")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help=f"размеры галереи для теста (по умолчанию: {DEFAULT_SIZES})")
    parser.add_argument("--max", dest="max_size", type=int, default=None,
                        help="максимальный размер (альтернативный способ задать диапазон)")
    parser.add_argument("--dim", type=int, default=2048,
                        help="размерность эмбеддинга (по умолчанию: 2048)")
    parser.add_argument("--limit", type=int, default=IMAGES_PER_VEHICLE,
                        help="число соседей (по умолчанию: 5, как снимков у одного ТС)")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef", type=int, default=128,
                        help="ef при поиске; больше = точнее, но медленнее")
    parser.add_argument("--ef-construct", type=int, default=128)
    parser.add_argument("--quantization", choices=("none", "scalar"), default="none",
                        help="режим квантизации (по умолчанию: none)")
    parser.add_argument("--queries", type=int, default=MEASURED_QUERIES,
                        help=f"число измерительных запросов на размер (по умолчанию: {MEASURED_QUERIES})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", "reidqdrantdevonly"))
    parser.add_argument("--output", "-o", default=None,
                        help="сохранить график в файл (.png или .pdf)")
    parser.add_argument("--json", dest="json_output", default=None,
                        help="сохранить сырые результаты в JSON")
    parser.add_argument("--no-show", action="store_true",
                        help="не показывать график интерактивно")
    parser.add_argument("--no-clean", action="store_true",
                        help="не удалять временные коллекции после теста")
    parser.add_argument("--prefix", default="bench_scale",
                        help="префикс имени временной коллекции")
    parser.add_argument("--resume", action="store_true",
                        help="пропускать уже существующие коллекции")
    parser.add_argument("--build-timeout", type=int, default=600,
                        help="таймаут построения индекса в секундах (по умолчанию: 600)")
    args = parser.parse_args()

    # Определяем размеры
    if args.sizes:
        sizes = sorted(args.sizes)
    elif args.max_size:
        sizes = sorted(s for s in DEFAULT_SIZES if s <= args.max_size)
    else:
        sizes = DEFAULT_SIZES

    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=600)
    scalar = args.quantization == "scalar"
    rng = np.random.default_rng(args.seed)

    results = []
    prev_name = None

    try:
        for i, n in enumerate(sizes):
            name = f"{args.prefix}_{n}"
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(sizes)}] Размер галереи: {n:,} векторов")
            print(f"{'='*60}")

            # Пропускаем если уже есть и включён --resume
            if args.resume and client.collection_exists(name):
                info = client.get_collection(name)
                if info.points_count == n and info.indexed_vectors_count == n:
                    print(f"  ⊳ Уже есть, пропускаю (points={info.points_count}, indexed={info.indexed_vectors_count})")
                    # Всё равно делаем замеры, если есть векторы для запросов
                    rng_for_queries = np.random.default_rng(args.seed + n)  # детерминированные, но разные запросы
                    # Создадим запросы на лету (они не привязаны к конкретным данным)
                    queries = normalize(rng_for_queries.standard_normal((args.queries, args.dim), dtype=np.float32))
                    latency = measure_latency(client, name, queries, args.limit,
                                              args.hnsw_ef, scalar)
                    row = {
                        "size": n,
                        "dim": args.dim,
                        "quantization": args.quantization,
                        "hnsw_m": args.hnsw_m,
                        "hnsw_ef": args.hnsw_ef,
                        "load_seconds": 0,
                        "build_seconds": 0,
                        **latency,
                    }
                    results.append(row)
                    print_latency(latency)
                    continue
                else:
                    # Существует, но неполная — удаляем
                    print(f"  ⊳ Существует, но неполная ({info.points_count}/{info.indexed_vectors_count}), пересоздаю")
                    client.delete_collection(name)

            # Создаём коллекцию
            if client.collection_exists(name):
                client.delete_collection(name)
            create_temporary_collection(client, name, args.dim, args.quantization,
                                        args.hnsw_m, args.ef_construct)

            # Генерируем и загружаем данные
            print(f"  Генерация {n:,} векторов (dim={args.dim})...")
            t0 = time.perf_counter()
            vectors = make_vectors(rng, n, args.dim)

            print(f"  Загрузка в Qdrant...", end=" ", flush=True)
            t0 = time.perf_counter()
            load_vectors(client, name, vectors)
            load_time = time.perf_counter() - t0
            print(f"({load_time:.1f} с)")

            # Построение индекса
            print(f"  Построение HNSW-индекса ({n} векторов)...", flush=True)
            build_time = build_index(client, name, n, args.build_timeout)
            if build_time < 0:
                print(f"  ⚠ Индекс не построен, пропускаю замеры для {n}")
                continue
            print(f"  Индекс построен за {build_time:.1f} с")

            # Запросы
            queries = make_queries(rng, vectors, args.queries, args.dim)
            latency = measure_latency(client, name, queries, args.limit,
                                      args.hnsw_ef, scalar)

            row = {
                "size": n,
                "dim": args.dim,
                "quantization": args.quantization,
                "hnsw_m": args.hnsw_m,
                "hnsw_ef": args.hnsw_ef,
                "load_seconds": round(load_time, 2),
                "build_seconds": round(build_time, 2),
                **latency,
            }
            results.append(row)

            print_latency(latency)

            # Удаляем предыдущую коллекцию (кроме последней, если --no-clean)
            if prev_name and not args.no_clean:
                client.delete_collection(prev_name)
            prev_name = name if args.no_clean else None

    finally:
        # Очистка последней коллекции
        if prev_name and not args.no_clean:
            try:
                client.delete_collection(prev_name)
            except Exception:
                pass

    # ── Сохранение JSON ──────────────────────────────────────────────────────
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты сохранены: {args.json_output}")

    # ── Построение графика ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg" if args.no_show else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠ matplotlib не установлен, график не построен")
        print("  Установите: pip install matplotlib")
        if args.output:
            print(f"  Файл {args.output} не создан")
        sys.exit(0)

    sizes_arr = np.array([r["size"] for r in results])
    p50 = np.array([r["p50"] for r in results])
    p95 = np.array([r["p95"] for r in results])
    p99 = np.array([r["p99"] for r in results])
    mean = np.array([r["mean"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ─── График 1: латентность от объёма ────────────────────────────────────
    ax1.plot(sizes_arr / 1000, p50, "o-", label="p50", color="#2196F3", linewidth=2, markersize=6)
    ax1.plot(sizes_arr / 1000, p95, "s-", label="p95", color="#FF9800", linewidth=2, markersize=6)
    ax1.plot(sizes_arr / 1000, p99, "^-", label="p99", color="#F44336", linewidth=2, markersize=6)
    ax1.plot(sizes_arr / 1000, mean, "d--", label="mean", color="#4CAF50", linewidth=1.5, markersize=5, alpha=0.7)

    ax1.set_xlabel("Количество векторов в галерее (тыс.)", fontsize=12)
    ax1.set_ylabel("Время поиска (мс)", fontsize=12)
    ax1.set_title("Латентность поиска HNSW\nот объёма данных в Qdrant", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(left=0)

    # Аннотации точек p50
    for s, v in zip(sizes_arr, p50):
        ax1.annotate(f"{v:.0f} мс", (s / 1000, v), textcoords="offset points",
                     xytext=(0, -14), fontsize=8, ha="center", color="#1565C0")

    # ─── График 2: пропускная способность ───────────────────────────────────
    qps_arr = np.array([r["qps"] for r in results])
    bars = ax2.bar([f"{s//1000}K" if s < 1_000_000 else "1M" for s in sizes_arr],
                   qps_arr, color="#42A5F5", alpha=0.85, edgecolor="#1E88E5", linewidth=1.2)
    for bar, v in zip(bars, qps_arr):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(qps_arr) * 0.02,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Количество векторов в галерее", fontsize=12)
    ax2.set_ylabel("Запросов/сек (один клиент)", fontsize=12)
    ax2.set_title("Пропускная способность\nпоиска HNSW в Qdrant", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Подписи параметров
    params_text = (f"dim={args.dim}, quantization={args.quantization}, "
                   f"ef={args.hnsw_ef}, m={args.hnsw_m}, "
                   f"limit={args.limit}, queries={args.queries}")
    fig.suptitle(f"Qdrant HNSW — масштабирование поиска\n{params_text}",
                 fontsize=11, color="gray", y=1.02)

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"График сохранён: {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    # ── Итоговая таблица ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Размер':>10} | {'p50,мс':>8} {'p95,мс':>8} {'p99,мс':>8} {'mean,мс':>8} {'qps':>8} | {'загрузка':>9} {'индекс':>9}")
    print(f"{'─'*70}")
    for r in results:
        size_label = f"{r['size']//1000}K" if r['size'] < 1_000_000 else "1M"
        print(f"{size_label:>10} | {r['p50']:>8.1f} {r['p95']:>8.1f} {r['p99']:>8.1f} "
              f"{r['mean']:>8.1f} {r['qps']:>8.1f} | {r['load_seconds']:>8.1f}s {r['build_seconds']:>8.1f}s")
    print(f"{'─'*70}")
    print(f"dim={args.dim}, quantization={args.quantization}, ef={args.hnsw_ef}, m={args.hnsw_m}")


if __name__ == "__main__":
    main()
