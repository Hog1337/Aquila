"""Замер производительности инференса (ТЗ п.7).

Измеряет:
- Время формирования признака на одно ТС (batch=1) — latency (полный пайплайн с YOLO)
- Пропускную способность (FPS) при пакетной обработке — throughput (V2-only, batch)

Используется:
    docker compose run --rm backend python -m app.bench_inference

Зависимости: контейнер inference должен быть запущен.
"""
import json
import time
from pathlib import Path

import numpy as np
import httpx
from PIL import Image
import io
import os


def load_test_images(image_dir: str, n: int = 100) -> list[tuple[str, bytes]]:
    """Загружает N реальных изображений для замера."""
    img_dir = Path(image_dir)
    jpgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not jpgs:
        raise SystemExit(f"Нет изображений в {image_dir}")
    samples = []
    for p in jpgs[:n]:
        samples.append((p.name, p.read_bytes()))
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark inference performance (ТЗ п.7)")
    parser.add_argument("--inference-url", default="http://inference:8001",
                        help="Inference server URL")
    parser.add_argument("--images", default="/input/images",
                        help="Directory with test images")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of images to test")
    parser.add_argument("--json", default=None,
                        help="Save report to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("БЕНЧМАРК ИНФЕРЕНСА (ТЗ п.7)")
    print("=" * 60)

    # Load images
    print(f"\n[1] Загрузка {args.n} изображений из {args.images}...")
    samples = load_test_images(args.images, args.n)
    avg_size = sum(len(d) for _, d in samples) / len(samples)
    print(f"    Загружено {len(samples)} изображений, "
          f"средний размер {avg_size / 1024:.0f} КБ")

    client = httpx.Client(timeout=120, base_url=args.inference_url)

    # Health check
    r = client.get("/health")
    health = r.json()
    if health["status"] != "ok":
        print(f"    ВНИМАНИЕ: inference сервер в статусе '{health['status']}'")
    print(f"    Модель: {health.get('model_version', '?')}, "
          f"device: {health.get('device', '?')}, "
          f"YOLO: {'✅' if health.get('yolo_loaded') else '❌'}")

    # ========== 2. LATENCY (batch=1, full pipeline with YOLO) ==========
    print(f"\n[2] Время формирования признака (batch=1, полный пайплайн)")
    print(f"    Измерение на {min(args.n, 50)} изображениях...")

    # Warmup: 5 прогонов
    for i in range(5):
        name, data = samples[i]
        client.post("/internal/embed",
                     files={"image": ("img.jpg", data, "image/jpeg")},
                     data={"x": 100, "y": 100, "w": 300, "h": 300})

    # Замер: 50 прогонов (или сколько есть)
    n_latency = min(args.n, 50)
    latencies = []
    for i in range(n_latency):
        name, data = samples[i]
        t0 = time.perf_counter()
        r = client.post("/internal/embed",
                         files={"image": ("img.jpg", data, "image/jpeg")},
                         data={"x": 300, "y": 200, "w": 500, "h": 400})
        r.raise_for_status()
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed)

    latencies.sort()
    p50 = np.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    mean = np.mean(latencies)

    print(f"    Среднее: {mean:.1f} ms")
    print(f"    Медиана (P50): {p50:.1f} ms")
    print(f"    P95: {p95:.1f} ms")
    latency_score = 1.0 if mean <= 40 else max(0, 1.0 - (mean - 40) / 40) if mean <= 80 else 0.0
    print(f"    Балл по ТЗ: {'✅ ПОЛНЫЙ (≤40ms)' if mean <= 40 else '⚠️ линейный (40-80ms)' if mean <= 80 else '❌ 0'}")

    # ========== 3. THROUGHPUT (batch, полный пайплайн с YOLO) ==========
    print(f"\n[3] Пропускная способность (полный пайплайн: YOLO + маска + V2)")
    print(f"    Измерение на реальных изображениях через embed-by-key-batch...")

    import json
    batch_sizes = [1, 4, 8, 16, 32]
    throughput_results = {}

    for bs in batch_sizes:
        ids = [f.split('.')[0] for f, _ in samples[:bs]]
        specs = [{'image_id': iid, 'x': 300, 'y': 200, 'w': 500, 'h': 400} for iid in ids]

        # Warmup
        client.post('/internal/embed-by-key-batch', data={'keys': json.dumps(specs[:1])})

        # Замер: ≥5 runs или ≥10s
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = client.post('/internal/embed-by-key-batch', data={'keys': json.dumps(specs)})
            r.raise_for_status()
            times.append(time.perf_counter() - t0)
            if len(times) >= 5 and sum(times) >= 5.0:
                break

        avg = sum(times) / len(times)
        fps = bs / avg
        score = 1.0 if fps >= 100 else max(0, (fps - 50) / 50) if fps >= 50 else 0.0
        throughput_results[bs] = {
            "batch_size": bs,
            "runs": len(times),
            "total_time_s": round(sum(times), 3),
            "fps": round(fps, 1),
            "ms_per_image": round(avg / bs * 1000, 1),
            "score": round(score, 2),
        }
        print(f"    batch={bs:2d}: {fps:6.1f} FPS, "
              f"{avg/bs*1000:.1f} ms/img, "
              f"балл={score:.2f} ({len(times)} runs, {sum(times):.1f}s total)")

    best_fps = max(v["fps"] for v in throughput_results.values())
    best_bs = max(throughput_results, key=lambda k: throughput_results[k]["fps"])
    throughput_score = throughput_results[best_bs]["score"]

    print(f"\n    Лучший FPS: {best_fps:.1f} (batch={best_bs})")
    print(f"    Балл по ТЗ: {'✅ ПОЛНЫЙ (≥100 FPS)' if best_fps >= 100 else '⚠️ линейный (50-100 FPS)' if best_fps >= 50 else '❌ 0'}")

    # ========== 4. MODEL WEIGHTS ==========
    print(f"\n[4] Веса модели")
    total_bytes = 0
    total_mb = 0
    limit_mb = 2048  # 2 GB
    weights_dir = Path("/app/weights")
    if weights_dir.exists():
        total_bytes = sum(f.stat().st_size for f in weights_dir.rglob("*") if f.is_file())
        total_mb = total_bytes / (1024 * 1024)
        print(f"    Всего: {total_mb:.0f} MB / {limit_mb} MB = {total_mb/limit_mb*100:.0f}%")
        print(f"    Лимит 2 GB: {'✅' if total_bytes <= limit_mb * 1024 * 1024 else '❌ ПРЕВЫШЕН'}")
    else:
        print(f"    Путь {weights_dir} не найден (веса в inference-контейнере)")

    # ========== 5. SUMMARY ==========
    print(f"\n{'=' * 60}")
    print("СВОДКА ПО ТЗ п.7")
    print(f"{'=' * 60}")
    print(f"  Latency (batch=1, full pipeline): {mean:.1f} ms  → балл {latency_score:.2f}")
    print(f"  Throughput (V2-only batch):       {best_fps:.1f} FPS → балл {throughput_score:.2f}")
    if total_bytes > 0:
        print(f"  Веса модели:                       {total_mb:.0f} MB / 2048 MB {'✅' if total_bytes <= limit_mb * 1024 * 1024 else '❌'}")
    performance_score = latency_score * 0.5 + throughput_score * 0.5
    print(f"\n  ИТОГО (производительность 20%): {performance_score:.2f} / 1.0")
    print()

    # Save report
    if args.json:
        report = {
            "model_version": health.get("model_version"),
            "device": health.get("device"),
            "latency_ms": {
                "mean": round(mean, 1),
                "p50": round(p50, 1),
                "p95": round(p95, 1),
                "score": round(latency_score, 2),
                "full_pipeline": "YOLO seg + V2 forward",
            },
            "throughput": throughput_results,
            "best_fps": {
                "batch_size": best_bs,
                "fps": round(best_fps, 1),
                "score": round(throughput_score, 2),
                "mode": "V2-only (embed-batch, synthetic crops)",
            },
            "weights_mb": round(total_mb, 0) if total_bytes > 0 else None,
            "weights_limit_mb": limit_mb,
            "performance_score": round(performance_score, 2),
        }
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Отчёт сохранён: {args.json}")


if __name__ == "__main__":
    main()
