#!/usr/bin/env python3
"""Строит графики масштабирования HNSW-поиска (Qdrant) по объёму галереи: docs/scaling.md.

Два PNG в --out-dir (по умолчанию ../../docs/img относительно скрипта):
  scaling_latency.png  задержка p50 поиска в зависимости от N (log-log), три источника роста:
                        - наивная оценка ~log(N) (для сравнения, не подтверждена измерениями);
                        - степенной рост cost ~ N^c из открытого исследования HNSW на восьми
                          датасетах (arXiv:2412.01940, "Down with the Hierarchy: The 'H' in
                          HNSW Stands for Hubs"), диапазон c = 0.185 (SIFT, recall 90%) до
                          0.658 (GIST, recall 99%), отдельно выделена точка c = 0.271 (OpenAI
                          text-embedding, recall 95%) как ближайший по природе аналог нашего
                          обученного эмбеддинга (в отличие от ручных признаков SIFT/GIST);
                        - измеренная у нас точка (по умолчанию 200000 векторов / 15 мс p50 —
                          controlled run на ноутбуке разработчика, docs/architecture.md, раздел 5),
                          её можно заменить на JSON-вывод benchmark_ann.py через --results.
  scaling_memory.png   память в RAM/на диске по формуле репозитория (vector/README.md, раздел
                        «Масштабирование»): векторы N*dim*4 байта (float32) или N*dim*1 байт
                        (int8, always_ram=True), граф HNSW ~ N*m*8 байт (эмпирически откалибровано
                        по «130 МБ при N=10^6, m=16» из того же README), диск при квантизации —
                        сумма квантованной и исходной float32 копии (нужна для рескоринга).

Кривые задержки — перенос чужой экспоненты на другие данные и железо, не измерение на нашем
стенде; это явно подписано на графике и в docs/scaling.md. Формулы памяти — прямая арифметика,
проверенная на официальных цифрах Qdrant (qdrant.tech/benchmarks) и на собственной таблице репозитория.

    python plot_scaling.py                                   # опорная точка по умолчанию
    python plot_scaling.py --results ../../bench_200k.json    # опорная точка из реального прогона
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / ".." / ".." / "docs" / "img"

# Показатели степени cost ~ N^c из arXiv:2412.01940, таблица по датасетам/recall (раздел с обзором).
# Полный диапазон (SIFT..GIST) — честная вилка неопределённости; OpenAI — ближайший по природе
# аналог обученного плотного эмбеддинга (в отличие от ручных признаков SIFT/GIST).
EXP_LOW = 0.185    # SIFT, recall@90%   — лучший случай в исследовании
EXP_MID = 0.271    # OpenAI embeddings, recall@95% — ближайший аналог по природе данных
EXP_HIGH = 0.658   # GIST, recall@99%   — худший случай в исследовании

# Память: N * dim * bytes_per_component, граф HNSW ~ N * m * BYTES_PER_GRAPH_EDGE
# (откалибровано по vector/README.md: ~130 МБ графа при N=10^6, m=16 => ~8 байт/вектор/m).
BYTES_PER_GRAPH_EDGE = 8


def latency_curves(n_grid, baseline_n, baseline_ms):
    naive = baseline_ms * np.log(n_grid) / math.log(baseline_n)
    power = {
        c: baseline_ms * (n_grid / baseline_n) ** c
        for c in (EXP_LOW, EXP_MID, EXP_HIGH)
    }
    return naive, power


def memory_curves(n_grid, dim, m):
    ram_f32_gb = n_grid * dim * 4 / 1e9
    ram_int8_gb = n_grid * dim * 1 / 1e9
    graph_gb = n_grid * m * BYTES_PER_GRAPH_EDGE / 1e9
    disk_f32_gb = ram_f32_gb  # некватизованные векторы: диск и RAM хранят одну и ту же float32-копию
    disk_int8_gb = ram_f32_gb + ram_int8_gb  # рескоринг: исходная float32-копия на диске + квантованная
    return {
        "ram_float32": ram_f32_gb + graph_gb,
        "ram_int8": ram_int8_gb + graph_gb,
        "disk_float32": disk_f32_gb + graph_gb,
        "disk_int8": disk_int8_gb + graph_gb,
    }


def load_baseline_from_results(paths):
    """Опорная точка (N, p50 мс) из JSON-вывода benchmark_ann.py; берёт последнюю по N запись."""
    best = None
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        n = data["gallery_size"]
        p50 = data["ann_latency_ms"]["p50"]
        if best is None or n > best[0]:
            best = (n, p50)
    return best


def plot_latency(baseline_n, baseline_ms, out_path):
    n_grid = np.logspace(4, 8, 60)
    naive, power = latency_curves(n_grid, baseline_n, baseline_ms)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_grid, naive, "--", color="#898781", linewidth=2, label="наивно: ~log(N) (не подтверждено измерением)")
    ax.plot(n_grid, power[EXP_LOW], color="#6250d6", linewidth=2,
            label=f"степенной рост, c={EXP_LOW} (SIFT, recall 90% — лучший случай)")
    ax.plot(n_grid, power[EXP_MID], color="#008300", linewidth=2,
            label=f"степенной рост, c={EXP_MID} (OpenAI embeddings, recall 95% — ближайший аналог)")
    ax.plot(n_grid, power[EXP_HIGH], "-.", color="#2a78d6", linewidth=2,
            label=f"степенной рост, c={EXP_HIGH} (GIST, recall 99% — худший случай)")
    ax.scatter([baseline_n], [baseline_ms], color="#eb6834", s=70, zorder=5,
               label=f"измерено на стенде (N={baseline_n:,}, p50={baseline_ms:.1f} мс)".replace(",", " "))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("размер галереи, векторов")
    ax.set_ylabel("задержка поиска p50, мс")
    ax.set_title("Задержка HNSW-поиска (Qdrant) в зависимости от размера галереи")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_memory(dim, m, out_path):
    n_grid = np.logspace(4, 8, 60)
    curves = memory_curves(n_grid, dim, m)
    milestone_n = np.array([1_000_000])
    milestone = memory_curves(milestone_n, dim, m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_grid, curves["ram_float32"], color="#2a78d6", linewidth=2, label="RAM, float32 (без квантизации)")
    ax.plot(n_grid, curves["ram_int8"], color="#008300", linewidth=2, label="RAM, int8 (QDRANT_QUANTIZATION=scalar)")
    ax.plot(n_grid, curves["disk_int8"], "--", color="#008300", linewidth=1.5, label="диск, int8 (+ float32 для рескоринга)")
    ax.scatter([1_000_000], milestone["ram_float32"], color="#eb6834", s=50, zorder=5)
    ax.scatter([1_000_000], milestone["ram_int8"], color="#eb6834", s=50, zorder=5,
               label="опорные точки из vector/README.md (N=10^6)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("размер галереи, векторов")
    ax.set_ylabel("память, ГБ")
    ax.set_title(f"Память Qdrant в зависимости от размера галереи (dim={dim}, HNSW m={m})")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-n", type=int, default=200_000,
                        help="опорный размер галереи (измеренная точка), по умолчанию прогон из docs/architecture.md")
    parser.add_argument("--baseline-ms", type=float, default=15.0, help="опорная задержка p50, мс")
    parser.add_argument("--results", nargs="+", default=None,
                        help="JSON-файлы вывода benchmark_ann.py: опорная точка берётся из них вместо --baseline-*")
    parser.add_argument("--dim", type=int, default=512, help="размерность эмбеддинга (EMBEDDING_DIM)")
    parser.add_argument("--m", type=int, default=16, help="параметр HNSW m (граф связей)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    baseline_n, baseline_ms = args.baseline_n, args.baseline_ms
    if args.results:
        baseline_n, baseline_ms = load_baseline_from_results(args.results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latency_path = args.out_dir / "scaling_latency.png"
    memory_path = args.out_dir / "scaling_memory.png"

    plot_latency(baseline_n, baseline_ms, latency_path)
    plot_memory(args.dim, args.m, memory_path)

    print(json.dumps({
        "baseline_n": baseline_n,
        "baseline_ms": baseline_ms,
        "exponents": {"low": EXP_LOW, "mid": EXP_MID, "high": EXP_HIGH},
        "dim": args.dim,
        "m": args.m,
        "latency_png": str(latency_path),
        "memory_png": str(memory_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
