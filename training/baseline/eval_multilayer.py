#!/usr/bin/env python3
"""Оценка multi-layer fusion: на извлечённых фичах пробуем разные комбинации слоёв.

Быстрый probe без обучения — просто cosine similarity на разных fusion-вариантах,
сравниваем mAP@10 с baseline (только layer 24).

Запуск: python3 eval_multilayer.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def load_split(name):
    z = np.load(OUT / f"multilayer_{name}.npz", allow_pickle=True)
    layers = z["layers"].tolist()
    return z, layers


def evaluate(query, gallery, layer_combo, mode="apool"):
    """Вычислить mAP@10 для fusion слоёв layer_combo.
    mode='apool': attention-pooled фичи (512d)
    mode='cls': CLS-токены (1024d)
    """
    q = query
    g = gallery
    q_veh, q_cam = q["vehicle_ids"], q["camera_ids"]
    g_veh, g_cam = g["vehicle_ids"], g["camera_ids"]

    if mode == "apool":
        dim = 512
        q_feats = np.concatenate([q["apool_%d" % l][:, None, :] for l in layer_combo], axis=1)
        g_feats = np.concatenate([g["apool_%d" % l][:, None, :] for l in layer_combo], axis=1)
    else:
        dim = 1024
        q_feats = np.concatenate([q["cls_%d" % l][:, None, :] for l in layer_combo], axis=1)
        g_feats = np.concatenate([g["cls_%d" % l][:, None, :] for l in layer_combo], axis=1)

    # Fusion: concat → L2
    Nq, Nl, D = q_feats.shape
    Ng = g_feats.shape[0]

    # Reshape: (N, Nl*D) — concat по слоям
    q_flat = q_feats.reshape(Nq, Nl * D)
    g_flat = g_feats.reshape(Ng, Nl * D)

    # L2-normalize
    q_flat = q_flat / (np.linalg.norm(q_flat, axis=1, keepdims=True) + 1e-9)
    g_flat = g_flat / (np.linalg.norm(g_flat, axis=1, keepdims=True) + 1e-9)

    # Cosine similarity
    scores = q_flat @ g_flat.T

    # mAP@10
    order = np.argsort(np.negative(scores), kind="stable", axis=1)
    aps = []
    for i in range(Nq):
        qv, qc = int(q_veh[i]), int(q_cam[i])
        same_veh = g_veh == qv
        junk = same_veh & (g_cam == qc)
        valid = same_veh & (g_cam != qc)
        n_pos = int(valid.sum())
        if n_pos == 0:
            continue
        keep = [int(j) for j in order[i] if not junk[j]]
        hits = valid[np.array(keep[:10], dtype=int)].astype(int) if keep else np.zeros(0, int)
        if len(hits) == 0:
            continue
        tp = np.cumsum(hits)
        prec = tp / np.arange(1, len(hits) + 1)
        aps.append(float((prec * hits).sum() / min(n_pos, 10)))

    return float(np.mean(aps)) if aps else 0.0


def main():
    print("Загрузка фич...", flush=True)
    q, layers = load_split("val_query")
    g, _ = load_split("val_gallery")
    print(f"  query: {len(q['image_ids'])} кадров, gallery: {len(g['image_ids'])} кадров")
    print(f"  слои: {layers}\n")

    # Сценарии fusion
    experiments = []
    # Baseline: только layer 24
    experiments.append(("baseline L24", [24], "apool"))
    experiments.append(("baseline L24 CLS", [24], "cls"))
    # Последние слои
    experiments.append(("L24+L23", [24, 23], "apool"))
    experiments.append(("L24+L23+L22", [24, 23, 22], "apool"))
    experiments.append(("L24+L23+L22+L20", [24, 23, 22, 20], "apool"))
    # Ранние + поздние
    experiments.append(("L24+L12", [24, 12], "apool"))
    experiments.append(("L24+L8", [24, 8], "apool"))
    experiments.append(("L24+L4", [24, 4], "apool"))
    experiments.append(("L24+L0", [24, 0], "apool"))
    # Все ключевые
    experiments.append(("L24+L12+L4", [24, 12, 4], "apool"))
    experiments.append(("L24+L12+L4+L0", [24, 12, 4, 0], "apool"))
    # CLS fusion
    experiments.append(("L24+L12 CLS", [24, 12], "cls"))

    print(f"{'Вариант':<30} {'mAP@10':>8} {'Δ vs L24':>10}")
    print("-" * 50)
    baseline_map = None
    results = []
    for name, combo, mode in experiments:
        try:
            m = evaluate(q, g, combo, mode)
            results.append((name, m, combo, mode))
            if baseline_map is None and "baseline" in name:
                baseline_map = m
            delta = m - baseline_map if baseline_map is not None else 0.0
            print(f"{name:<30} {m:>8.4f} {delta:>+10.4f}")
        except Exception as e:
            print(f"{name:<30} ОШИБКА: {e}")

    print("\nЛучшие:")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, m, combo, mode in results[:5]:
        delta = m - baseline_map if baseline_map is not None else 0.0
        print(f"  {name:<30} {m:>8.4f} (Δ={delta:+.4f})")

    # Также попробуем mean pool (не concat)
    print("\n=== Mean pool fusion ===")
    for label, combo in [("L24+L23 mean", [24, 23]),
                          ("L24+L23+L22 mean", [24, 23, 22]),
                          ("L24+L12 mean", [24, 12]),
                          ("all layers mean", layers)]:
        q_keys = ["apool_%d" % l for l in combo]
        g_keys = ["apool_%d" % l for l in combo]
        q_feats = np.mean(np.stack([q[k] for k in q_keys], axis=0), axis=0)
        g_feats = np.mean(np.stack([g[k] for k in g_keys], axis=0), axis=0)
        # L2
        q_fn = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-9)
        g_fn = g_feats / (np.linalg.norm(g_feats, axis=1, keepdims=True) + 1e-9)
        scores_m = q_fn @ g_fn.T
        order_m = np.argsort(np.negative(scores_m), kind="stable", axis=1)
        aps = []
        for i in range(len(q_fn)):
            qv, qc = int(q["vehicle_ids"][i]), int(q["camera_ids"][i])
            same_veh = g["vehicle_ids"] == qv
            junk = same_veh & (g["camera_ids"] == qc)
            valid = same_veh & (g["camera_ids"] != qc)
            n_pos = int(valid.sum())
            if n_pos == 0: continue
            keep = [int(j) for j in order_m[i] if not junk[j]]
            hits = valid[np.array(keep[:10], dtype=int)].astype(int) if keep else np.zeros(0, int)
            if len(hits) == 0: continue
            tp = np.cumsum(hits)
            prec = tp / np.arange(1, len(hits) + 1)
            aps.append(float((prec * hits).sum() / min(n_pos, 10)))
        mm = float(np.mean(aps)) if aps else 0.0
        delta = mm - baseline_map if baseline_map is not None else 0.0
        print(f"  {label:<30} {mm:>8.4f} (Δ={delta:+.4f})")


if __name__ == "__main__":
    main()
