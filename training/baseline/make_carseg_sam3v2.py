#!/usr/bin/env python3
"""Фильтр SAM3-псевдолейблов и сборка YOLO-seg датасета carseg_sam3.
Критерии «чистого» лейбла:
- coverage: 0.15 <= доля авто в кропе <= 0.97
- solidity: area/convex_hull >= 0.78 (пёстрые/дырявые маски отбрасываем)
- маска не «приклеена» ко всем 4 граням кропа одновременно
Выход: dataset/carseg_sam3v2/ (images/{train,val}, labels/{train,val}, data.yaml)
train/val — тот же разрез по vehicle_id, что и reid splits (train_sub / val_*).
"""
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT.parent / "splits"
IMGS = ROOT.parent.parent / "data" / "reid" / "images"
OUT = ROOT / "dataset" / "carseg_sam3v2"
OUT.mkdir(parents=True, exist_ok=True)

COVER_MIN, COVER_MAX = 0.15, 0.97
SOLIDITY_MIN = 0.78


def mask_to_polygon(m):
    """binary (H,W) -> normalized polygon [x,y]*k (<=128 pts) или None."""
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1e3:
        return None
    eps = 0.002 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True)
    poly = poly.reshape(-1, 2).astype(np.float32)
    if len(poly) > 128:
        poly = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    h, w = m.shape
    return poly.tolist(), float(cv2.contourArea(c) / hull_area)


def main():
    stats = {}
    n_bad = {"cover": 0, "solidity": 0, "noglue": 0, "nomask": 0, "ok": 0}
    for split in ("train_sub", "val_query", "val_gallery"):
        mdir = ROOT / "out" / f"sam3masks_v2full_{split}"
        dst_img = OUT / "images" / (split)
        dst_lab = OUT / "labels" / (split)
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        rows = list(csv.DictReader(open(SPLITS / f"{split}.csv")))
        n_ok = 0
        for r in rows:
            iid = r["image_id"]
            src = str(IMGS / f"{iid}.jpg")
            mp = mdir / f"{iid}.png"
            dst_imgf = dst_img / f"{iid}.jpg"
            dst_labf = dst_lab / f"{iid}.txt"
            if dst_imgf.exists() and dst_labf.exists():
                n_ok += 1
                continue
            if not mp.exists():
                n_bad["nomask"] += 1
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                n_bad["nomask"] += 1
                continue
            h, w = m.shape
            cover = m.sum() / 255 / (h * w)
            if cover < COVER_MIN or cover > COVER_MAX:
                n_bad["cover"] += 1
                continue
            poly, solidity = mask_to_polygon(m)
            if poly is None or solidity < SOLIDITY_MIN:
                n_bad["solidity"] += 1
                continue
            # приклеена ко всем 4 граням
            k = 4
            glued = all([
                (m[:k, :] > 0).mean() > 0.5, (m[-k:, :] > 0).mean() > 0.5,
                (m[:, :k] > 0).mean() > 0.5, (m[:, -k:] > 0).mean() > 0.5,
            ])
            if glued:
                n_bad["noglue"] += 1
                continue
            # хардлинк изображения + лейбл YOLO-seg
            img = cv2.imread(src)
            x, y, w, h = (int(r[k]) for k in "xywh")
            cv2.imwrite(str(dst_imgf), img[y:y + h, x:x + w])
            xs = [p[0] / w for p in poly]
            ys = [p[1] / h for p in poly]
            pts = " ".join(f"{x:.5f} {y:.5f}" for x, y in zip(xs, ys))
            with open(dst_labf, "w") as f:
                f.write(f"0 {pts}\n")
            n_ok += 1
        stats[split] = (n_ok, len(rows))
        print(f"{split}: ok={n_ok}/{len(rows)}", flush=True)
    (OUT / "data.yaml").write_text(
        "path: " + str(OUT) + "\ntrain: images/train_sub\nval: images/val_query\n"
        "names: {0: car}\n")
    print("недопустимо:", n_bad)
    print("датасет:", OUT)


if __name__ == "__main__":
    main()
