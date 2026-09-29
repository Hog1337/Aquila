#!/usr/bin/env python3
"""
Сборка датасета «car mask» (1 класс) из stage2_real_v2 (20 частей, YOLO-seg):
  car = union 20 масок частей + dilation -> внешний контур -> YOLO-seg polygon.
Вход:  /home/limon/data/university/lct/datasets/stage2_real_v2
Выход: reid/baseline/dataset/carseg/{images/{train,val} (symlink), labels/{train,val}/*.txt, data.yaml}
"""
import os
import shutil

import cv2
import numpy as np

SRC = "/home/limon/data/university/lct/datasets/stage2_real_v2"
OUT = "/home/limon/data/university/lct/reid/baseline/dataset/carseg"


def build_split(split):
    img_dir_src = os.path.join(SRC, "images", split)
    lbl_dir_src = os.path.join(SRC, "labels", split)
    img_dir_out = os.path.join(OUT, "images", split)
    lbl_dir_out = os.path.join(OUT, "labels", split)
    os.makedirs(lbl_dir_out, exist_ok=True)
    if not os.path.islink(img_dir_out) and not os.path.isdir(img_dir_out):
        os.makedirs(os.path.dirname(img_dir_out), exist_ok=True)
        os.symlink(img_dir_src, img_dir_out)

    files = sorted(f for f in os.listdir(img_dir_src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    n_car = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for fname in files:
        base = os.path.splitext(fname)[0]
        lbl = os.path.join(lbl_dir_src, base + ".txt")
        if not os.path.exists(lbl):
            continue
        img = cv2.imread(os.path.join(img_dir_src, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        with open(lbl) as f:
            for line in f:
                toks = line.split()
                if len(toks) < 7:
                    continue
                k = (len(toks) - 1) // 2
                if k < 3:
                    continue
                pts = np.array([[float(toks[1 + 2 * i]) * w, float(toks[2 + 2 * i]) * h] for i in range(k)], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
        if mask.sum() < 1000:
            continue
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 3:
            continue
        approx = np.clip(approx / [w, h], 0, 1)
        if len(approx) > 128:
            keep = np.linspace(0, len(approx) - 1, 128).astype(int)
            approx = approx[keep]
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in approx)
        with open(os.path.join(lbl_dir_out, base + ".txt"), "w") as f:
            f.write(f"0 {coords}\n")
        n_car += 1
    print(f"[{split}] images={len(files)} car-labels={n_car}", flush=True)


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    for split in ("train", "val"):
        build_split(split)
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(
            f"path: {OUT}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names: {0: car}\n"
        )
    print("done ->", OUT)


if __name__ == "__main__":
    main()
