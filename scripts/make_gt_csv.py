#!/usr/bin/env python3
"""Генерирует GT CSV в формате evaluate.py из валидационных сплитов.

Формат GT: image_id,vehicle_id,camera_id,split

Исходные файлы:
  reid/splits/val_query_clean.csv    (image_id,x,y,w,h,vehicle_id,camera_id)
  reid/splits/val_gallery_clean.csv  (image_id,x,y,w,h,vehicle_id,camera_id)
"""
import csv
import sys
from pathlib import Path

AQUILA = Path(__file__).resolve().parent.parent
SPLITS = AQUILA.parent / "reid" / "splits"
OUT = AQUILA / "data"
OUT.mkdir(exist_ok=True)


def convert(split_name: str, out_name: str = "val_gt.csv"):
    """Объединяет val_query_clean и val_gallery_clean в один GT CSV."""
    query_path = SPLITS / f"val_{split_name}_clean.csv"
    gallery_path = SPLITS / f"val_gallery_clean.csv"
    out_path = OUT / out_name

    # Проверяем наличие файлов
    if not query_path.exists():
        print(f"Не найден: {query_path}")
        sys.exit(1)
    if not gallery_path.exists():
        print(f"Не найден: {gallery_path}")
        sys.exit(1)

    rows = []

    # Query
    with open(query_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_id": row["image_id"],
                "vehicle_id": row["vehicle_id"],
                "camera_id": row["camera_id"],
                "split": "query",
            })

    # Gallery
    with open(gallery_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_id": row["image_id"],
                "vehicle_id": row["vehicle_id"],
                "camera_id": row["camera_id"],
                "split": "gallery",
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "vehicle_id", "camera_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_q = sum(1 for r in rows if r["split"] == "query")
    n_g = sum(1 for r in rows if r["split"] == "gallery")
    print(f"GT CSV -> {out_path}")
    print(f"  query:  {n_q}")
    print(f"  gallery: {n_g}")
    print(f"  total:  {len(rows)}")


if __name__ == "__main__":
    convert("query", "val_gt.csv")
