"""
Генератор val-сплита для задачи ReID (Фалькон Тех, задача 7).

Принципы (см. reid/DATASET_NOTES.md):
  1. Разрезка ЦЕЛИКОМ по машинам (vehicle_id): все кадры одной машины
     только на одной стороне train_sub/val.
  2. md5-дубли кадров (одна сцена, два BBox, две разные машины): обе машины
     пары — на одной стороне (union-find по «общему кадру»).
  3. val_vehicles (~25% машин) делятся:
       - orphans (~20%): все кадры -> val_query, в val_gallery их НЕТ
         (имитация ~20% «пустых» query скрытого теста -> калибровка порога отказа);
       - paired  (~80%): КАМЕРЫ машины разбиваются на query-сторону и
         gallery-сторону так, чтобы ни одна (машина, камера) не оказалась с
         обеих сторон. Тогда у КАЖДОГО val_query из paired гарантирован
         >=1 кросс-камерный позитив в val_gallery (как в публичном тесте).
  4. Камеры НЕ откладываем целиком (в реальной эксплуатации галерея строится
     теми же камерами).

Результат (reid/splits/):
  train_sub.csv    - для обучения (image_id, x, y, w, h, vehicle_id, camera_id)
  val_query.csv    - для локальной оценки (тот же формат + метки для evaluate.py)
  val_gallery.csv  - для локальной оценки (тот же формат + метки)
  meta.json        - параметры нарезки и сводка

Скрипт детерминирован: SEED = 42.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.25      # доля машин, уходящих в val
ORPHAN_QUERY_FRACTION = 0.20  # доля «сирот» среди val QUERY (как в скрытом тесте)

DATA = Path(__file__).resolve().parent.parent / "data" / "reid"
OUT = Path(__file__).resolve().parent / "splits"
OUT.mkdir(exist_ok=True)

FIELDS = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]


def load_train():
    rows = list(csv.DictReader(open(DATA / "train.csv")))
    for r in rows:
        r["vehicle_id"] = int(r["vehicle_id"])
        r["camera_id"] = int(r["camera_id"])
    return rows


def dup_components(rows):
    """Соединяет машины, снятые в одном и том же кадре (md5-дубли)."""
    by_hash = defaultdict(list)
    import hashlib

    for r in rows:
        h = hashlib.md5((DATA / "images" / f"{r['image_id']}.jpg").read_bytes()).hexdigest()
        by_hash[h].append(r["vehicle_id"])
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for vids in by_hash.values():
        if len(vids) > 1:
            for v in vids[1:]:
                union(vids[0], v)
    comps = defaultdict(list)
    for r in rows:
        comps[find(r["vehicle_id"])].append(r["vehicle_id"])
    return [sorted(set(v)) for v in comps.values()]


def main():
    random.seed(SEED)
    rows = load_train()
    veh_imgs = defaultdict(list)
    for r in rows:
        veh_imgs[r["vehicle_id"]].append(r)

    comps = dup_components(rows)
    # базовая «атомарная единица» отсечения: компонент (обычно 1 машина)
    comps.sort()
    random.shuffle(comps)

    val_vehicles = set()
    for comp in comps:
        if len(val_vehicles) >= int(len(veh_imgs) * VAL_FRACTION):
            break
        val_vehicles.update(comp)

    # orphans vs paired внутри val:
    # подбираем количество orphans так, чтобы доля «пустых» запросов
    # (orphan-кадры / все val-запросы) ~= ORPHAN_QUERY_FRACTION
    # (в скрытом тесте ~20% именно запросов, а не машин).
    val_veh_list = sorted(val_vehicles)
    random.shuffle(val_veh_list)
    a = {v: len(veh_imgs[v]) for v in val_veh_list}

    def orphan_ratio(k):
        oq = sum(a[val_veh_list[i]] for i in range(k))
        pq = sum(round(a[val_veh_list[i]] / 2) for i in range(k, len(val_veh_list)))
        return oq / (oq + pq)

    lo, hi = 0, len(val_veh_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if orphan_ratio(mid) < ORPHAN_QUERY_FRACTION:
            lo = mid + 1
        else:
            hi = mid
    n_orphans = lo
    orphans, paired = set(val_veh_list[:n_orphans]), set(val_veh_list[n_orphans:])

    train_rows, q_rows, g_rows = [], [], []

    for v in sorted(veh_imgs):
        if v not in val_vehicles:
            train_rows.extend(veh_imgs[v])
            continue
        if v in orphans:
            q_rows.extend(veh_imgs[v])  # в галерее этих машин нет вообще
            continue
        # paired: разбиваем КАМЕРЫ машины на query/gallery стороны
        cams = defaultdict(list)
        for im in veh_imgs[v]:
            cams[im["camera_id"]].append(im)
        # перебор подмножеств камер: closest к половине снимков, обе стороны непусты
        cam_ids = sorted(cams)
        n = len(cam_ids)
        total = len(veh_imgs[v])
        best, best_diff = None, None
        for mask in range(1, (1 << n) - 1):
            qset = {c for i, c in enumerate(cam_ids) if mask >> i & 1}
            diff = abs(sum(len(cams[c]) for c in qset) - total / 2)
            if best_diff is None or diff < best_diff:
                best, best_diff = qset, diff
        for c in cam_ids:
            for im in cams[c]:
                (q_rows if c in best else g_rows).append(im)

    # сортируем в «естественном» порядке: по image_id-порядку исходного train.csv,
    # чтобы файлы были устойчивыми к повторным прогонам
    order = {r["image_id"]: i for i, r in enumerate(rows)}
    for part in (train_rows, q_rows, g_rows):
        part.sort(key=lambda r: order[r["image_id"]])

    for name, part in [("train_sub.csv", train_rows), ("val_query.csv", q_rows), ("val_gallery.csv", g_rows)]:
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(part)

    # ---------- самодиагностика ----------
    tv = {r["vehicle_id"] for r in train_rows}
    qv = {r["image_id"]: r["vehicle_id"] for r in q_rows}
    gv = {r["image_id"]: r["vehicle_id"] for r in g_rows}
    assert not (tv & set(qv.values())), "машина из train в val_query!"
    assert not (tv & set(gv.values())), "машина из train в val_gallery!"
    assert not (set(qv.values()) & set(gv.values())) or True  # пересечение ДОПУСТИМО (paired)
    # orphan-машины не должны быть в галерее
    assert not (orphans & set(gv.values())), "orphan-машина попала в val_gallery!"
    # paired query: >=1 кросс-камерный позитив в галерее
    pos = defaultdict(list)
    for r in g_rows:
        pos[r["vehicle_id"]].append(r["camera_id"])
    bad = 0
    for r in q_rows:
        v = r["vehicle_id"]
        if v in paired and not any(c != r["camera_id"] for c in pos[v]):
            bad += 1
    assert bad == 0, f"{bad} paired-query без кросс-камерного позитива"
    # md5-дубли: обе машины сцены на одной стороне
    seen = {}
    import hashlib
    for r in rows:
        h = hashlib.md5((DATA / "images" / f"{r['image_id']}.jpg").read_bytes()).hexdigest()
        side = "train" if r["image_id"] in {x["image_id"] for x in train_rows} else "val"
        if h in seen and seen[h] != side:
            raise AssertionError(f"md5-дубль по разные стороны: {h} {seen[h]}/{side}")
        seen[h] = side

    meta = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "orphan_query_fraction": ORPHAN_QUERY_FRACTION,
        "counts": {
            "train_sub_images": len(train_rows),
            "train_sub_vehicles": len(tv),
            "val_query_images": len(q_rows),
            "val_query_vehicles": len(set(qv.values())),
            "val_gallery_images": len(g_rows),
            "val_gallery_vehicles": len(set(gv.values())),
            "val_paired_vehicles": len(paired),
            "val_orphan_vehicles": len(orphans),
            "val_orphan_queries": sum(1 for r in q_rows if r["vehicle_id"] in orphans),
            "cameras_in_train_sub": len({r["camera_id"] for r in train_rows}),
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
