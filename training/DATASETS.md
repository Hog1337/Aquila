# Датасеты для обучения

Для воспроизведения обучения необходимы три датасета. Два из них — публичные
(DN-ReID, VeRI-Wild), третий (Falcon ReID) предоставляется организаторами хакатона.

---

## 1. Falcon ReID (основной)

| Характеристика | Значение |
|---------------|----------|
| Источник | Предоставлен организаторами хакатона «Фалькон Тех» |
| Лицензия | Служебная (организаторов) |
| Изображения | 9 556 JPEG, 1920×1080 |
| Машин (ID) | 1 541 |
| Камер | 96 |
| Формат | `images/` (плоский каталог) + `train.csv` |

### Ожидаемая структура

```
/path/to/data/reid/
├── images/              # все JPEG, плоский каталог
│   ├── 000fced9a580495599a6e9b12fbe0050.jpg
│   └── ...
├── train.csv            # image_id,x,y,w,h,vehicle_id,camera_id
├── test_gallery.csv     # image_id,x,y,w,h (без меток)
└── test_query.csv       # image_id,x,y,w,h (без меток)
```

### Формат CSV

```csv
image_id,x,y,w,h,vehicle_id,camera_id
000fced9a580495599a6e9b12fbe0050,380,240,670,490,42,7
```

- `image_id` — уникальный идентификатор кадра (он же имя файла без расширения)
- `x, y, w, h` — BBox в пикселях исходного кадра
- `vehicle_id` — идентификатор автомобиля (только в train)
- `camera_id` — анонимный идентификатор камеры (только в train)

---

## 2. DN-ReID (Day/Night)

| Характеристика | Значение |
|---------------|----------|
| Источник | [DN-ReID на GitHub](https://github.com/zhangchaobin0928/DN-ReID) |
| Лицензия | CC BY-NC |
| Изображений (train) | ~106 000 |
| Машин (ID) | 1 574 |
| Формат | Конвертирован в единый CSV + плоский каталог изображений |

### Подготовка

```bash
# Склонировать репозиторий
git clone https://github.com/zhangchaobin0928/DN-ReID.git

# Запустить конвертацию в единый формат
python baseline/make_car_dataset.py \
  --input /path/to/DN-ReID \
  --output /path/to/datasets/dnreid_convert
```

### Ожидаемая структура после конвертации

```
/path/to/datasets/dnreid_convert/
├── images/              # все JPEG, плоский каталог
└── train.csv            # image_id,vehicle_id (с конвертированными ID)
```

---

## 3. VeRI-Wild

| Характеристика | Значение |
|---------------|----------|
| Источник | [VeRI-Wild на GitHub](https://github.com/PKU-IMRE/VERI-Wild) |
| Лицензия | CC BY-NC |
| Изображений (train) | ~277 000 |
| Машин (ID) | 30 671 |
| Формат | Оригинальный (изображения по папкам ID) |

### Подготовка

```bash
# Скачать датасет (требуется регистрация)
# Распаковать в /path/to/datasets/veriwild_images/

# Структура после распаковки:
/path/to/datasets/veriwild_images/
├── images/
│   ├── 000001/          # папка ID машины
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── 000002/
│   └── ...
└── train_test_split/
    └── train_list_start0.txt   # список файлов для обучения
```

> **Важно:** Скрипты обучения ожидают именно такую структуру.
> Файл `train_list_start0.txt` содержит строки формата:
> `000001/000001.jpg camera_id vehicle_id`
> (пробелы — разделители; 3 колонки: путь, камера, ID)

---

## Сопоставление путей в скриптах

Скрипты обучения используют переменные в начале файла, которые нужно
исправить под ваше окружение:

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `BASE` | `/home/a.a.milkevich/temp_exp` | Корень данных |
| `DINO` | `$BASE/reid/models/dinov3-vitl16` | DINOv3 ViT-L/16 (HuggingFace) |
| `CKPT` | `$BASE/reid/verid/artifacts/checkpoints/LLRD6b_best.pt` | Warm-start чекпоинт |
| `OUT` | `$BASE/reid/baseline/out` | Куда сохранять результаты |
| `SPLITS` | `$BASE/reid/splits` | Сплиты (копия `training/splits/`) |
| `IMGS` | `$BASE/data/reid/images` | Falcon изображения |
| `DN_DATA` / `DN_IMG` | `$BASE/datasets/dnreid_convert` | DN-ReID |
| `VW_IMG` | `$BASE/datasets/veriwild_images/images` | VeRI-Wild |

---

## Сплит для валидации

Сплит уже сгенерирован и лежит в `training/splits/`. Для перегенерации:

```bash
python split.py
# Результат: splits/train_sub.csv, val_query.csv, val_gallery.csv
```

Принципы нарезки (детально: [`docs/training/data.md`](../docs/training/data.md)):
- Разрезка по машинам (vehicle_id), не по кадрам — **никакого leakage**
- 25% машин → val, 75% → train_sub
- 20% val-машин → orphan (только query, без пары в галерее) — для калибровки TNR
- 80% val-машин → paired (каждый query имеет ≥1 кросс-камерный позитив)
- Учёт md5-дублей (одна сцена, две машины) — обе на одной стороне
