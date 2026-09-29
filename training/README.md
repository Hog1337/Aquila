# Обучение модели ReID

Эта папка содержит исходный код обучения модели идентификации транспортных средств
(см. [`docs/training.md`](../docs/training.md) — главный документ с описанием методологии).

## Структура

```
training/
├── README.md                   ← этот файл
├── DATASETS.md                 ← где брать датасеты, как подготовить
├── requirements.txt            ← версии пакетов на момент обучения
├── split.py                    ← генерация val-сплита (train_sub / val_query / val_gallery)
├── clean_dataset.py            ← инструмент чистки меток (веб-интерфейс)
├── verify_checkpoint.py        ← проверка целостности чекпоинта (загрузка + test forward)
├── verify_best.py              ← быстрая проверка лучшего чекпоинта на val (mAP)
├── baseline/
│   ├── joint_dn_falcon.py      ← Этап 1: Joint DN-ReID + Falcon (CE baseline)
│   ├── joint_supcon.py         ← Этап 2: +Supervised Contrastive Loss
│   ├── joint_bnneck_proto.py   ← Этап 3: +BNNeck + Prototype Contrastive
│   ├── joint_triple.py         ← Этап 4: +VeRI-Wild, LoRA rank 32, 2048d
│   ├── joint_masked_ft.py      ← Этап 5: Masked fine-tune (финальный)
│   ├── eval_multilayer.py      ← Валидация модели на val-сплите
│   ├── bench_latency.py        ← Замер производительности (latency / throughput)
│   ├── make_car_dataset.py     ← Подготовка датасета для YOLO-seg (carseg)
│   └── make_carseg_sam3v2.py   ← Подготовка масок через SAM3
├── splits/                     ← Сгенерированные сплиты (CSV)
│   ├── train_sub.csv           ← обучающая выборка (7 176 изобр., 1 156 машин)
│   ├── val_query.csv           ← запросы валидации (1 261 изобр., 385 машин)
│   ├── val_gallery.csv         ← галерея валидации (1 119 изобр., 341 машина)
│   ├── *_clean.csv             ← те же сплиты после чистки меток
│   └── meta.json               ← параметры нарезки
└── out/                        ← сюда сохраняются чекпоинты при обучении (в репозитории пусто;
                                  финальный чекпоинт — `inference/weights/JDNFV_MASKED_FT_best_fp16.pt`)
```

## Быстрый старт

Для воспроизведения обучения потребуется:

1. **GPU** с ≥24 ГБ VRAM (рекомендуется RTX 5000 Ada / A5000)
2. **Датасеты:** Falcon (от организаторов), DN-ReID, VeRI-Wild (см. `DATASETS.md`)
3. **Предобученный DINOv3 ViT-L/16** (скачивается автоматически через HuggingFace)

```bash
cd training

# Установка зависимостей
pip install -r requirements.txt

# Генерация val-сплита (если нужно пересоздать)
python split.py

# Обучение по этапам (см. docs/training/stages.md)
python baseline/joint_dn_falcon.py
python baseline/joint_supcon.py
python baseline/joint_bnneck_proto.py
python baseline/joint_triple.py
python baseline/joint_masked_ft.py

# Валидация финального чекпоинта
python baseline/eval_multilayer.py --checkpoint out/JDNFV_MASKED_FT_best_fp16.pt
```

> **Важно:** Пути к данным в скриптах захардкожены на окружение разработчика.
> Перед запуском необходимо исправить пути в соответствии с `DATASETS.md`.
