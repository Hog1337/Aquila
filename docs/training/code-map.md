# Карта файлов: код обучения

## Обучение модели (по этапам)

| Файл | Назначение | Ключевые особенности |
|------|-----------|---------------------|
| [`training/baseline/joint_dn_falcon.py`](../training/baseline/joint_dn_falcon.py) | **Этап 1:** CE baseline, Joint DN+Falcon | OptionC Head, LoRA r16, 512d, 30ep, финиш только на Falcon |
| [`training/baseline/joint_supcon.py`](../training/baseline/joint_supcon.py) | **Этап 2:** +SupCon Loss (λ=0.3, T=0.07) | CE + SupCon, 40ep |
| [`training/baseline/joint_bnneck_proto.py`](../training/baseline/joint_bnneck_proto.py) | **Этап 3:** +BNNeck + ProtoCon (λ=0.15) | BatchNorm перед CE, momentum prototype bank, 40ep |
| [`training/baseline/joint_triple.py`](../training/baseline/joint_triple.py) | **Этап 4:** +VeRI-Wild, r32, 2048d | 3 датасета, аугментации, EMA 0.999, Cross-View Hard Sampler |
| [`training/baseline/joint_masked_ft.py`](../training/baseline/joint_masked_ft.py) | **Этап 5:** Masked fine-tune | YOLO-маскированные кропы, чистые метки, LR=5e-6 |

## Валидация и оценка

| Файл | Назначение |
|------|-----------|
| [`training/baseline/eval_multilayer.py`](../training/baseline/eval_multilayer.py) | Полная валидация модели: mAP@10, Rank-1/5, кандидатная метрика |
| [`training/baseline/bench_latency.py`](../training/baseline/bench_latency.py) | Замер latency (batch=1) и throughput (лучший FPS) |
| [`training/verify_checkpoint.py`](../training/verify_checkpoint.py) | Проверка целостности чекпоинта |
| [`training/verify_best.py`](../training/verify_best.py) | Проверка лучшего чекпоинта на val |
| [`training/../reid/analyze_failures.py`](../../reid/analyze_failures.py) | Анализ ошибок модели (веб-инструмент на Flask + V2-эмбеддинги) |
| [`training/../reid/analyze_failures_v2.py`](../../reid/analyze_failures_v2.py) | Улучшенная версия анализа ошибок |

## Датасеты и данные

| Файл | Назначение |
|------|-----------|
| [`training/split.py`](../training/split.py) | Генерация val-сплита (train_sub / val_query / val_gallery) |
| [`training/clean_dataset.py`](../training/clean_dataset.py) | Интерактивная чистка датасета (Flask + V2-эмбеддинги) |
| [`training/baseline/make_car_dataset.py`](../training/baseline/make_car_dataset.py) | Подготовка датасета carseg для YOLO |
| [`training/baseline/make_carseg_sam3v2.py`](../training/baseline/make_carseg_sam3v2.py) | Подготовка масок через SAM3 (альтернативный подход) |

## Вспомогательные скрипты (из `../reid/`, не включены в `training/`)

Эти скрипты использовались в процессе исследований, но не являются частью
основного пайплайна обучения:

| Файл (в `../reid/baseline/`) | Назначение |
|------------------------------|-----------|
| `falcon_ft.py` | Ранний baseline (только Falcon, без external data) |
| `falcon_polish.py` | Улучшенный baseline (0.7026 mAP) |
| `convnext_joint.py` | ConvNeXt-L трек (закрыт: 0.7527 < ViT-L 0.8431) |
| `joint_aug_finetune.py` | Эксперимент с расширенными аугментациями |
| `boxnoise_ablation.py` | Абляция по шуму BBox |
| `joint_arcface.py` | Эксперимент с ArcFace (закрыт) |
| `eval_tta_rerank.py` | TTA + re-ranking эксперименты |
| `analize_failures.py` | Анализ ошибок модели |

## Сплиты

| Файл | Описание |
|------|----------|
| [`training/splits/train_sub.csv`](../training/splits/train_sub.csv) | Обучающая выборка (7 176 изобр., 1 156 машин) |
| [`training/splits/val_query.csv`](../training/splits/val_query.csv) | Запросы валидации (1 261 изобр., 385 машин) |
| [`training/splits/val_gallery.csv`](../training/splits/val_gallery.csv) | Галерея валидации (1 119 изобр., 341 машина) |
| [`training/splits/*_clean.csv`](../training/splits/) | Те же сплиты после чистки меток |
| [`training/splits/meta.json`](../training/splits/meta.json) | Параметры нарезки сплита |

## Архитектура модели (код инференса)

| Файл | Назначение |
|------|-----------|
| [`inference/app/reid_embed.py`](../inference/app/reid_embed.py) | LoRALinear, HeadBNNeck, AttnPoolHead, загрузчик чекпоинта, эмбеддинг |
| [`inference/app/main.py`](../inference/app/main.py) | FastAPI-сервис: эндпоинты `/internal/embed`, `/internal/attention`, YOLO-маскирование |
| [`inference/app/yolo_seg.py`](../inference/app/yolo_seg.py) | Функция маскирования `apply_masking_gpu` |
