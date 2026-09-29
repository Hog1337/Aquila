# Обучение модели ReID

Модель идентификации транспортных средств основана на **DINOv3 ViT-L/16**
с адаптацией через **LoRA** и головой **HeadBNNeck** (Multi-Query Attention Pooling).
Обучение проходило в 5 последовательных этапов на трёх датасетах.

## Быстрая сводка

| Параметр | Значение |
|----------|----------|
| Backbone | DINOv3 ViT-L/16 (HuggingFace `facebook/dinov3-large`) |
| Адаптация | LoRA rank=32, alpha=64, на `q/k/v/o` всех 24 слоёв + unfreeze blocks 14–23 |
| Голова | HeadBNNeck: 6 query-векторов, attention pool над слоями [0,4,8,12,16,20], proj 6144→2048, BN, L2 |
| Размер входа | 320×320 (нормализация ImageNet) |
| Размерность эмбеддинга | 2048 × float32, L2-нормированный |
| Функции потерь | CE (label smoothing 0.1) + 0.3·SupCon + 0.15·ProtoCon |
| Оптимизатор | AdamW, EMA 0.999, Cosine LR schedule |
| Precision | BF16 mixed precision |
| Датасеты | Falcon ReID (основной) + DN-ReID + VeRI-Wild |
| Финальный mAP@10 | **0.839** |
| Финальный чекпоинт | `inference/weights/JDNFV_MASKED_FT_best_fp16.pt` |

## Документы

| Документ | О чём |
|----------|-------|
| [`data.md`](training/data.md) | Датасеты, генерация сплита, чистка данных |
| [`architecture.md`](training/architecture.md) | Архитектура модели: backbone, LoRA, head, loss |
| [`stages.md`](training/stages.md) | 5 этапов обучения с гиперпараметрами и результатами |
| [`hyperparameters.md`](training/hyperparameters.md) | Сводная таблица гиперпараметров |
| [`reproduction.md`](training/reproduction.md) | Пошаговая инструкция воспроизведения |
| [`checkpoints.md`](training/checkpoints.md) | Все чекпоинты с mAP |
| [`code-map.md`](training/code-map.md) | Карта файлов: какой скрипт за что отвечает |

## Ключевые бусты mAP

| Изменение | Δ mAP | Этап |
|-----------|:-----:|:----:|
| Joint DN-ReID + Falcon (вместо только Falcon) | +0.038 | 1 |
| +Supervised Contrastive Loss | +0.013 | 2 |
| +BNNeck + Prototype Contrastive | +0.020 | 3 |
| +VeRI-Wild + LoRA r32 + 2048d | +0.069 | 4 |
| EMA 0.999 | +0.032 | 4 |
| Masked fine-tune + чистка данных | +0.006 | 5 |

## Где находится код обучения

Весь код обучения — в каталоге [`training/`](../training/).
Скрипты запускались на стенде разработчика (2× RTX 5000 Ada 32GB,
центральный сервер gpusad с Blackwell RTX PRO 6000 98GB).
