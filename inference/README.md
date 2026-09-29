# Inference (эмбеддинги ReID)

Сервис FastAPI + PyTorch: принимает кадр и BBox, вырезает автомобиль,
маскирует чужие машины через YOLO, возвращает 2048-мерный L2-нормированный эмбеддинг.
Также строит карты внимания для окна «Grad-CAM».

## Конвейер обработки

```
кадр JPEG/PNG + BBox (x, y, w, h)
       │
       ▼
torchvision.io.decode → GPU (CUDA)
       │
       ▼
YOLO11s-seg → маски всех автомобилей в кадре
       │
       ▼
apply_masking_gpu(): оставить только целевой автомобиль
  - найти маску с макс. IoU с GT BBox
  - залить фоном все остальные машины
  - вырезать кроп по BBox
       │
       ▼
Resize 320×320 → нормализация ImageNet
       │
       ▼
DINOv3 ViT-L/16 (LoRA r=32, α=64, blocks 14-23 unfrozen)
  → hidden states со слоёв [0, 4, 8, 12, 16, 20]
       │
       ▼
HeadBNNeck: 6 query-векторов → attention pool → concat → Linear(6144→2048)
  → BatchNorm1d → L2-normalization
       │
       ▼
эмбеддинг 2048 × float32 (L2-нормированный)
```

**Точность:** bf16 (Ampere+) / fp16 (Turing/RTX 2060) / fp32 (CPU) — автоопределение.

## Модель

| Параметр | Значение |
|----------|----------|
| Backbone | DINOv3 ViT-L/16 (HuggingFace `facebook/dinov3-large`) |
| Адаптация | LoRA rank=32, alpha=64, на `q/k/v/o` всех 24 слоёв |
| Разморожено | Blocks 14–23 (10 из 24) |
| Голова | HeadBNNeck: 6 queries, слои [0,4,8,12,16,20], proj 6144→2048, BN |
| Размерность | 2048 × float32, L2-норм |
| Вход | 320×320 (нормализация ImageNet) |
| Чекпоинт | `weights/JDNFV_MASKED_FT_best_fp16.pt` (~306 МБ) |
| Backbone веса | `weights/dinov3-vitl16-bf16/` (~579 МБ, bf16 safetensors) |
| YOLO | `weights/yolo11s-seg.pt` (~20 МБ) |
| Веса всего | **~905 МБ** (Git LFS) |

Подробнее: [`docs/training/architecture.md`](../docs/training/architecture.md),
[`docs/methods.md`](../docs/methods.md).

## Запуск

Вместе с остальным решением: `./run.sh`; без GPU `SKIP_SERVICES="inference" ./run.sh`.

| Что | Значение |
|---|---|
| Требования | NVIDIA GPU, Container Toolkit, ~4 ГБ VRAM |
| Порт | `127.0.0.1:8001` (`INFERENCE_PORT`) |
| Образ | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| Память | `INFERENCE_MEM_LIMIT` (по умолч. 4 ГБ) |

## Эндпоинты

| Метод | Путь | Назначение |
|-------|------|-----------|
| GET | `/health` | Статус сервиса |
| GET | `/internal/status` | Версия модели, dim, device, dtype |
| POST | `/internal/embed` | **Основной:** эмбеддинг + YOLO-маскирование |
| POST | `/internal/embed-batch` | Батч-эмбеддинг (без YOLO) |
| POST | `/internal/embed-batch-full` | Полный батч-пайплайн (YOLO + маски) |
| POST | `/internal/cls-attention` | CLS self-attention карта (Grad-CAM) |
| POST | `/internal/compare` | Grad-ATTN для пары (query vs candidate) |

Backend использует `/internal/embed` (для поиска) и `/internal/cls-attention` (для Grad-CAM).
