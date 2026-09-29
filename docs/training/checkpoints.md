# Чекпоинты

## Финальный чекпоинт (в сервисе)

| Файл | Размер | mAP@10 |
|------|:------:|:------:|
| `inference/weights/JDNFV_MASKED_FT_best_fp16.pt` | 306 МБ | **0.839** |

Этот чекпоинт содержит LoRA-адаптеры + веса головы + размороженные блоки 14–23.
Backbone (DINOv3 ViT-L/16, bf16) — в отдельной папке `inference/weights/dinov3-vitl16-bf16/` (579 МБ).

## Промежуточные чекпоинты

Все чекпоинты хранятся в `training/out/` (после обучения).

| Файл | Этап | mAP@10 | Размер | Описание |
|------|:----:|:------:|:------:|----------|
| `LLRD6b_best.pt` | 0 | — | ~400 МБ | Warm-start (VeRi-776, AttnPoolHead) |
| `JOINT_DNF_MLC_best.pt` | 1 | 0.741 | ~400 МБ | Joint DN+Falcon, OptionC Head |
| `JDNF_SUPCON_best.pt` | 2 | 0.754 | ~400 МБ | +SupCon Loss |
| `JDNF_BNNECK_PROTO_best.pt` | 3 | 0.774 | ~400 МБ | +BNNeck+ProtoCon |
| `JDNFV_SUPCON_PROTO_R32_best_ema.pt` | 4 | 0.8431 | 864 МБ | +VeRI-Wild, r32, 2048d |
| `JDNFV_SUPCON_PROTO_R32_best_ema_fp16.pt` | 4 | 0.8431 | 432 МБ | То же в fp16 |
| `JDNFV_MASKED_FT_best_fp16.pt` | 5 | **0.839** | 306 МБ | Masked fine-tune (финальный) |

> **Примечание:** Финальный чекпоинт (Этап 5) показывает mAP 0.839 против 0.8431
> у EMA-весов Этапа 4. Это связано с тем, что masked fine-tune делался на
> маскированных кропах и чистых метках — метрики считались на оригинальных
> (немaskированных) изображениях для сопоставимости с Этапом 4.

## Расположение весов в инференсе

```
inference/weights/
├── dinov3-vitl16-bf16/        ← Backbone DINOv3 ViT-L/16 (bf16, 579 МБ)
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── JDNFV_MASKED_FT_best_fp16.pt  ← LoRA + Head + Blocks (306 МБ)
└── yolo11s-seg.pt              ← YOLO11s-seg для маскирования (20 МБ)
```

Итого ~905 МБ (в пределах лимита 2 ГБ, ТЗ п. 7).
