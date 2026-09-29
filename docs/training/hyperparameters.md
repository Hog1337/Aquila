# Гиперпараметры

Сводная таблица гиперпараметров по всем этапам обучения.

## Единые для всех этапов

| Параметр | Значение |
|----------|----------|
| Оптимизатор | AdamW |
| Weight decay | 1e-4 |
| Batch strategy | P=16, K=4 (64 изображения/шаг) |
| Размер входа | 320×320 |
| ImageNet norm | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Precision | bf16 autocast (CUDA) |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |

## По этапам

| Параметр | Этап 1 | Этап 2 | Этап 3 | Этап 4 | Этап 5 |
|----------|:------:|:------:|:------:|:------:|:------:|
| **Скрипт** | `joint_dn_falcon.py` | `joint_supcon.py` | `joint_bnneck_proto.py` | `joint_triple.py` | `joint_masked_ft.py` |
| **Датасеты** | Falcon, DN | Falcon, DN | Falcon, DN | Falcon, DN, **VeRI-Wild** | Falcon (masked), DN |
| **LoRA rank** | 16 | 16 | 16 | **32** | 32 |
| **LoRA alpha** | 32 | 32 | 32 | **64** | 64 |
| **Proj dim** | 512 | 512 | 512 | **2048** | 2048 |
| **Голова** | OptionC (LN) | OptionC (LN) | HeadBNNeck (LN) | HeadBNNeck (no LN) | HeadBNNeck (no LN) |
| **Слои head** | 16,18,20,22,23,24 | те же | те же | **0,4,8,12,16,20** | те же |
| **Блоки unfreeze** | 16–23 | 16–23 | 16–23 | **14–23** | 14–23 |
| **Loss: CE** | ✅ | ✅ | ✅ (через BN) | ✅ (через BN) | ✅ (через BN) |
| **Loss: SupCon** | — | ✅ λ=0.3 | ✅ λ=0.3 | ✅ λ=0.3 | ✅ λ=0.3 |
| **Loss: ProtoCon** | — | — | ✅ λ=0.15 | ✅ λ=0.15 | ✅ λ=0.15 |
| **SupCon T** | — | 0.07 | 0.07 | 0.07 | 0.07 |
| **ProtoBank momentum** | — | — | 0.99 | 0.99 | 0.99 |
| **LR LoRA** | 3e-5 | 3e-5 | 3e-5 | 3e-5 | **5e-6** |
| **LR Head** | 3e-4 | 3e-4 | 3e-4 | 3e-4 | **1.5e-5** |
| **LR blocks** | 2.5e-6…2e-5 | те же | те же | 5e-6…2e-5 | 2.5e-6…1e-5 |
| **Scheduler** | warmup 3ep + Cosine | те же | те же | те же | Cosine (без warmup) |
| **Эпох** | 30 | 40 | 40 | 30 | 10 |
| **Шагов/эпоха** | ~460 | ~600 | ~600 | 600 | 400 |
| **Аугментации** | — | — | — | HFlip, ColorJitter | HFlip, ColorJitter |
| **EMA decay** | — | — | — | **0.999** | 0.999 |
| **Микс датасетов** | 30/70 | 30/70 | 30/70 | **25/35/40** | **50/50** |
| **mAP@10** | 0.741 | 0.754 | 0.774 | **0.8431 (EMA)** | **0.839** |

## Расписание LR (Cosine decay)

```
if epoch < 3:  LR = 0.1 + 0.9 * (epoch + 1) / 3                        # linear warmup
else:          LR = 0.5 * (1 + cos(π * (epoch - 3) / (EPOCHS - 3 - 1))) # cosine decay
```

Базовый LR умножается на коэффициент для каждой группы параметров.

## Аугментации (Этапы 4–5)

```python
transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Оборудование

| Параметр | Разработка | Стенд организаторов |
|----------|:----------:|:-------------------:|
| GPU | RTX 5000 Ada (32GB) / Blackwell RTX PRO 6000 (98GB) | RTX A5000 (24GB) |
| CUDA | 12.2 | 12.2 |
| CPU | — | 2× Xeon Gold 6338 (128 логических) |
| RAM | — | ~256 ГБ |
