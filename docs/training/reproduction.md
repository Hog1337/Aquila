# Воспроизведение обучения

> **Важно:** Полное воспроизведение требует внешних датасетов и GPU ≥24 ГБ
> VRAM. Код предоставляется «как есть» для прозрачности методологии.
> Финальные веса модели уже в `inference/weights/` (Git LFS) — обучение
> заново не требуется для запуска сервиса.

## Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|:-------:|:-------------:|
| GPU VRAM | 24 ГБ | 32+ ГБ (batch P=16, K=4) |
| CUDA | 12.1 | 12.2 |
| RAM | 32 ГБ | 64 ГБ |
| Диск | 50 ГБ (датасеты + веса) | 100 ГБ |
| ОС | Linux x86_64 | Linux x86_64 |
| Python | 3.10+ | 3.12 |

## Зависимости

```bash
pip install -r training/requirements.txt
```

Ключевые пакеты:
- `torch>=2.2.0` + `torchvision>=0.17.0` (с CUDA)
- `transformers>=4.44.0` (HuggingFace, для DINOv3)
- `numpy`, `Pillow`, `opencv-python`, `scikit-learn`

## Датасеты

Необходимо скачать и подготовить 3 датасета
(подробные инструкции: [`training/DATASETS.md`](../training/DATASETS.md)):

1. **Falcon ReID** — от организаторов
2. **DN-ReID** — публичный, CC BY-NC
3. **VeRI-Wild** — публичный, CC BY-NC

Перед запуском скриптов нужно исправить переменные в начале каждого `.py`-файла
(см. `DATASETS.md`, раздел «Сопоставление путей в скриптах»).

## Пошаговая инструкция

### 1. Подготовка сплита

```bash
cd training
python split.py
# Результат: splits/train_sub.csv, val_query.csv, val_gallery.csv
```

### 2. Чистка меток (опционально)

```bash
python clean_dataset.py
# Веб-интерфейс для ручного просмотра и исключения мусорных кадров
```

### 3. Этап 1: Joint DN-ReID + Falcon

```bash
python baseline/joint_dn_falcon.py
# Результат: out/JOINT_DNF_best.pt (mAP~0.741)
# Лог: /tmp/joint_dn_falcon.log
```

### 4. Этап 2: +SupCon

```bash
python baseline/joint_supcon.py
# Результат: out/JDNF_SUPCON_best.pt (mAP~0.754)
```

### 5. Этап 3: +BNNeck + ProtoCon

```bash
python baseline/joint_bnneck_proto.py
# Результат: out/JDNF_BNNECK_PROTO_best.pt (mAP~0.774)
```

### 6. Этап 4: +VeRI-Wild, r32, 2048d

```bash
python baseline/joint_triple.py
# Результат: out/JDNFV_SUPCON_PROTO_R32_best_ema.pt (mAP~0.8431)
# Время: ~3-4 часа на RTX 5000 Ada
```

### 7. Этап 5: Masked Fine-tune

```bash
# Предварительно: подготовить маскированные изображения
python baseline/make_carseg_sam3v2.py   # подготовка масок (опционально)

# Дообучение
python baseline/joint_masked_ft.py
# Результат: out/JDNFV_MASKED_FT_best_fp16.pt (mAP~0.839)
```

### 8. Валидация

```bash
python baseline/eval_multilayer.py --checkpoint out/JDNFV_MASKED_FT_best_fp16.pt --split val
```

## Проверка воспроизведения

После завершения обучения можно сверить метрики с эталонными:

| Метрика | Ожидаемое значение |
|---------|:------------------:|
| mAP@10 | 0.839 |
| Rank-1 | 0.819 |
| Rank-5 | 0.936 |
| F1 (τ=0.87, TNR≥0.9) | 0.704 |

## Известные особенности

1. **Пути в скриптах** захардкожены — нужно править вручную
2. **DN-ReID** требует конвертации из оригинального формата (`make_car_dataset.py`)
3. **VeRI-Wild** требует регистрации для скачивания
4. **YOLO11s-seg** скачивается автоматически через ultralytics (нужен интернет)
5. **DINOv3** скачивается из HuggingFace (нужен интернет при первом запуске)
6. **Python 3.14** имеет баг с `-arr` для numpy-массивов внутри функций (UNARY_NEGATIVE
   вычисляется in-place). При использовании Python ≥3.14 используйте `np.negative(arr)`
   вместо `-arr` в функциях
