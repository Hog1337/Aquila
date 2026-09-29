# Фалькон ReID: сервис цифрового признака транспортного средства

Сервис сопоставляет снимки одного автомобиля с разных камер без государственного номера: строит эмбеддинг по кадру и BBox, ищет ближайших в галерее и умеет отказаться, если уверенного совпадения нет (ТЗ, разделы 3 и 4).

---

## Для организаторов (запуск и проверка)

### Требования
- Linux x86_64, Docker Engine 24+ и Docker Compose v2.20+
- NVIDIA GPU + Container Toolkit (для inference), ~4 ГБ VRAM
- Git LFS (для весов модели)

### 1. Клонирование и подготовка

```bash
git clone <репозиторий> aquila && cd aquila
git lfs pull           # веса модели ~0.9 ГБ (bf16 backbone + LoRA + YOLO)
```

### 2. Запуск стека

```bash
./run.sh               # сборка образов + запуск (нужен интернет при запуске без параметров)
```

| Флаг / Переменная | Описание |
|---|---|
| *(без флагов)* | Собрать образы и запустить стек |
| `--registry` | Использовать образы из GitLab Registry (без сборки) |
| `--local` | Использовать уже собранные локальные образы (без сборки, без пулла) |
| `--down` | Остановить стек (данные в volume сохраняются) |
| `SKIP_SERVICES="inference"` | Не поднимать указанные сервисы (через пробел). Без GPU: `SKIP_SERVICES="inference"` |

### 3. Запуск на закрытом тесте (одна команда)

Организаторы передают каталог с `images/` + CSV. Решение принимает его
и возвращает три файла сдачи.

```bash
# Всё в одной команде: очистка → сборка → импорт → экспорт → 3 файла
scripts/submit.sh --input /path/to/test --output /path/to/out --rerank --clear
```

Параметры:

| Флаг | Описание |
|------|----------|
| `--input DIR` | Каталог с `images/`, `test_gallery.csv`, `test_query.csv` |
| `--output DIR` | Куда сохранить результаты |
| `--threshold N` | Порог отказа (по умолчанию 0.87) |
| `--rerank` | Использовать Query Expansion (рекомендуется) |
| `--clear` | Очистить данные перед запуском |
| `--no-cleanup` | Не удалять данные после завершения |
| `--gallery-csv NAME` | Имя CSV галереи (по умолч. `test_gallery.csv`) |
| `--query-csv NAME` | Имя CSV запросов (по умолч. `test_query.csv`) |

**Результат:** в `--output` появляются `submission.csv`, `embeddings.npy`, `candidates.csv`.

Также доступен **веб-интерфейс** на `http://localhost:8081` — через него можно
вручную загрузить данные, выполнить поиск, посмотреть метрики и выбрать порог отказа.
Для пакетного прогона на закрытом тесте используйте CLI (одна команда выше).

Или напрямую, без обёртки:

```bash
docker compose run --rm \
  -v /path/to/test:/input:ro \
  -v /path/to/out:/output \
  backend python -m app.cli submit \
    --input /input --output /output --rerank
```

### 4. Проверка стека

```bash
scripts/smoke.sh        # все сервисы healthy?
curl -s localhost:8081  # UI доступен?
```

### 5. Очистка

```bash
docker compose down -v  # удалить все данные
# или через API:
# DELETE /api/v1/gallery  (очистить галерею без перезапуска)
# или CLI:
# docker compose run --rm backend python -m app.cli clear
```

---

## Архитектура

```
браузер ──> frontend (nginx :8080) ──/api──> backend (:8000) ──> inference (:8001, GPU)
                                               ├─> Qdrant      qdrant:6333   векторы
                                               ├─> SeaweedFS   seaweedfs:8333 кадры (S3)
                                               └─> PostgreSQL  postgres:5432  метаданные
```

| Компонент | Каталог | Роль |
|---|---|---|
| Frontend | `frontend/` | React + nginx, порт 8081 |
| Backend | `backend/` | FastAPI, порт 8000 |
| Inference | `inference/` | DINOv3 + LoRA + YOLO, порт 8001, GPU |
| Qdrant | `vector/` | Векторная БД, 2048d, Cosine |
| SeaweedFS | `storage/` | S3-совместимое хранилище кадров |
| PostgreSQL | `postgres/` | Метаданные, порог, история |

**Единая галерея:** все импортированные объекты — в одной коллекции без ролей.
Поисковые запросы обрабатываются на лету (не сохраняются в Qdrant).

## Артефакты сдачи (ТЗ п. 8)

Файлы для финальной проверки решения лежат в [`submission/`](submission/):

| Файл | Описание | Размер |
|------|----------|:------:|
| `submission.csv` | top-10 кандидатов для каждого запроса | ~400 КБ |
| `embeddings.npy` | матрица эмбеддингов (N×2048, float32) | ~15 МБ |
| `candidates.csv` | принятые кандидаты с confidence (отказ = нет строки) | ~200 КБ |

Эти файлы генерируются скриптом [`scripts/submit.sh`](scripts/submit.sh)
или командой `python -m app.cli submit --input <dir> --output <dir>`.
Подробнее о формате — в [`docs/methods.md`](docs/methods.md).

---

## Метрики (на val-сплите: query→gallery, честный протокол)

| Метрика | Значение |
|---------|:--------:|
| mAP@10 | **0.84** |
| Rank-1 — Rank-5 | 0.82 / 0.94 |
| PR-AUC | 0.96 |
| Порог отказа | **0.87** |

> Валидационные сплиты: [`training/splits/val_query.csv`](training/splits/val_query.csv)
> (1 261 запрос, 385 автомобилей) и [`training/splits/val_gallery.csv`](training/splits/val_gallery.csv)
> (1 119 объектов, 341 автомобиль). 44 open-set запроса (11 %) без пары в галерее.
> Метрики посчитаны `POST /metrics/runs` на реальном стеке.
> Обоснование порога отказа и анализ ошибок — в [`docs/methods.md`](docs/methods.md).
> Подробнее: [`docs/training/data.md`](docs/training/data.md)

---

## Документация

| Раздел | Описание |
|--------|----------|
| [`docs/training.md`](docs/training.md) + [`docs/training/`](docs/training/) | Обучение модели: датасеты, архитектура, этапы, воспроизведение (подразделы: data, architecture, stages, hyperparameters, reproduction, checkpoints, code-map) |
| [`docs/architecture.md`](docs/architecture.md) | Функциональная и компонентная архитектура |
| [`docs/backend.md`](docs/backend.md) | Backend API, модели данных, сценарии |
| [`docs/frontend.md`](docs/frontend.md) | Экраны, кнопки, контракт API |
| [`docs/methods.md`](docs/methods.md) | Методы обработки и алгоритмы |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | Внешние библиотеки и датасеты |

---


