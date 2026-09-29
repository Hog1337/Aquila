# Backend и inference: как это реализовано

Документ описывает то, что лежит в `backend/` и `inference/`. Прежняя версия документа была проектом («что нужно добавить»); сервисы к настоящему моменту написаны, поэтому текст заменён описанием реальной реализации. Архитектура целиком: [`architecture.md`](architecture.md), методы и метрики: [`methods.md`](methods.md), фронтенд: [`frontend.md`](frontend.md).

**Основа сверки.** Код на коммите `60fc04e` (26.09.2026), рабочее дерево без изменений. Почти всё ниже получено **чтением кода**; стек целиком (с моделью на GPU) при сверке не запускался. Запускалось только два изолированных опыта: миграции PostgreSQL на временном контейнере и порядок маршрутов FastAPI на тестовом приложении с тем же порядком объявления. Что именно проверено, указано в разделе 8. Если факт нигде не подтверждён, он помечен «неизвестно».

| Пометка | Смысл |
|---|---|
| Реализовано | есть в коде и соответствует описанию |
| Расхождение | код делает не то, что описано в других документах или требует ТЗ |
| Неизвестно | по репозиторию установить нельзя |

## 1. Схема

```
[Браузер] ──HTTP:8080──> [frontend: nginx + React]
                              │  /api/*, /docs, /redoc, /openapi.json (прокси)
                              v
                         [backend  FastAPI :8000]  (CPU)
                              │  POST /internal/embed, /internal/cls-attention
                              v
                         [inference  FastAPI + PyTorch :8001]  (GPU, runtime nvidia)
        backend ─────────┬───────────────┬───────────────┐
                         v               v               v
                   [Qdrant :6333]  [SeaweedFS :8333]  [PostgreSQL :5432]
                   векторы 512d    S3-бакет gallery   схема reid
```

Все сервисы в docker-сети `falcon-reid`. Inference вызывается только backend-ом; данные из хранилищ он не читает (кроме двух эндпоинтов из раздела 4, которым нужен каталог `/data/images`, в compose не смонтированный).

## 2. Два сервиса

| | `backend` | `inference` |
|---|---|---|
| Каталог | `backend/` | `inference/` |
| Образ | `falcon-reid-backend`, база `python:3.12-slim` | `falcon-reid-inference`, база `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| Порт | `8000` (`BACKEND_PORT`) | `8001` (`INFERENCE_PORT`) |
| Публикация на хост | `"${BACKEND_PORT:-8000}:8000"`, **на всех интерфейсах** (переменной `*_BIND`, как у БД, нет) | `"${INFERENCE_PORT:-8001}:8001"`, **на всех интерфейсах** |
| Ресурсы | `mem_limit` 512 МБ (`BACKEND_MEM_LIMIT`), `pids_limit` 256 | `mem_limit` 4 ГБ (`INFERENCE_MEM_LIMIT`), `pids_limit` 256, `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all` |
| Изоляция | `read_only`, tmpfs `/tmp`, `cap_drop: [ALL]`, `no-new-privileges` | то же |
| Healthcheck | `GET /api/v1/status` возвращает 200 (это проверяет и связь с PostgreSQL) | `GET /health` содержит `"status":"ok"`, то есть модель и чекпоинт загружены; `start_period` 180 с |
| Зависимости compose | ждёт healthy у `postgres`, `qdrant`, `seaweedfs`; **от `inference` не зависит** | нет |
| Веса | нет | не в образе: монтируются томами `inference/weights/` (`dinov3-vitl16-bf16/` + `JDNFV_MASKED_FT_best_fp16.pt` + `yolo11s-seg.pt`) только на чтение |
| Зависимости Python | `requirements.txt`: fastapi, uvicorn, python-multipart, asyncpg, qdrant-client, boto3, httpx, numpy, Pillow (только нижние границы версий) | `requirements.txt`: fastapi, uvicorn, python-multipart, torch, torchvision, transformers, safetensors, numpy, Pillow (только нижние границы) |

Почему сервисы раздельны (как заложено в compose): GPU нужен только inference, backend работает на CPU; веса и torch в отдельном образе; модель можно менять без пересборки backend.

**Аутентификации нет** ни у backend, ни у inference. В backend включён CORS `allow_origins=["*"]` с `allow_credentials=True`. Наружу через nginx проксируются `/api/`, `/docs`, `/redoc`, `/openapi.json`; inference через nginx недоступен, но порт `8001` опубликован на хосте (см. таблицу).

## 3. Настройка (переменные окружения)

### backend (`backend/app/config.py`, значения задаёт `backend/docker-compose.yml`)

| Переменная | По умолчанию в коде | В compose |
|---|---|---|
| `POSTGRES_DSN` | `postgresql://reidbackend:…@localhost:5432/reid` | `postgresql://${POSTGRES_APP_USER}:${POSTGRES_APP_PASSWORD}@postgres:5432/reid` |
| `QDRANT_HOST`, `QDRANT_PORT` | `localhost`, `6333` | `qdrant`, `6333` |
| `QDRANT_API_KEY`, `QDRANT_COLLECTION` | пусто, `gallery` | из `vector/.env` |
| `S3_ENDPOINT` | `http://localhost:8333` | `http://seaweedfs:8333` |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` | заглушки, `gallery` | `S3_APP_ACCESS_KEY`, `S3_APP_SECRET_KEY`, `S3_BUCKET` из `storage/.env` |
| `INFERENCE_URL` | `http://localhost:8001` | `http://inference:8001` |
| `MODEL_VERSION` | `JDNF_BNNECK_PROTO2` | не задаётся |
| `EMBEDDING_DIM` | `512` | не задаётся |
| `DEFAULT_THRESHOLD` | `0.87` | `${DEFAULT_THRESHOLD:-0.87}` из корневого `.env` (используется, только если `current_threshold` пуст) |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | то же |

`MODEL_VERSION` в backend это просто строка из окружения: она пишется в `search_queries.model_version` и `metric_runs.model_version` и отдаётся в `/status`. С реально загруженным в inference чекпоинтом она не сверяется.

Секреты для compose собирает `run.sh`: он пересоздаёт `backend/.env` из `postgres/.env`, `vector/.env`, `storage/.env` при каждом запуске (файл в git не попадает).

### inference (`inference/app/config.py`, значения задаёт `inference/docker-compose.yml`)

| Переменная | По умолчанию в коде | В compose |
|---|---|---|
| `DINO_DIR` | `<inference>/weights/dinov3-vitl16` (авто-fallback на `dinov3-vitl16-bf16/`) | `/app/weights/dinov3-vitl16-bf16` |
| `CKPT_PATH` | `<inference>/weights/JDNFV_MASKED_FT_best_fp16.pt` | `/app/weights/JDNFV_MASKED_FT_best_fp16.pt` |
| `MODEL_VERSION` | `JDNFV_MASKED_FT` | `JDNFV_MASKED_FT` |
| `DEVICE` | `auto` (cuda, если доступна, иначе cpu) | `auto` |
| `SIZE` | `auto` (для этого чекпоинта 320) | `320` |
| `HOST`, `PORT` | `0.0.0.0`, `8001` | то же |

Код умеет работать на CPU, но compose-файл требует `runtime: nvidia`; на машине без NVIDIA Container Toolkit сервис не стартует, поэтому `run.sh` предусматривает `SKIP_SERVICES="inference"`. Запуск на CPU без правки compose-файла не предусмотрен; работоспособность и скорость на CPU не проверялись.

## 4. Inference

### Эндпоинты (`inference/app/main.py`)

Префикс `/internal` означает «только для backend», но защиты нет.

| Метод и путь | Что делает | Использует backend |
|---|---|---|
| `GET /health` | `{status: ok\|degraded, device, model_error, model_version}`; отвечает 200 и без модели | нет (healthcheck compose) |
| `GET /internal/status` | версия, размерность 512, устройство, `bf16`/`fp32`, загружена ли модель | нет |
| `POST /internal/embed` | multipart `image`, `x, y, w, h` → `{embedding: [512], elapsed_ms}`; вырезает рамку из кадра и считает эмбеддинг | **да**: поиск, добавление объекта, импорт, метрики |
| `POST /internal/embed-batch` | несколько изображений (поле `images`) → эмбеддинги, батч до 64 | нет |
| `POST /internal/embed-by-key` | `image_id` + рамка; кадр читается из `/data/images/<image_id>.jpg` внутри контейнера | нет |
| `POST /internal/embed-by-key-batch` | то же для списка `keys` (JSON), кадры читаются параллельно (16 потоков) | нет (нужен `scripts/load_dataset.py`) |
| `POST /internal/attention` | эмбеддинг и веса внимания пулинга по 256 патчам (сетка 16×16) | нет |
| `POST /internal/compare` | «Grad-ATTN»: два кропа, косинусное сходство, карты релевантности по градиенту, оверлеи и вклад четырёх квадрантов | нет |
| `POST /internal/cls-attention` | один кроп: CLS self-attention последнего слоя DINOv3, усреднённая по 16 головам; оверлей (data URL, JPEG) и вклад четырёх квадрантов | **да**: окно «Grad-CAM» |

Каталог `/data/images` в `inference/docker-compose.yml` не смонтирован, а корневая ФС контейнера доступна только на чтение, поэтому `embed-by-key*` отвечают 404, пока каталог не подмонтирован отдельно. Границы рамки inference не проверяет; проверку делает только `POST /api/v1/queries` в backend (раздел 5).

Без модели (`model is None`) эндпоинты эмбеддинга отвечают 503, `/health` даёт `degraded` и поле `model_error` (чекпоинт или каталог DINOv3 не найдены, либо ошибка загрузки). Чекпоинт при этом загружается **при старте**; в логе сервис печатает `[inference] model loaded in … s`.

### Модель

Что видно из кода `inference/app/reid_embed.py`; **обучение, данные и гиперпараметры в репозитории отсутствуют** (см. раздел 10).

| Элемент | Значение |
|---|---|
| Backbone | DINOv3 ViT-L/16: `hidden_size` 1024, 24 слоя, 16 голов, 4 регистровых токена, патч 16 (`weights/dinov3-vitl16/config.json`), загрузка через `transformers.AutoModel` |
| Адаптация | LoRA r = 16, alpha = 32 (масштаб 2,0) на `q_proj, k_proj, v_proj, o_proj`: обёрнуто ровно 96 модулей, иначе загрузка прерывается ошибкой |
| Дообученные блоки | слои 16–23 загружаются из чекпоинта целиком (`blocks`); при другом наборе выдаётся предупреждение |
| Голова | `HeadBNNeck`: 6 обучаемых запросов attention-pooling по промежуточным слоям (индексы с нуля: 15, 17, 19, 21, 22, 23), к каждому выходу применяется финальный LayerNorm backbone, затем LayerNorm головы; конкатенация 6 × 1024 → линейный слой 512 → BatchNorm1d → L2-нормировка |
| Чекпоинт | `JDNF_BNNECK_PROTO2_best.pt` (по комментарию в коде: «joint BNNeck + SupCon + ProtoCon»; обучающего скрипта `joint_bnneck_proto.py`, на который ссылается код, в репозитории нет) |
| Вход | вырезка по BBox → `AutoImageProcessor` DINOv3 с `size=(320, 320)`; среднее/СКО ImageNet (`preprocessor_config.json`); центрального кропа нет |
| Выход | 512 × float32, L2-нормированный |
| Точность вычислений | на CUDA bf16 через autocast, на CPU fp32; веса хранятся в fp32 |
| Размер весов | `model.safetensors` 1 212 559 808 Б + чекпоинт 438 302 047 Б = 1 650 861 855 Б (около 1,65 ГБ); ограничение ТЗ п.7: 2 ГБ |

Ветка для другого типа чекпоинта (`FalconFT`, `AttnPoolHead`, 256 px) в коде есть, но в ней используется `T.Compose` без импорта `torchvision.transforms` на уровне модуля (в `compare_with_relevance` импорт локальный), поэтому при загрузке такого чекпоинта ожидается `NameError`. Для нужного чекпоинта (`BNNeck`) ветка не задействуется. Не проверялось запуском.

Как изменение размера соотносится с пропорциями рамки, по коду однозначно не следует: размер задаётся кортежем `(320, 320)` процессору DINOv3; сжатие по осям без сохранения пропорций вероятно, но запуском не подтверждено. Прежняя версия документа писала «256 × 256 stretch»: это относилось к другой ветке (AttnPool), для текущего чекпоинта неверно.

Время инференса, пропускная способность (FPS) и требования к видеопамяти: **не измерены**.

## 5. Backend: API

Префикс `/api/v1`. OpenAPI формирует FastAPI автоматически: Swagger по `/docs`, схема по `/openapi.json` (через nginx фронтенда и напрямую на `:8000`). Ошибки: HTTP-код и `{"detail": "текст"}`.

| Метод и путь | Параметры (по умолчанию) | Что делает |
|---|---|---|
| `GET /status` | нет | `{model_version, embedding_dim, gallery: {objects, images, vehicles}}`; счётчики из PostgreSQL, версия и размерность из конфигурации backend |
| `GET /threshold` | нет | действующий порог: строка `reid.current_threshold`; если таблица пуста, `DEFAULT_THRESHOLD` |
| `PUT /threshold` | тело `{value ∈ [−1, 1], reason}` | дописывает строку в `threshold_history` (`changed_by` всегда `backend`) |
| `POST /queries` | multipart: `image`, `x, y, w, h`, `top_n` (10), `threshold` (действующий), `rerank` (false) | поиск, раздел 6.1 |
| `GET /queries` | `refused` (только значение `true` включает фильтр), `limit` (20), `offset` (0) | журнал запросов: `total`, `summary`, `items`; новые сверху |
| `GET /queries/{id}` | | сохранённый запрос с кандидатами; **неполный** (раздел 8, №6) |
| `GET /queries/{id}/image` | | кадр запроса из S3 (`image/jpeg` независимо от исходного формата) |
| `GET /queries/{id}/crop` | `size` | JPEG вырезки по BBox; `size` ограничивает большую сторону |
| `GET /queries/{id}/candidates.csv` | `threshold` (порог запроса) | CSV `rank,image_id,vehicle_id,score,accepted`, **все** top-N с флагом (не только принятые) |
| `GET /queries/{id}/candidates/{rank}/gradcam` | | окно «Grad-CAM», раздел 6.4 |
| `GET /queries/history.csv` | `refused` | CSV журнала (не более 10 000 строк) |
| `GET /objects/{id}/crop` | `size` | JPEG вырезки объекта галереи (кадр `images/<image_id>.jpg` из S3 + BBox из PostgreSQL) |
| `GET /gallery/objects` | `role`, `vehicle_id` (подстрока, без учёта регистра), `limit` (24), `offset` (0) | список объектов, новые сверху по `created_at` |
| `POST /gallery/objects` | multipart: `image`, `x, y, w, h`, `vehicle_id` | добавить один объект (`role = gallery`, `batch_id = NULL`), раздел 6.2; отвечает 201 |
| `POST /gallery/import-sessions` | multipart: `annotations` (CSV), `role` (обязателен: `gallery`/`query`/`val_gallery`/`val_query`) | шаг 1 импорта: проверка CSV, ответ 201 `{id, source_name, role, total, image_ids}`, раздел 6.2 |
| `POST /gallery/import-sessions/{id}/images` | multipart: `image_id`, `image` (один кадр) | шаг 2: кадр пишется в бакет как `images/{image_id}.jpg`; ответ 204, повтор перезаписывает |
| `POST /gallery/import-sessions/{id}/start` | — | шаг 3: создаёт фоновое задание импорта; отвечает 202 `ImportJob` |
| `GET /gallery/imports` | `limit` (**1**), `role`, `status` | последние задания импорта; `role`+`status=done` — список источников для селекторов на «Экспорте» |
| `GET /gallery/imports/{id}` | | состояние задания |
| `DELETE /gallery/imports/{id}` | | отмена: `task.cancel()` и статус `cancelled` |
| `POST /metrics/runs` | — (тела нет) | фоновая оценка на `val_query`/`val_gallery`, раздел 6.5; отвечает 202 |
| `GET /metrics/runs/{id}` | | состояние задания оценки (в `run` подставляется **последний** запуск, а не именно этого задания) |
| `GET /metrics/runs/latest` | — | последний запуск; 404 «Оценка ещё не запускалась» |
| `POST /exports` | тело `{threshold, query_batch_id}` | фоновый экспорт партии запросов против текущей галереи (role='gallery'), раздел 6.6; отвечает 202 |
| `GET /exports` | `limit` (**1**) | последние задания экспорта |
| `GET /exports/{id}` | | состояние задания |
| `GET /exports/{id}/files/{name}` | | файл из S3 `exports/<id>/<name>` (`attachment`) |

Пути `/queries/history.csv` и `/metrics/runs/latest` объявлены **после** маршрутов с параметром (`/queries/{query_id}`, `/metrics/runs/{job_id}`): см. раздел 8, №1.

Все схемы ответов лежат в `backend/app/schemas/models.py` (Pydantic); фронтенд ожидает формат из [`frontend.md`](frontend.md#5-контракт-api-по-коду-backend).

## 6. Сценарии, как они реализованы

### 6.1 Поиск (`POST /queries`, `routers/queries.py`)

1. **Валидация.** Тип файла строго `image/jpeg` или `image/png` (по заголовку Content-Type, не по содержимому), размер не более 50 МБ, рамка не выходит за правый и нижний край кадра. **Не проверяется:** `x, y ≥ 0`, `w, h > 0`, границы `top_n` и `threshold`. Нарушение даёт 422.
2. Кадр в исходных байтах сохраняется в S3 как `queries/<uuid>.jpg` (расширение фиксированное, даже для PNG).
3. Порог берётся из запроса, иначе из `current_threshold`.
4. Inference: `POST /internal/embed` (синхронный `httpx.Client`, таймаут 60 с) → эмбеддинг 512 и время. Ошибка inference (в том числе 503 при незагруженной модели) **не переводится** в понятный ответ: пользователь получает 500. Кадр запроса к этому моменту уже лежит в S3.
5. Поиск в Qdrant (`searcher.search`): `query_points(limit=top_n, with_payload=True, query_filter={role: "gallery"})` **без** `score_threshold`, **без** группировки по `vehicle_id` и без `exact`, то есть обычный HNSW среди объектов с ролью `gallery` (запросы и валидационные партии в этот поиск не попадают). Возвращаются все top-N, включая ниже порога.
6. `accepted = score ≥ threshold`, `best_score` — максимум `score`, `accepted_count` — число принятых. Отказ = `accepted_count = 0` (в БД столбец `refused` вычисляется сам).
7. Запись в PostgreSQL: `search_queries` (bbox, порог, `top_n`, `best_score`, `accepted_count`, `elapsed_ms`, версия модели, эмбеддинг `real[512]`, ключ кадра) и `search_results` (все top-N). Ответ: `SearchResult` со временем инференса и поиска и размером галереи (`searcher.count()`, тоже отфильтрован по `role='gallery'`).
8. `elapsed_ms` считается от начала вызова inference до конца сборки кандидатов (запись в БД в него не входит).

**Re-ranking (`rerank=true`, `searcher.search_rerank`).** Первый проход берёт `max(top_n × 5, 50)` точек с их векторами, запрос усредняется с векторами первых пяти результатов (query expansion), результат L2-нормируется, второй поиск выдаёт top-N. Комментарий в коде обещает «+0,5–1,5 mAP»: в репозитории этому подтверждения нет, замера нет. Метка «QE re-ranking» показывается в интерфейсе. Порог `accepted` при этом сравнивается с `score` **второго** поиска (по усреднённому вектору): шкала сходства сдвигается, и порог, подобранный без QE, для него не калибровался.

**Что интерфейс получает вместо «один кандидат на автомобиль».** Группировки по `vehicle_id` нет, поэтому несколько строк top-N могут принадлежать одному автомобилю (`methods.md`, раздел 3 описывал обратное).

### 6.2 Добавление объектов в галерею

**Один объект (`POST /gallery/objects`).** `image_id = sha256(файл)[:16]`, `id = uuid5(NAMESPACE_DNS, "image_id:x:y:w:h")`; inference → эмбеддинг; кадр в S3 как `images/<image_id>.jpg`; запись в `images` и `gallery_objects` (`role = gallery`, `batch_id = NULL` — ручное добавление не привязано ни к одной партии импорта); точка в Qdrant с payload `{image_id, bbox: {x,y,w,h}, vehicle_id, role, batch_id}`. Рамка **не проверяется** на выход за кадр и на знак.

**Импорт (`/gallery/import-sessions`).** Импорт идёт тремя запросами, ничего крупного в памяти backend не остаётся (прежний `POST /gallery/imports` с ZIP удалён: он читал архив целиком в память и на архиве в несколько ГБ убивал контейнер, nginx отвечал 502). (1) `POST /gallery/import-sessions`: CSV (`image_id, x, y, w, h[, vehicle_id]`, UTF-8, до 64 МБ) проверяется целиком (колонки, целые `x, y, w, h`, допустимый `image_id`) и сохраняется в бакет как `imports/{id}/annotations.csv` вместе с `meta.json`; ответ содержит список нужных кадров. `role` обязателен (`gallery`/`query`/`val_gallery`/`val_query`) и передаётся явно вызывающим — угадывания по имени файла больше нет (раньше подстрока `train`/`val`/`test` в имени CSV задавала `split`, это была часть той же ML-терминологии, которую убрали). (2) `POST …/{id}/images`: браузер шлёт кадры по одному (до 50 МБ, проверяется, что это изображение) в несколько потоков; кадр сразу пишется в `images/{image_id}.jpg`. (3) `POST …/{id}/start`: создаётся задание в `import_jobs` (сама запись `import_jobs` и есть «партия» — её `id` становится `batch_id` всех объектов) и выполняется как `asyncio`-задача в самом процессе backend через общую функцию `services/ingest.ingest_row` (её же использует `app.cli submit`, раздел 6.7): блокирующие вызовы S3, inference и Qdrant вынесены в потоки, поэтому опрос задания отвечает; **перезапуск backend обрывает импорт**, а запись в БД остаётся `running`. Для каждой строки: кадр читается из бакета (подряд идущие строки одного кадра читают его один раз), inference по одному объекту, запись в PostgreSQL (с шириной и высотой кадра, `role`, `batch_id`, `row_no`) и Qdrant, обновление прогресса. Чтение кадра из бакета при обрыве или недоступности хранилища повторяется с нарастающей паузой (до ~60 с, хранилище после рестарта поднимается ~30 с); если не отвечает и после этого, задание завершается со статусом `failed`. Ошибка одной строки (битый кадр, рамка за пределами кадра) не останавливает импорт: строка пропускается и попадает в `error` («Строк с ошибкой пропущено: N»), 20 таких строк подряд считаются поломкой inference и останавливают задание. Оборванное или отменённое задание можно запустить заново тем же `POST …/{id}/start` (кадры уже в бакете; уже обработанные строки перезаписываются идемпотентно). **Строка без загруженного кадра пропускается**; их число попадает в `error` завершённого задания («Строк без кадра пропущено: N»). Сессии в PostgreSQL не хранятся: брошенная сессия оставляет только CSV и кадры в бакете. Порядок строки CSV сохраняется как `row_no` — по нему `/exports` и `app.cli submit` строят строки `embeddings.npy`.

Оба сценария вызывают `db.save_image(...)` с шириной и высотой 0, а в схеме `images` стоит `CHECK (width > 0)` и `CHECK (height > 0)`: см. раздел 8, №2.

### 6.3 Ключи и идентификаторы

| | Backend (импорт, добавление) | Скрипты `vector/`, `postgres/` | Скрипт `storage/` | Черновой `scripts/load_dataset.py` |
|---|---|---|---|---|
| Пространство имён uuid5 | `uuid.NAMESPACE_DNS` | `6f0f8a3e-5a0e-4c53-9d0a-4f414c4b4c4f` | нет | `uuid.NAMESPACE_DNS` |
| Строка для uuid5 | `image_id:x:y:w:h` | `image_id:x:y:w:h` | | `image_id:x:y:w:h` |
| Ключ кадра в S3 | `images/<image_id>.jpg`, `queries/<uuid>.jpg` | не используют S3 | `<имя файла>` (без префикса `images/`) | не пишет в S3 (передаёт `img_bytes=None`) |
| `bbox` в payload Qdrant | объект `{x, y, w, h}` | список `[x, y, w, h]` | | объект |

Следствия описаны в разделе 8 (№3 и №4).

### 6.4 «Grad-CAM» (`GET /queries/{id}/candidates/{rank}/gradcam`)

Название сохранено ради ТЗ п.10 и надписи кнопки, но **по существу это карта внимания, а не Grad-CAM**. Backend забирает из S3 кадр запроса и кадр кандидата, вырезает обе рамки и **дважды** вызывает `POST /internal/cls-attention` (синхронно, таймаут 30 с). Ответ: `query_overlay_url` и `candidate_overlay_url` — data URL JPEG (не PNG с прозрачностью: оверлей уже смешан с исходной вырезкой без прозрачности, 60 % цвета и 40 % изображения), `regions` — восемь строк (по четыре квадранта для запроса и кандидата), `note`. Карта строится по каждому изображению **отдельно** и не отражает вклад в сходство пары; заголовок окна в интерфейсе («CLS Attention · область внимания модели») это отражает, подпись «Вклад областей в сходство» под ним не совсем точна. Метод `/internal/compare` (градиентная релевантность именно для пары) в inference реализован, но backend его не вызывает.

Ответ 404, если у кандидата `object_id = null` или кадр не найден.

### 6.5 Оценка метрик (`POST /metrics/runs`, `routers/metrics.py`)

Что делает код (определения из [`methods.md`](methods.md#5-метрики-качества) — реализованы с одной оговоркой, см. ниже и раздел 8, №5):

1. Запросы: все объекты `gallery_objects` с ролью `val_query` (`db.get_objects_by_role("val_query")`); галерея для них отдельная — объекты с ролью `val_gallery` (`db.get_objects_by_role("val_gallery")`). Раньше запрос и галерея были одним и тем же `split='val'` (self-search, см. историю бага в разделе 6.6) — теперь это те же две роли, что у продуктового `query`/`gallery`, просто для собственной калибровки, а не для сдачи ТЗ п.8.
2. Эмбеддинг запроса не пересчитывается через inference: он уже лежит в Qdrant с момента импорта, читается батчами по 500 (`searcher.retrieve_vectors`).
3. Поиск: `searcher.search_within_role(embedding, top_n=10, role='val_gallery')` — фильтр Qdrant по `role`; исключать сам запрос не нужно, `val_query` и `val_gallery` — разные объекты по построению; `exact` не используется, поиск по HNSW.
4. Число валидных позитивов на запрос (`n_pos`) считается не по результатам поиска, а напрямую по счётчику `vehicle_id` среди объектов `val_gallery` — так же, как `valid_positives()` в эталонном `evaluate.py`. Если `n_pos = 0`, запрос open-set: не входит в mAP/Rank-1/Rank-5 (считается только для F1/TNR), и это не требует отдельного «конструирования» отрицательных запросов.
5. mAP@10: среднее AP по top-10 (после исключения самого запроса), знаменатель `min(n_pos, 10)`; Rank-1, Rank-5 — доли запросов с верным кандидатом в первой позиции/пятёрке, считаются только среди запросов с `n_pos > 0`.
6. Кривая по порогам 0,30…0,95 шагом 0,01: запрос принят, если лучший score из top-10 ≥ порога; `TP` — принят, есть валидный позитив (`n_pos>0`) и лучший кандидат верный; `FP` — принят, но не то (включая ложное срабатывание на open-set запросе, что увеличивает `FP_openset`); `FN` — не принят, но позитив есть; `TN` — не принят, позитива нет. `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `f1`, `tnr = TN/(TN+FP_openset)` (только по open-set запросам, как в `evaluate.py`, а не по всем `FP`).
7. Порог запуска (`metric_runs.threshold`, F1, TNR) — порог с максимумом F1 при `TNR ≥ 0,9`; если такого нет, остаётся 0,7 с нулевыми F1 и TNR.
8. `pr_auc` — реальная площадь под кривой Precision-Recall по решению «у запроса есть совпадение» (функция `_pr_auc`, независима от порога), не `F1 × TNR`.

**Чего код не может дать, как эталонный скрипт.** Junk-фильтр `evaluate.py` — «тот же `vehicle_id` И та же `camera_id`»; camera_id в схеме backend нет вообще (ТЗ п.5.2, датасет), поэтому исключается только сам объект-запрос (а он и так не входит в `val_gallery` — роли не пересекаются), но несколько снимков одного автомобиля с одной и той же точки съёмки внутри `val_gallery` всё ещё считаются валидными позитивами друг для друга — это ограничение данных, не код.

Тело запроса `POST /metrics/runs` пустое: выбора выборки больше нет, режим оценки один — `val_query` против `val_gallery`. Прогресс до 80 % идёт по запросам, остальное — расчёт кривой. Ошибка (`Нет данных для оценки: не загружены val_query и val_gallery`, если одна из ролей пуста, или векторов части объектов нет в Qdrant) сохраняется в задание как `failed`. Проверено на настоящем стеке (Postgres + Qdrant + inference): импорт `val_gallery.csv`/`val_query.csv` через `/gallery/import-sessions` (роли `val_gallery`/`val_query`) и `POST /metrics/runs` на 10 объектах датасета отработали корректно (mAP@10 = 1,0 — маленькая выборка, не показатель качества модели, важен сам факт, что пайплайн query→gallery считается без ошибок).

### 6.6 Экспорт (`POST /exports`, `routers/exports.py`)

**История бага.** До перехода на `role`/`batch_id` у `gallery_objects` не было понятия «запрос»/«галерея» — только один `split='test'`, и каждый объект по очереди искался **среди того же `split='test'`** (сам объект исключался фильтром Qdrant `must_not HasId`, а не разделением на две выборки). Живой прогон на `data/val_query.csv` + `data/val_gallery.csv` (оба загружены как единый `split='test'`, 2380 объектов) это вскрыл: эталонный `evaluate.py` при строгом query/gallery-разделении показал mAP@10 = 0,741, Rank-1 = 0,790, но предупредил `12620 gallery_id из submission.csv нет в галерее — отброшены` — top-10 был часто заполнен другими объектами-запросами, которых в настоящей галерее нет. `candidates.csv` пострадал сильнее: `Precision` падала до 0,04, потому что кандидатов с ролью «запрос» принимали как «совпадение», хотя по протоколу организаторов такого совпадения не существует.

**Текущая реализация.** Галерея у экспорта не выбирается отдельно — это тот же единый пул `role='gallery'`, что и у обычного поиска (заполняется на странице «Галерея», как и раньше). Источник на странице «Экспорт» — только партия запросов: тело запроса `{threshold, query_batch_id}`, где `query_batch_id` — `import_jobs.id` завершённого импорта с ролью `query`. Загрузка новой партии запросов происходит прямо на странице «Экспорт» (тот же трёхшаговый импорт, что и «Галерея», но с ролью `query` — раздел 5.6 `frontend.md`); список для выбора среди уже загруженных отдаёт `GET /gallery/imports?role=query&status=done`. Источник данных — `db.get_objects_by_batch(query_batch_id)` (запросы) и `db.get_objects_by_role("gallery")` (вся текущая галерея), оба упорядочены по `row_no` (номер строки исходного CSV; `row_no NULLS LAST`). Каждый объект query-партии по очереди берётся запросом, галерея поиска — вся галерея (`searcher.search_within_role(embedding, top_n, "gallery")`, фильтр Qdrant по `role`); исключать сам запрос не нужно — запросы и галерея не пересекаются по построению.

| Файл | Что записывается |
|---|---|
| `embeddings.npy` | матрица float32: сначала все объекты query-партии, затем вся текущая галерея, обе группы в порядке `row_no` (порядок исходных CSV) |
| `submission.csv` | без заголовка: `image_id` запроса и через запятую top-10 `image_id` из галереи по убыванию косинусного сходства; пустые ячейки, если кандидатов меньше 10 |
| `candidates.csv` | заголовок `query_id,gallery_id,confidence`; строка на каждого кандидата со score ≥ порога задания (`export_jobs.threshold`); если ни один кандидат не прошёл порог — для этого запроса строк нет (кодирует отказ) |

Реализация вынесена в `services/export.build_export_artifacts` — использует и `POST /exports`, и `python -m app.cli submit` (раздел 6.7), чтобы веб-экспорт и пакетный прогон на закрытом тесте не расходились в логике. Прогресс задания: 5 % после чтения списка, далее до 95 % по мере обхода запросов.

Проверено на настоящем стеке: `POST /exports` с реальными `test_query.csv`/`test_gallery.csv` (по 3–8 строк) дал корректный `submission.csv`/`candidates.csv`/`embeddings.npy` формы `(n_query + n_gallery, 512)`; та же проверка через `docker compose run backend python -m app.cli submit` и через `scripts/submit.sh` целиком — раздел 6.7.

### 6.7 Пакетный прогон (`python -m app.cli submit`, `scripts/submit.sh`)

Реализует сценарий из Q&A организаторов (вопрос №40: «пачкой» — каталог кадров и CSV целиком, три файла на выходе, без запросов по одному) одной командой, без ручных шагов через UI.

`scripts/submit.sh --input <dir> --output <dir> [--threshold T] [--gallery-csv NAME] [--query-csv NAME]`: поднимает стек через `./run.sh` (тот уже правильно разводит по времени хранилища, разовые init-сервисы и приложение — см. корневой `README.md`, «Сборка и запуск» — вместо голого `docker compose up -d --wait`, который иначе принял бы штатное завершение `postgres-init`/`qdrant-init` кодом 0 за ошибку), монтирует `--input` в контейнер `backend` как `/input:ro`, `--output` как `/output`, выполняет `python -m app.cli submit --input /input --output /output`, затем **безусловно** гасит стек вместе с volumes (`docker compose down -v`) — Postgres/Qdrant/SeaweedFS не должны копить данные между прогонами оценки; если стек уже был поднят до запуска скрипта, это тоже стирает его данные (предупреждение выводится заранее). На Windows под Git Bash пути хоста конвертируются через `cygpath -w`, а `MSYS_NO_PATHCONV=1` отключает автоконвертацию MSYS для аргументов `--input /input`/`--output /output` — без этого MSYS переписывает контейнерные пути в Windows-путь и падает `Read-only file system`.

`app.cli submit` (`backend/app/cli.py`) по умолчанию ищет в `--input` каталог `images/` (плоский, `<image_id>.jpg`/`.png`) и файлы `test_gallery.csv`, `test_query.csv` — как в датасете хакатона. Импортирует галерею (`role=gallery`), затем запросы (`role=query`) той же функцией `services/ingest.ingest_row`, что и веб-импорт (каждая создаёт свою запись `import_jobs` — партию), после чего вызывает `services/export.build_export_artifacts` (раздел 6.6) и пишет три файла прямо в `--output` — без похода через S3 `exports/<id>/…`, к которому обращается только `GET /exports/{id}/files/{name}`.

Проверено на настоящем стеке дважды: изнутри контейнера (`docker exec` + потоковая передача файлов через `tar`, в обход `docker cp`/примонтированных томов) и целиком через `scripts/submit.sh` с реальными `test_gallery.csv`/`test_query.csv` — оба раза корректные `submission.csv`/`candidates.csv`/`embeddings.npy`, стек корректно поднимался и гасился с `-v`.

## 7. Модель данных backend

Использует таблицы схемы `reid` из `postgres/migrations/`; `002_import_jobs.sql` уже в репозитории (прежний документ называл её «нужно добавить»).

| Таблица | Кто пишет |
|---|---|
| `images`, `gallery_objects` | импорт, добавление объекта, `python -m app.cli submit` (скрипт `postgres/scripts/load_catalog.py` тоже пишет сюда, но после `003_roles.sql` упадёт — раздел 8, №13) |
| `threshold_history` | `PUT /threshold` |
| `search_queries`, `search_results` | `POST /queries` |
| `import_jobs` (`queued, running, done, failed, cancelled`) | импорт |
| `metric_jobs` (`queued, running, done, failed`, `run_id → metric_runs`) | оценка |
| `metric_runs` | оценка; **все** запуски остаются, читается последний по `created_at` |
| `export_jobs` | экспорт |

Соединение с PostgreSQL — `asyncpg`, пул 2…10, с кодеком `jsonb ↔ list/dict` (`_init_connection`, вызывается для каждого нового соединения пула); без него `export_jobs.artifacts` и `metric_runs.details` читались и писались как raw JSON-строка, а не список/словарь, и `GET /exports/{id}` падал с ошибкой валидации Pydantic — воспроизведено и **исправлено на настоящем стеке** (Postgres + Qdrant + inference на GPU): после фикса `GET /exports/{id}` многократно отдавал структурированные `artifacts` без ошибок валидации. Обращения к Qdrant (`qdrant-client`), S3 (`boto3`) и inference (`httpx.Client`) **синхронные** и вызываются прямо из `async`-обработчиков и фоновых задач: пока идёт такой вызов, цикл событий backend занят. Во время импорта или оценки остальные запросы к API могут отвечать с задержкой (замеров нет).

## 8. Известные расхождения и дефекты

Нумерация общая для документов. Признак «проверено» означает: воспроизведено на временной среде, но не на самом backend.

| № | Что | Где | Последствие | Проверка |
|---|---|---|---|---|
| 1 | ~~`GET /queries/history.csv` и `GET /metrics/runs/latest` объявлены после `/queries/{query_id}` и `/metrics/runs/{job_id}`~~ — **не подтвердилось**. В обоих файлах `history.csv`/`latest` объявлены раньше параметризованного маршрута (`routers/queries.py:135` до `:146`; `routers/metrics.py:163` до `:172`), FastAPI матчит по порядку объявления корректно. На живом стеке оба эндпоинта отдали 200 и ожидаемое содержимое; экран «Метрики и порог» сам подтянул последний запуск при открытии | — | опровергнуто живым прогоном (прежняя запись была основана на тестовом приложении с другим порядком маршрутов, не на реальном коде) |
| 2 | ~~Импорт и добавление объекта пишут в `images` ширину и высоту 0~~ | `services/database.py`, `routers/gallery.py`, `routers/imports.py` | Раньше отклонялось `CHECK (width > 0)`/`CHECK (height > 0)`. **Исправлено и проверено**: импорт 2380 объектов (`data/val_gallery.csv` + `data/val_query.csv`) через веб на настоящей БД прошёл без единой ошибки, `SELECT MIN(width), MIN(height) FROM images` = 1920×1080 (не 0) | живой прогон, реальная БД |
| 3 | Разные пространства имён uuid5: backend и `scripts/load_dataset.py` используют `NAMESPACE_DNS`, скрипты `vector/` и `postgres/` — `6f0f8a3e-…` | `routers/imports.py:69`, `routers/gallery.py:37`; `vector/scripts/load_gallery.py:25`, `postgres/scripts/load_catalog.py:25` | Один и тот же объект получает разные `id` в зависимости от способа загрузки. При смешении в Qdrant появляются дубли, а в PostgreSQL вставка второго `id` для той же пары кадр + рамка нарушает `UNIQUE (image_id, bbox…)`. Утверждение «единый id в Qdrant и PostgreSQL» верно только внутри одного способа загрузки | чтение кода |
| 4 | Ключи кадров в S3: `storage/scripts/upload_gallery.py` кладёт `<имя файла>` в корень бакета, backend читает `images/<image_id>.jpg` | `storage/scripts/upload_gallery.py:39`; `routers/objects.py`, `routers/queries.py` | Кадры, загруженные скриптом, backend не находит (миниатюры и Grad-CAM пустые), пока ключи не приведены к одной схеме. Соотношение `image_id` и имени файла (с расширением или без) в общем случае неизвестно: в черновых скриптах разработчика `image_id` без расширения | чтение кода |
| 5 | Метрики в `metrics.py` расходились с `methods.md` | `routers/metrics.py` | (а)–(г) **исправлено и проверено на настоящем стеке** ([`docs/methods.md`](methods.md#результаты-на-валидационной-выборке)): запрос и галерея больше не один и тот же пул — `val_query` ищет строго в `val_gallery` (`search_within_role`), самоисключение не нужно; открытые запросы естественно выделяются без ручного построения «отрицательных»; `PR-AUC` — площадь под кривой; при первой попытке поиск не возвращал `vehicle_id` у кандидатов — mAP/Rank/F1 были нулевыми при правдоподобном `PR-AUC` — исправлено; кросс-камерного исключения всё ещё нет и не может быть — camera_id участникам не передаётся вообще (ТЗ п.5.2), это ограничение датасета, а не код | живой прогон, Postgres + Qdrant + inference на GPU |
| 6 | ~~`GET /queries/{id}` возвращает неполные данные (нет `split` у кандидатов)~~ — снято вместе со `split`: поле `split` убрано из `Candidate` целиком (было информационным довеском ML-разработки, а не частью протокола поиска), схема больше его не ожидает. `gallery_size = 0` и время инференса/поиска = 0 у восстановленного запроса — это отдельное, оставшееся ограничение `GET /queries/{id}` | чтение кода |
| 7 | ~~Экспорт: `submission.csv` и `candidates.csv` были заглушками, `embeddings.npy` не соответствовал порядку `test.csv`~~ — **исправлено и проверено на настоящем стеке** (раздел 6.6/6.7): поиск top-10 идёт по текущей галерее (role='gallery'), а не внутри одного `split`; `embeddings.npy` — query-партия, затем вся галерея, обе в порядке `row_no`; проверено и через `POST /exports`, и через `app.cli submit`/`scripts/submit.sh` | живой прогон, Postgres + Qdrant + inference |
| 8 | Группировки по `vehicle_id` в поиске нет | `services/searcher.py` | В top-N несколько снимков одного автомобиля занимают несколько мест | чтение кода |
| 9 | Порты backend `8000` и inference `8001` опубликованы на всех интерфейсах, авторизации нет, CORS открыт | `backend/docker-compose.yml`, `inference/docker-compose.yml`, `main.py` | Прямой доступ к API и к `/internal/*` снаружи хоста, в отличие от БД и S3, которые слушают только `127.0.0.1` | чтение кода |
| 10 | Ошибка inference не переводится в понятное сообщение; кадр запроса остаётся в S3 | `routers/queries.py`, `services/inference_client.py` | Оператор видит «Ошибка сервера (500)» | чтение кода |
| 11 | Фоновые задания живут в памяти процесса | `routers/imports.py`, `metrics.py`, `exports.py` | После перезапуска backend задания остаются в статусе `running`/`queued` без исполнителя | чтение кода |
| 12 | Сводка «сегодня» в журнале считается с 00:00 **UTC**, а не по локальному времени сервера | `services/database.py` (`list_queries`) | Граница суток сдвинута относительно местного времени | чтение кода |
| 13 | В репозитории закоммичены рабочие ключи и пароли в `scripts/evaluate_api.py` и `scripts/load_dataset.py`, а также локальные пути разработчика | `scripts/evaluate_api.py`, `scripts/load_dataset.py` | Утечка секретов через git, скрипты не запускаются на другой машине. Значения не приводятся здесь намеренно: их нужно сменить (Qdrant, S3, PostgreSQL) и убрать из истории. Плюс к этому `scripts/load_dataset.py` пишет `split` в PostgreSQL — после `003_roles.sql` такой колонки нет, скрипт упадёт на первой вставке. Рабочий путь загрузки данных теперь — импорт через `/gallery/import-sessions` (UI «Галерея») или `python -m app.cli submit`/`scripts/submit.sh` (раздел 6.7); эти три скрипта (вместе с `vector/scripts/load_gallery.py`, `postgres/scripts/load_catalog.py` — см. `architecture.md`, «Расхождение (новое)») стоит удалить, а не чинить | чтение кода |

## 9. Структура файлов

```
backend/
├── Dockerfile, docker-compose.yml, requirements.txt, .dockerignore, .gitignore
└── app/
    ├── main.py                    приложение FastAPI, CORS, подключение роутеров (префикс /api/v1)
    ├── config.py                  переменные окружения
    ├── cli.py                     `python -m app.cli submit` — пакетный прогон (раздел 6.7)
    ├── routers/                   status, threshold, queries, objects, gallery, imports, metrics, exports
    ├── schemas/models.py          Pydantic-модели ответов
    ├── services/
    │   ├── database.py            asyncpg: все SQL-запросы
    │   ├── searcher.py            Qdrant: поиск (role='gallery'), search_within_batch/role, upsert, scroll
    │   ├── storage.py             S3 (boto3)
    │   ├── inference_client.py    HTTP-клиент inference
    │   ├── ingest.py              общая логика «кадр → эмбеддинг → запись» для импорта и app.cli submit
    │   └── export.py              общая логика построения submission.csv/embeddings.npy/candidates.csv
    └── worker/tasks.py            пустой файл-комментарий: фоновые задачи живут в роутерах

inference/
├── Dockerfile, docker-compose.yml, requirements.txt, .dockerignore, .gitignore
├── app/
│   ├── main.py                    FastAPI, эндпоинты /health и /internal/*
│   ├── reid_embed.py              LoRA, головы, загрузчики чекпоинтов, эмбеддинг, карты внимания
│   └── config.py
└── weights/                       вне образа, монтируется; в git оба больших файла лежат через LFS
```

Файлов из прежнего плана (`.env.example`, `services/metrics_calc.py`, `services/export.py`, `inference/app/model.py`) нет: расчёт метрик и экспорт находятся в роутерах, модель в `reid_embed.py`.

## 10. Не сделано и неизвестно

| Что | Статус |
|---|---|
| Код обучения модели (ТЗ п.8 требует «исходный код обучения и инференса») | в репозитории **нет** |
| Данные обучения, гиперпараметры, воспроизведение обучения | **неизвестно** |
| Метрики на валидации и обоснование порога | **измерено** на живом стеке: mAP@10 0,921, Rank-1 0,966, Rank-5 0,977 (раздел 8, №5); порог 0,79 выбран автоматически по кривой, отдельного обоснования стоимости ошибки нет |
| Время инференса, FPS, потребление видеопамяти, требуемая GPU | **не измерены** |
| Работа сквозного сценария UI → backend → inference → хранилища на живом стеке | в рамках этой сверки **не проверялась** |
| Версии Python-пакетов в собранных образах | **неизвестны**: закреплены только нижние границы, lock-файлов нет (после сборки: `docker compose run --rm --no-deps --entrypoint pip backend list`, для inference то же с сервисом `inference`) |
| Совместимость `transformers>=5.0` с torch из базового образа 2.2.0 | **неизвестна**: в `requirements.txt` torch указан как `>=2.2`, `pip` может заменить версию из базового образа |
| Что произойдёт, если рамка запроса в inference лежит за пределами кадра | **неизвестно** (проверки в inference нет) |
| Grad-CAM в смысле градиентной карты | для пары есть в inference (`/internal/compare`), в интерфейс не подключён |
