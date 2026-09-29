# Реляционная СУБД (PostgreSQL)

Здесь хранится всё, что не является ни кадром, ни вектором галереи: каталог галереи, история поиска и результатов, действующий порог отказа, метрики качества и задания экспорта. Так закрывается пункт ТЗ п.6 о реляционной СУБД для метаданных.

| Хранилище | Что лежит | Каталог |
|---|---|---|
| SeaweedFS | кадры, ключ объекта = `image_id` | `../storage` |
| Qdrant | эмбеддинги галереи, векторный поиск (этап анализа, ТЗ п.4) | `../vector` |
| PostgreSQL | каталог, история, порог, метрики, экспорт | здесь |

Выбран [PostgreSQL](https://www.postgresql.org/) 17 (лицензия PostgreSQL, свободное ПО, работает без интернета) как обычная реляционная база, без расширений: эмбеддинг запроса хранится в колонке `real[]` (длина проверяется ограничением `search_queries_embedding_dim`) вместе с историей, а значит любой прошлый запрос можно повторить при другом пороге без нового инференса. Векторный поиск по галерее целиком за Qdrant, PostgreSQL по векторам не ищет.

## Запуск

```bash
docker compose up -d --build --wait   # из каталога postgres/ (или из aquila/ вместе с остальными сервисами)
```

Готовность: `postgres` в состоянии `healthy`, `postgres-init` завершился с кодом 0 (роль создана, миграции применены). Без `.env` стек стартует с dev-паролями из `docker-compose.yml`: этого достаточно для локальной проверки, для демо-хоста пароли задаются отдельно (см. `.env.example`).

| Что | Значение по умолчанию |
|---|---|
| Порт снаружи | `localhost:5432` (`POSTGRES_BIND`, `POSTGRES_PORT`) |
| Адрес из сети compose | `postgres:5432`, сеть `falcon-reid` |
| База и схема | `reid`, схема `reid` |
| Роль backend-а | `reidbackend` (`POSTGRES_APP_USER`, пароль `POSTGRES_APP_PASSWORD`) |
| Администратор | `reidadmin` (`POSTGRES_ADMIN_USER`): только миграции и обслуживание |
| Образ | `postgres:17-alpine` (`POSTGRES_VERSION`) |
| Данные | volume `postgres-data` (переживает `down`, удаляется только `down -v`) |

Порт слушает только localhost. Открывая его наружу (`POSTGRES_BIND=0.0.0.0`), обязательно меняйте оба пароля. Внутри сети compose и через порт хоста вход только по паролю (SCRAM-SHA-256).

Две роли, как и в `../storage`: `reidbackend` (чтение и запись данных схемы `reid`, без права менять структуру, его отдаём backend-у) и `reidadmin` (владелец объектов, нигде в сервисах не используется). Журнал порога `threshold_history` для backend-а append-only: правка и удаление запрещены.

## Схема `reid`

Миграции лежат в `migrations/` и вшиты в образ `postgres-init`. Применённые записаны в `public.reid_migrations`, повторный запуск ничего не меняет.

| Таблица | Назначение | Экран |
|---|---|---|
| `images` | кадр в бакете (`image_id`, `split`, размеры, sha256) | Галерея |
| `gallery_objects` | один автомобиль в кадре: bbox, `vehicle_id`, `split`, `row_no`; `id` = id точки в Qdrant | Галерея |
| `threshold_history` | журнал порога отказа с обоснованием; действующий порог читается из представления `current_threshold` | Метрики и порог |
| `search_queries` | запрос: bbox, порог на момент запроса, лучшее сходство, `refused`, время ответа, версия модели, ключ кадра запроса в S3, эмбеддинг `real[]` (512 значений) | История |
| `search_results` | ранжированные кандидаты запроса: `rank`, `score`, `accepted` (сходство не ниже порога) | Поиск, История |
| `metric_runs` | mAP, Rank-1, Rank-5, F1, TNR, PR-AUC для выборки и порога, кривая по порогам в `details.curve` (значения считает backend; как именно, см. `../docs/backend.md`, раздел 6.5) | Метрики и порог |
| `export_jobs` | задание экспорта: статус, прогресс, порог, список файлов (`submission.csv`, `embeddings.npy`, `candidates.csv`) | Экспорт |
| `import_jobs` | задание импорта галереи (ZIP + CSV): статус `queued/running/done/failed/cancelled`, прогресс, счётчики (миграция `002_import_jobs.sql`) | Галерея |
| `metric_jobs` | задание оценки метрик: статус, прогресс, ссылка `run_id` на `metric_runs` (миграция `002_import_jobs.sql`) | Метрики и порог |

Связи между хранилищами: `image_id` (ключ объекта в S3) и `gallery_objects.id` (тот же `uuid5(image_id, x, y, w, h)`, что использует Qdrant, см. `../vector/README.md`). Загрузка галереи пишет в обе базы: повторный запуск того же способа загрузки перезаписывает записи, а не создаёт дубли. **Пространство имён uuid5 у способов загрузки разное:** скрипты `scripts/load_catalog.py` и `../vector/scripts/load_gallery.py` используют `6f0f8a3e-5a0e-4c53-9d0a-4f414c4b4c4f`, backend (импорт и «Добавить ТС») использует `uuid.NAMESPACE_DNS`. Один объект, загруженный обоими способами, получит два разных `id`, а вторая вставка нарушит `UNIQUE (image_id, bbox…)`; см. `../docs/backend.md`, раздел 8, №3.

Отказ (ТЗ п.4): `search_queries.accepted_count = 0`, столбец `refused` вычисляется сам. В `search_results` хранятся все `top_n` кандидатов, а `accepted` показывает, какие из них прошли порог; так на экране «История» видно, насколько близко к порогу был отказ.

## Загрузка каталога галереи

```bash
pip install -r scripts/requirements.txt
python scripts/load_catalog.py --csv ./dataset/train.csv --split gallery
```

Тот же CSV (ТЗ п.5: `image_id, x, y, w, h, vehicle_id`), что и для `../vector/scripts/load_gallery.py`, и та же метка `--split`: скрипты работают в паре, id объектов у них совпадают. Скрипт идемпотентен. Доступ берётся из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_APP_USER`, `POSTGRES_APP_PASSWORD` (по умолчанию `127.0.0.1:5432`, dev-пароль из `docker-compose.yml`).

## Подключение из backend

Реальный backend работает через `asyncpg` (пул 2…10 соединений, `POSTGRES_DSN`), а не через `psycopg`; пример ниже показывает тот же доступ синхронным драйвером, который используется в скриптах этого каталога.

```python
import psycopg   # psycopg[binary] >=3.2

conn = psycopg.connect(host="postgres", dbname="reid", user=POSTGRES_APP_USER, password=POSTGRES_APP_PASSWORD)
# search_path роли уже reid, public: таблицы доступны без префикса

threshold = conn.execute("SELECT value FROM current_threshold").fetchone()[0]

conn.execute(
    "INSERT INTO threshold_history (value, reason, changed_by) VALUES (%s, %s, %s)",
    (0.72, "максимум F1 на val, TNR не ниже 0.9", "operator"),
)
```

- Порог живёт здесь: один на поиск, метрики и экспорт (`PRODUCT.md`). Backend читает его из `current_threshold` и сравнивает с ним `score` кандидатов сам (`accepted = score >= порог`); в Qdrant порог не передаётся (`score_threshold` не используется).
- Адрес БД задаётся backend-у одной строкой `POSTGRES_DSN` (`postgresql://<роль>:<пароль>@postgres:5432/reid`, собирается в `backend/docker-compose.yml` из `POSTGRES_APP_USER` и `POSTGRES_APP_PASSWORD`); backend подключён к сети `falcon-reid`.
- Размерность `EMBEDDING_DIM` должна совпадать с `../vector/.env.example`.

## Обслуживание

```bash
docker compose logs postgres                     # логи
docker compose run --rm postgres-init            # повторная инициализация (безопасна)
docker compose exec postgres psql -U reidadmin -d reid   # консоль администратора
docker compose down -v                           # ВНИМАНИЕ: удалит всю историю, каталог и метрики
```

Резервная копия и восстановление:

```bash
docker compose exec -T postgres pg_dump -U reidadmin -Fc reid > reid.dump
docker compose exec -T postgres pg_restore -U reidadmin -d reid --clean --if-exists < reid.dump
```

Смена пароля: поменять значение в `.env` и выполнить `docker compose up -d --wait`. Пароль администратора хранится в volume с момента первой инициализации, поэтому его смена в `.env` без `ALTER ROLE` не действует; пароль роли backend-а синхронизируется `postgres-init` при каждом запуске.

Новая миграция: файл `migrations/005_описание.sql` (`001_init.sql`, `002_import_jobs.sql`, `003_roles.sql`, `004_export_gallery_pool.sql` уже есть), затем `docker compose up -d --build --wait`. Переменные psql `:embedding_dim` и `:app_role` доступны в каждой миграции; таблицы, созданные администратором в схеме `reid`, автоматически получают права роли backend-а.

Обновление минорной версии PostgreSQL: поменять `POSTGRES_VERSION` в `.env`. Переход на другую мажорную версию PostgreSQL требует `pg_dump` и `pg_restore`, перед обновлением на демо-хосте сделайте копию. Образ на PostgreSQL 18 хранит данные по другому пути, поэтому переход на него отдельная задача.

## Зависимости (ТЗ п.7, п.12)

| Компонент | Версия |
|---|---|
| PostgreSQL | 17 |
| Образ | `postgres:17-alpine` |
| psycopg (скрипт `load_catalog.py`) | >=3.2,<4 |
| asyncpg (backend) | >=0.29 (нижняя граница, точная версия не зафиксирована) |

## Запуск отдельно от остальных

Имя проекта compose у всех частей и корня одно (`falcon-reid`): общая сеть `falcon-reid` и тома `falcon-reid_<ключ>` одинаковы при запуске из корня и из этого каталога. Если соседние сервисы уже работают, при запуске отсюда Compose предупредит про orphan-контейнеры: это нормально, **не используйте `--remove-orphans`** (он остановит соседей); предупреждение убирает `COMPOSE_IGNORE_ORPHANS=true`.

## Ограничения

- Один узел без репликации: этого достаточно для демо и истории в сотни тысяч запросов. Репликация и пул соединений в поставку не входят.
- Пароли только из латиницы и цифр; имя роли backend-а: строчные латинские буквы, цифры и `_`.
- Смена `EMBEDDING_DIM` после первой инициализации требует новой миграции или пересоздания volume: инициализация остановится с понятной ошибкой.
- В `images` стоят `CHECK (width > 0)` и `CHECK (height > 0)`: значение `NULL` проходит, `0` нет (проверено на временной БД). Backend при импорте и добавлении объекта пишет 0 и потому ожидаемо получает отказ; `../docs/backend.md`, раздел 8, №2.
- В комментариях `migrations/001_init.sql` упомянута модель `reid-vit-s`: это устаревшее название, фактическая модель описана в `../docs/methods.md`, раздел 2 (менять применённую миграцию ради комментария не стали).
