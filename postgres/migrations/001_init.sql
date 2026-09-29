-- Схема реляционной СУБД «Фалькон ReID» (ТЗ п.6: PostgreSQL).
--
-- Разделение ответственности между хранилищами:
--   SeaweedFS  сами кадры (ключ объекта = image_id);
--   Qdrant     эмбеддинги галереи и векторный поиск (этап анализа, ТЗ п.4);
--   PostgreSQL каталог галереи, история запросов и результатов, порог отказа, метрики, задания экспорта.
--
-- Файл выполняется скриптом migrate.sh от имени администратора БД. Переменные psql:
--   :embedding_dim  размерность эмбеддинга (EMBEDDING_DIM, текущая модель JDNFV_MASKED_FT даёт 2048)
--   :app_role       имя роли backend-а (только DML, без DDL)
-- Файл применяется один раз, дальнейшие изменения схемы добавляйте новыми файлами 002_*.sql и т.д.

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid() для старых минорных версий

CREATE SCHEMA IF NOT EXISTS reid;
SET search_path = reid, public;

-- ---------------------------------------------------------------------------
-- Каталог галереи (ТЗ п.5)
-- ---------------------------------------------------------------------------

-- Кадр в бакете SeaweedFS. Плоский каталог images/ датасета: имя файла = image_id.
CREATE TABLE images (
    image_id    text PRIMARY KEY,                     -- ключ объекта в бакете gallery
    split       text NOT NULL CHECK (split IN ('gallery', 'train', 'val', 'test', 'user')),
    source      text NOT NULL DEFAULT 'dataset' CHECK (source IN ('dataset', 'upload')),
    width       integer CHECK (width  > 0),
    height      integer CHECK (height > 0),
    sha256      text,                                 -- контроль целостности и защита от дублей при загрузке
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX images_split_idx ON images (split);

-- Один размеченный автомобиль в кадре (в кадре бывает несколько ТС).
-- id совпадает с id точки в Qdrant: uuid5(image_id, x, y, w, h), повторная загрузка не создаёт дубль.
CREATE TABLE gallery_objects (
    id          uuid PRIMARY KEY,
    image_id    text NOT NULL REFERENCES images (image_id) ON DELETE CASCADE,
    bbox_x      integer NOT NULL CHECK (bbox_x >= 0),
    bbox_y      integer NOT NULL CHECK (bbox_y >= 0),
    bbox_w      integer NOT NULL CHECK (bbox_w > 0),
    bbox_h      integer NOT NULL CHECK (bbox_h > 0),
    vehicle_id  text,                                 -- идентичность ТС; NULL для test и запросов (ТЗ п.5, open-set)
    split       text NOT NULL CHECK (split IN ('gallery', 'train', 'val', 'test', 'user')),
    row_no      integer,                              -- номер строки CSV = строка embeddings.npy (ТЗ п.8)
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (image_id, bbox_x, bbox_y, bbox_w, bbox_h)
);
CREATE INDEX gallery_objects_image_idx   ON gallery_objects (image_id);
CREATE INDEX gallery_objects_vehicle_idx ON gallery_objects (vehicle_id) WHERE vehicle_id IS NOT NULL;
CREATE INDEX gallery_objects_split_idx   ON gallery_objects (split);

-- ---------------------------------------------------------------------------
-- Порог отказа (ТЗ п.3, п.9: порог выбирает команда и обосновывает на защите)
-- ---------------------------------------------------------------------------

-- Журнал изменений: действующий порог = последняя запись. Он один на поиск, метрики и экспорт (PRODUCT.md).
CREATE TABLE threshold_history (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    value       real NOT NULL CHECK (value BETWEEN -1 AND 1),   -- косинусное сходство
    reason      text,                                            -- обоснование выбора (метрики, кривая PR)
    changed_by  text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
-- Порог по умолчанию до первого выбора на странице «Метрики и порог»; backend читает его отсюда
INSERT INTO threshold_history (value, reason, changed_by)
VALUES (0.70, 'Значение по умолчанию до калибровки на валидационной выборке', 'migration');

CREATE VIEW current_threshold AS
    SELECT value, reason, changed_by, created_at
    FROM threshold_history
    ORDER BY id DESC
    LIMIT 1;

-- ---------------------------------------------------------------------------
-- История поиска (экран «История»)
-- ---------------------------------------------------------------------------

CREATE TABLE search_queries (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at        timestamptz NOT NULL DEFAULT now(),
    query_image_key   text,                          -- ключ кадра запроса в бакете (например, queries/<uuid>.jpg)
    bbox_x            integer NOT NULL CHECK (bbox_x >= 0),
    bbox_y            integer NOT NULL CHECK (bbox_y >= 0),
    bbox_w            integer NOT NULL CHECK (bbox_w > 0),
    bbox_h            integer NOT NULL CHECK (bbox_h > 0),
    top_n             smallint NOT NULL DEFAULT 10 CHECK (top_n > 0),
    threshold         real NOT NULL,                 -- порог, действовавший в момент запроса
    best_score        real,                          -- лучшее сходство (NULL, если галерея пуста)
    accepted_count    integer NOT NULL DEFAULT 0 CHECK (accepted_count >= 0),
    refused           boolean GENERATED ALWAYS AS (accepted_count = 0) STORED,   -- режим отказа (ТЗ п.4)
    elapsed_ms        integer CHECK (elapsed_ms >= 0),
    model_version     text,                          -- версия весов, которой получен эмбеддинг
    embedding         real[],                        -- эмбеддинг запроса: повторный поиск без инференса
    CONSTRAINT search_queries_embedding_dim CHECK (embedding IS NULL OR cardinality(embedding) = :embedding_dim)
);
CREATE INDEX search_queries_created_idx ON search_queries (created_at DESC);
CREATE INDEX search_queries_refused_idx ON search_queries (created_at DESC) WHERE refused;

-- Ранжированный список кандидатов запроса. Хранятся все top_n, флаг accepted отделяет принятых (score >= порога).
CREATE TABLE search_results (
    query_id    uuid     NOT NULL REFERENCES search_queries (id) ON DELETE CASCADE,
    rank        smallint NOT NULL CHECK (rank > 0),
    object_id   uuid     REFERENCES gallery_objects (id) ON DELETE SET NULL,
    image_id    text     NOT NULL,                   -- дублируется: история переживает очистку галереи
    vehicle_id  text,
    score       real     NOT NULL,
    accepted    boolean  NOT NULL,
    PRIMARY KEY (query_id, rank)
);
CREATE INDEX search_results_object_idx ON search_results (object_id);

-- ---------------------------------------------------------------------------
-- Метрики качества (экран «Метрики и порог», документация ТЗ п.12)
-- ---------------------------------------------------------------------------

CREATE TABLE metric_runs (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at     timestamptz NOT NULL DEFAULT now(),
    split          text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    model_version  text,
    threshold      real,                              -- порог, при котором посчитаны F1 и TNR
    map            real,                              -- mAP (кросс-камерные сопоставления, ТЗ п.9)
    rank1          real,
    rank5          real,
    f1             real,
    tnr            real,
    pr_auc         real,
    n_queries      integer,
    details        jsonb NOT NULL DEFAULT '{}'::jsonb -- кривая PR, распределения сходства и т.п.
);
CREATE INDEX metric_runs_created_idx ON metric_runs (created_at DESC);

-- ---------------------------------------------------------------------------
-- Экспорт (экран «Экспорт», ТЗ п.8: submission.csv, embeddings.npy, candidates.csv)
-- ---------------------------------------------------------------------------

CREATE TABLE export_jobs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    status       text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    progress     smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    source       text NOT NULL DEFAULT 'test.csv',
    threshold    real NOT NULL,
    row_count    integer,
    artifacts    jsonb NOT NULL DEFAULT '[]'::jsonb,  -- [{"name":"submission.csv","key":"exports/<id>/submission.csv","bytes":123}]
    error        text
);
CREATE INDEX export_jobs_created_idx ON export_jobs (created_at DESC);

-- ---------------------------------------------------------------------------
-- Права backend-а: только данные, без DDL и без права владения
-- ---------------------------------------------------------------------------

GRANT USAGE ON SCHEMA reid TO :"app_role";
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA reid TO :"app_role";
GRANT SELECT ON current_threshold TO :"app_role";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA reid TO :"app_role";
-- Таблицы из будущих миграций (создаёт администратор) получают те же права автоматически
ALTER DEFAULT PRIVILEGES IN SCHEMA reid GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"app_role";
ALTER DEFAULT PRIVILEGES IN SCHEMA reid GRANT USAGE, SELECT ON SEQUENCES TO :"app_role";
-- Журнал порога append-only: backend только дописывает строки, правка и удаление задним числом запрещены
REVOKE DELETE, UPDATE ON threshold_history FROM :"app_role";
