-- Таблицы для фоновых заданий backend
-- import_jobs: импорт датасета (ZIP + CSV)
-- metric_jobs: запуски оценки метрик

SET search_path = reid, public;

CREATE TABLE IF NOT EXISTS import_jobs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  timestamptz NOT NULL DEFAULT now(),
    status      text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    progress    smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    processed   integer NOT NULL DEFAULT 0,
    total       integer NOT NULL DEFAULT 0,
    source_name text,
    split       text,
    error       text
);

CREATE TABLE IF NOT EXISTS metric_jobs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  timestamptz NOT NULL DEFAULT now(),
    status      text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    progress    smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    split       text NOT NULL,
    error       text,
    run_id      bigint REFERENCES metric_runs(id)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON import_jobs TO :"app_role";
GRANT SELECT, INSERT, UPDATE, DELETE ON metric_jobs TO :"app_role";
