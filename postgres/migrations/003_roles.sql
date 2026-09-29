-- Убирает ML-сплиты (train/val/test/gallery/user) — они смешивали два разных смысла:
-- "какая это подвыборка для обучения модели" (не нужно бэкенду) и "что делать с объектом
-- в продукте" (искать по нему или в нём). Второе остаётся под новым, явным именем:
--   role     — gallery (галерея поиска) | query (объекты для идентификации) |
--              val_gallery / val_query (собственный calibration-сплит для страницы «Метрики»,
--              который теперь тоже "запрос против своей галереи", а не пул, ищущий сам в себе);
--   batch_id — какая именно партия импорта принесла объект. Нужен, чтобы экспорт мог
--              работать со строго конкретной парой query-CSV/gallery-CSV, а не со всей
--              когда-либо накопленной галереей (ТЗ п.8: воспроизводимость embeddings.npy).
--
-- Старые данные размечены по train/val/test без привязки к batch_id и под новую модель
-- не пересчитываются — это dev-стенд, переносить их не нужно (согласовано с пользователем).
SET search_path = reid, public;

DELETE FROM metric_jobs;
DELETE FROM metric_runs;
DELETE FROM export_jobs;
DELETE FROM images;        -- каскадом удаляет gallery_objects (ON DELETE CASCADE)
DELETE FROM import_jobs;

ALTER TABLE images DROP COLUMN split;   -- значение никогда не читалось, только писалось

ALTER TABLE gallery_objects DROP COLUMN split;
ALTER TABLE gallery_objects
    ADD COLUMN role     text NOT NULL CHECK (role IN ('gallery', 'query', 'val_gallery', 'val_query')),
    ADD COLUMN batch_id uuid REFERENCES import_jobs (id) ON DELETE SET NULL;  -- NULL — ручное «Добавить ТС»
CREATE INDEX gallery_objects_role_idx  ON gallery_objects (role);
CREATE INDEX gallery_objects_batch_idx ON gallery_objects (batch_id);

ALTER TABLE import_jobs DROP COLUMN split;
ALTER TABLE import_jobs ADD COLUMN role text NOT NULL CHECK (role IN ('gallery', 'query', 'val_gallery', 'val_query'));
CREATE INDEX import_jobs_role_idx ON import_jobs (role) WHERE status = 'done';

ALTER TABLE metric_runs DROP COLUMN split;   -- единственный режим оценки: val_query против val_gallery
ALTER TABLE metric_jobs DROP COLUMN split;

-- Экспорт теперь явно привязан к конкретной паре партий (источник на странице «Экспорт», ТЗ п.8),
-- а не к неявному "текущему состоянию" галереи под именем test.csv.
ALTER TABLE export_jobs DROP COLUMN source;
ALTER TABLE export_jobs
    ADD COLUMN query_batch_id   uuid NOT NULL REFERENCES import_jobs (id),
    ADD COLUMN gallery_batch_id uuid NOT NULL REFERENCES import_jobs (id);
