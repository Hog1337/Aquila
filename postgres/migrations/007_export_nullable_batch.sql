-- Делает query_batch_id в export_jobs опциональным.
-- Раньше экспорт всегда привязывался к сессии импорта (import_jobs),
-- теперь можно создавать export без batch_id (через новый флайер загрузки).

ALTER TABLE reid.export_jobs ALTER COLUMN query_batch_id DROP NOT NULL;
ALTER TABLE reid.export_jobs DROP CONSTRAINT IF EXISTS export_jobs_query_batch_id_fkey;
