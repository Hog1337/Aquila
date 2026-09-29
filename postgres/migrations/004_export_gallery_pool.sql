-- Экспорт больше не требует выбирать партию-галерею отдельно: галерея одна и та же для поиска,
-- экспорта и «Добавить ТС» (role='gallery'), как и раньше до появления ролей. Источник на странице
-- «Экспорт» — это только партия запросов; галерея всегда текущая (services/export.py).
SET search_path = reid, public;

DELETE FROM export_jobs;  -- gallery_batch_id обязателен в старой схеме — очищаем перед сменой колонок
ALTER TABLE export_jobs DROP COLUMN gallery_batch_id;
