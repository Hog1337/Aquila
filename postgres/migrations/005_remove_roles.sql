-- Убираем роли (gallery/query/val_gallery/val_query) — всё становится просто галереей.
-- Роли больше не нужны: объекты не делятся по назначению, все поиски идут по всей галерее,
-- а для валидации/экспорта используются явные CSV-файлы, задающие подмножества.
SET search_path = reid, public;

ALTER TABLE gallery_objects DROP COLUMN role;
DROP INDEX IF EXISTS gallery_objects_role_idx;

ALTER TABLE import_jobs DROP COLUMN role;
DROP INDEX IF EXISTS import_jobs_role_idx;

-- В Qdrant роль тоже больше не пишем — payload чистится при перезаписи точек.
