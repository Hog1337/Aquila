-- Добавляет статус обработки в таблицу images для асинхронной векторизации.
-- Статусы: uploaded → processing → ready / failed

ALTER TABLE reid.images
    ADD COLUMN IF NOT EXISTS processing_status text
        NOT NULL DEFAULT 'ready'
        CHECK (processing_status IN ('uploaded', 'processing', 'ready', 'failed'));

ALTER TABLE reid.images
    ADD COLUMN IF NOT EXISTS processing_error text;
