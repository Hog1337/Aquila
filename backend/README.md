# Backend (FastAPI)

REST API `/api/v1`: поиск с режимом отказа, галерея, порог, метрики, история, экспорт.
Ходит в PostgreSQL (`asyncpg`), Qdrant (`qdrant-client`), SeaweedFS (`boto3`)
и в сервис `inference` (`httpx`). Раздаёт OpenAPI: Swagger на `/docs`, схема на `/openapi.json`.
Подробное описание, сценарии: [`../docs/backend.md`](../docs/backend.md).

## Запуск

Вместе с остальным решением: `./run.sh` из корня `aquila/`.
Отдельно:

```bash
docker compose up --build -d
```

Нужны запущенные `postgres`, `qdrant`, `seaweedfs` и сеть `falcon-reid`.
Без `inference` сервис стартует, но поиск, импорт и карты внимания будут недоступны.

| Что | Значение |
|---|---|
| Порт | `127.0.0.1:8000` (`BACKEND_PORT`) |
| Образ | `python:3.12-slim` |
| Память | `BACKEND_MEM_LIMIT` (по умолч. 2 ГБ) |

## Проверка

```bash
curl -s http://localhost:8000/api/v1/status      # модель, размерность, галерея
curl -s http://localhost:8000/api/v1/threshold   # действующий порог
```

## Структура

```
app/main.py               приложение, CORS, роутеры, запуск флашеров
app/config.py             переменные окружения
app/routers/              status, threshold, queries, objects, gallery, imports, metrics, exports
app/schemas/models.py     модели ответов (Pydantic)
app/services/             database (asyncpg), searcher (Qdrant), storage (S3),
                          inference_client, processor (async batch inference + Qdrant flusher),
                          ingest, export
app/cli.py                CLI: submit (полный цикл), evaluate
```

## Что важно знать

- Фоновые задания — `asyncio`-задачи в процессе backend, не переживают перезапуск
- Вызовы Qdrant, S3 и inference — синхронные, обёрнуты в `asyncio.to_thread`
- Батч-инференс и Qdrant upsert буферизируются и сбрасываются фоновыми флашерами
