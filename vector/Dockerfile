# Два образа из одного файла (цель выбирается `target` в docker-compose.yml):
#   init   одноразовая инициализация коллекции;
#   server векторная БД Qdrant (по умолчанию, последняя стадия).
# Базовые образы скачиваются при `docker build` (сборке нужен интернет), при запуске сеть не нужна.
ARG PYTHON_IMAGE=python:3.12-alpine
ARG QDRANT_IMAGE=qdrant/qdrant:v1.19.1

# ---- init: скрипт использует только стандартную библиотеку Python, поэтому pip при сборке не нужен
FROM ${PYTHON_IMAGE} AS init

RUN adduser -D -H -u 10001 init
USER 10001
COPY --chmod=0555 scripts/init_collection.py /init_collection.py

ENTRYPOINT ["python", "-u", "/init_collection.py"]

# ---- server: официальный Qdrant с настройками решения (ключ API и уровень логов задаёт compose при запуске)
FROM ${QDRANT_IMAGE} AS server

# Телеметрия выключена; снимки лежат в volume вместе с данными: файловая система контейнера остаётся read-only
ENV QDRANT__TELEMETRY_DISABLED=true \
    QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots
LABEL org.opencontainers.image.title="falcon-reid qdrant" \
      org.opencontainers.image.description="Векторная БД галереи Фалькон ReID"
