# Два образа из одного файла (цель выбирается `target` в docker-compose.yml):
#   init   одноразовая инициализация: роль backend-а и миграции;
#   server сервер PostgreSQL (по умолчанию, последняя стадия).
# Базовый образ скачивается при `docker build` (сборке нужен интернет), при запуске сеть не нужна.
ARG POSTGRES_IMAGE=postgres:17-alpine

# ---- init: тот же образ, что и у сервера (там уже есть psql и pg_isready), плюс скрипт миграций и сами миграции.
# Вшиты в образ, а не монтируются с хоста: стек не зависит от путей и прав на хосте (в том числе при деплое из CI/CD).
FROM ${POSTGRES_IMAGE} AS init

COPY --chmod=0555 scripts/migrate.sh /migrate.sh
# 0555: у каталога нужен бит x, иначе клиент под uid 10001 не сможет читать файлы
COPY --chmod=0555 migrations/ /migrations/

# Не root; uid 10001 не совпадает с postgres (999), клиенту права на данные сервера не нужны
USER 10001
ENTRYPOINT ["/bin/sh", "/migrate.sh"]

# ---- server: официальный PostgreSQL с настройками решения (пароли и имя БД задаёт compose при запуске)
FROM ${POSTGRES_IMAGE} AS server

ENV POSTGRES_DB=reid \
    POSTGRES_USER=reidadmin
# Быстрая остановка: SIGINT, при выключении PostgreSQL успевает сделать checkpoint
STOPSIGNAL SIGINT
LABEL org.opencontainers.image.title="falcon-reid postgres" \
      org.opencontainers.image.description="PostgreSQL для метаданных Фалькон ReID"
