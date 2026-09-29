#!/bin/sh
# Одноразовая инициализация PostgreSQL для «Фалькон ReID» (идемпотентна, безопасна при повторном запуске):
#   1. ждёт готовности сервера;
#   2. создаёт (или синхронизирует пароль) роль backend-а: только DML, без DDL;
#   3. применяет миграции /migrations/NNN_*.sql по порядку; применённые записываются в reid_migrations.
# Подключается администратором БД; пароли берутся из переменных окружения (в командную строку не попадают).
set -eu

: "${PGHOST:?не задан PGHOST}"
: "${PGUSER:?не задан PGUSER}"
: "${PGPASSWORD:?не задан PGPASSWORD}"
: "${PGDATABASE:?не задан PGDATABASE}"
: "${APP_DB_USER:?не задан APP_DB_USER}"
: "${APP_DB_PASSWORD:?не задан APP_DB_PASSWORD}"
: "${EMBEDDING_DIM:=2048}"

# Имя роли подставляется в SQL как идентификатор: допускаем только безопасные символы
case "$APP_DB_USER" in
  ''|*[!a-z0-9_]*) echo "[init] APP_DB_USER должен состоять из строчных латинских букв, цифр и _" >&2; exit 1 ;;
esac
case "$EMBEDDING_DIM" in
  ''|*[!0-9]*) echo "[init] EMBEDDING_DIM должен быть числом" >&2; exit 1 ;;
esac

PSQL="psql -X -v ON_ERROR_STOP=1 -q"

i=0
until pg_isready -q; do
  i=$((i + 1))
  if [ "$i" -ge 60 ]; then echo "[init] PostgreSQL не отвечает по адресу ${PGHOST}" >&2; exit 1; fi
  sleep 2
done

# Роль backend-а. Пароль читается psql-ом из окружения (\getenv), а не из аргументов процесса.
$PSQL -v app_role="$APP_DB_USER" <<'SQL'
\getenv app_password APP_DB_PASSWORD
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_role', :'app_password')
 WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_role')
\gexec
-- Пароль синхронизируется при каждом запуске: смена APP_DB_PASSWORD в .env применяется повторным up
SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', :'app_role', :'app_password')
\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'app_role')
\gexec
-- Backend обращается к таблицам без префикса: схема reid в пути поиска роли
SELECT format('ALTER ROLE %I IN DATABASE %I SET search_path = reid, public', :'app_role', current_database())
\gexec
SQL

$PSQL -c "CREATE TABLE IF NOT EXISTS public.reid_migrations (
            name       text PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now())"

# Глоб сортируется лексикографически: имена NNN_*.sql задают порядок применения
for file in /migrations/*.sql; do
  name=$(basename "$file")
  done_already=$($PSQL -At -c "SELECT 1 FROM public.reid_migrations WHERE name = '$name'")
  if [ -n "$done_already" ]; then
    echo "[init] $name уже применена, пропускаю"
    continue
  fi
  echo "[init] применяю $name"
  # Миграция и запись о ней в одной транзакции: сбой на середине ничего не оставляет.
  # Файл собирается заранее, а не через конвейер: иначе `set -e` не заметит сбой cat.
  script=$(mktemp)
  {
    echo "BEGIN;"
    cat "$file"
    echo "INSERT INTO public.reid_migrations (name) VALUES ('$name');"
    echo "COMMIT;"
  } > "$script"
  $PSQL -v app_role="$APP_DB_USER" -v embedding_dim="$EMBEDDING_DIM" -f "$script"
  rm -f "$script"
done

# Размерность нельзя менять на месте: ловим расхождение с ограничением, созданным миграцией 001
# (pg_get_constraintdef возвращает, например, `CHECK (((embedding IS NULL) OR (cardinality(embedding) = 512)))`)
actual=$($PSQL -At -c "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'search_queries_embedding_dim' AND conrelid = 'reid.search_queries'::regclass")
if [ -n "$actual" ] && ! printf '%s' "$actual" | grep -q "cardinality(embedding) = $EMBEDDING_DIM)"; then
  echo "[init] В БД ограничение размерности embedding: $actual, а EMBEDDING_DIM=$EMBEDDING_DIM." >&2
  echo "[init] Создайте миграцию, меняющую тип колонки, либо удалите volume postgres-data." >&2
  exit 1
fi

echo "[init] PostgreSQL готов: схема reid, роль ${APP_DB_USER}"
