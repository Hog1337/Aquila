#!/usr/bin/env bash
# Дымовые проверки поднятого стека (запускать из любого места, использует docker compose корня решения).
#   scripts/smoke.sh
# Одни и те же проверки выполняют: разработчик после ./run.sh, Jenkins на агенте и после деплоя на aquila.
# Проект compose берётся из COMPOSE_PROJECT_NAME, если он задан (в CI это уникальный проект на сборку).
set -euo pipefail

cd "$(dirname "$0")/.."
fail=0

ok() { echo "[smoke] ok     $1"; }
bad() { echo "[smoke] ОШИБКА $1" >&2; fail=1; }

# 1. Все долгоживущие сервисы healthy (init-сервисы по замыслу завершаются)
not_healthy=$(docker compose ps -a --format '{{.Service}} {{.Status}}' | grep -v -- '-init' | grep -v -E ' Up .*\(healthy\)' || true)
if [ -z "$not_healthy" ]; then ok "все сервисы healthy"; else bad "не healthy: $not_healthy"; fi

# 2. PostgreSQL: обычная база без pgvector, схема reid создана миграциями.
# SQL идёт через stdin: так не нужно экранировать кавычки.
psql_query() { docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At' | tr -d '\r'; }
extensions=$(echo "select string_agg(extname, ',' order by extname) from pg_extension" | psql_query)
if [ "$extensions" = "pgcrypto,plpgsql" ]; then
  ok "PostgreSQL: расширения $extensions"
else
  bad "PostgreSQL: расширения '$extensions', ожидалось pgcrypto,plpgsql"
fi
tables=$(echo "select count(*) from pg_tables where schemaname = 'reid'" | psql_query)
if [ "${tables:-0}" -ge 7 ]; then
  ok "PostgreSQL: таблиц в схеме reid: $tables"
else
  bad "PostgreSQL: таблиц в схеме reid: ${tables:-0}, ожидалось не меньше 7"
fi

# 3. Qdrant: коллекция gallery создана, 2048 x float32. В образе нет curl, поэтому запрос через bash и /dev/tcp.
collection=$(docker compose exec -T qdrant bash -c \
  'exec 3<>/dev/tcp/127.0.0.1/6333; printf "GET /collections/gallery HTTP/1.0\r\napi-key: %s\r\nHost: localhost\r\n\r\n" "$QDRANT__SERVICE__API_KEY" >&3; cat <&3' \
  | tr -d '\r')
if printf '%s' "$collection" | grep -q '"size":2048' && printf '%s' "$collection" | grep -q '"status":"green"'; then
  ok "Qdrant: коллекция gallery готова, dim=2048"
else
  bad "Qdrant: коллекция gallery не готова"
fi

# 4. SeaweedFS: бакет gallery создан (скрипт запуска кладёт флаг готовности)
if docker compose exec -T seaweedfs test -f /tmp/bucket.ready; then ok "SeaweedFS: бакет gallery создан"; else bad "SeaweedFS: бакет gallery не создан"; fi

# 5. Frontend: живость и отдача index.html
if [ "$(docker compose exec -T frontend wget -qO- http://127.0.0.1:8080/healthz | tr -d '\r')" = "ok" ]; then
  ok "frontend: /healthz"
else
  bad "frontend: /healthz"
fi
if docker compose exec -T frontend wget -qO- http://127.0.0.1:8080/ | grep -q '<div id="root">'; then
  ok "frontend: отдаёт index.html"
else
  bad "frontend: не отдаёт index.html"
fi

# 6. Backend: API отвечает и видит PostgreSQL (/api/v1/status читает счётчики галереи из БД)
if docker compose exec -T backend python3 -c \
  "import httpx; r = httpx.get('http://localhost:8000/api/v1/status'); assert r.status_code == 200 and 'model_version' in r.json()" \
  >/dev/null 2>&1; then
  ok "backend: /api/v1/status"
else
  bad "backend: /api/v1/status"
fi

# 7. Inference: модель загружена. Сервис может быть не запущен (SKIP_SERVICES=inference, нет GPU), тогда пропускаем.
if [ -n "$(docker compose ps --status running --services | grep -x inference || true)" ]; then
  if docker compose exec -T inference curl -fsS http://localhost:8001/health | grep -q '"status":"ok"'; then
    ok "inference: модель загружена"
  else
    bad "inference: модель не загружена (нет весов в inference/weights или GPU?)"
  fi
else
  echo "[smoke] пропуск inference: сервис не запущен"
fi

if [ "$fail" = 0 ]; then echo "[smoke] все проверки пройдены"; else echo "[smoke] есть ошибки" >&2; exit 1; fi
