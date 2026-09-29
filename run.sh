#!/usr/bin/env bash
# Запуск всего решения одной командой.
#   ./run.sh             собрать образы (нужен интернет: базовые образы, npm) и поднять стек
#   ./run.sh --registry  образы берутся из GitLab Registry, без сборки (деплой из CI/CD);
#                        нужны переменные REGISTRY_BASE (например registry.git.hog1337.com/wallcrepers2/aquila) и TAG
#   ./run.sh --down      остановить стек (данные в volume сохраняются)
#   SKIP_SERVICES="inference" ./run.sh   не поднимать перечисленные сервисы (через пробел),
#                        например inference на машине без GPU или без весов модели (inference/weights)
#
# Сборка образов идёт с интернетом, запуск собранных образов интернета не требует.
#
# Что делает:
#   0. создаёт ./.env (лимиты памяти контейнеров) из .env.example и подхватывает его для всех сервисов;
#   1. создаёт <сервис>/.env из .env.example, заменяя все `change-me-*` случайными значениями
#      (существующие .env не трогает: секреты переживают перезапуск и обновление);
#   2. собирает образы всех сервисов (frontend, backend, inference, seaweedfs, postgres, qdrant и init-образы)
#      и поднимает хранилища (postgres, qdrant, seaweedfs);
#   3. выполняет одноразовые init-сервисы (коллекция Qdrant, роль и миграции PostgreSQL);
#   4. поднимает остальные сервисы (backend, frontend, inference): им нужны роль в БД и коллекция из шага 3.
set -euo pipefail

cd "$(dirname "$0")"

# Лимиты памяти и другие несекретные настройки стека живут в ./.env (создаётся из .env.example, существующий не трогается).
# Значения экспортируются, поэтому действуют на compose-файлы всех сервисов, а не только на корневой.
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi
if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  . ./.env
  set +a
fi

MODE=build
case "${1:-}" in
  --down) docker compose down; exit 0 ;;
  --registry) MODE=registry ;;
  --local) MODE=local ;;
  "") ;;
  *) echo "[run] неизвестный параметр: $1 (допустимо --registry, --local, --down)" >&2; exit 2 ;;
esac

command -v docker >/dev/null || { echo "[run] docker не найден" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "[run] нужен Docker Compose v2.20 или новее" >&2; exit 1; }

UP_BUILD=--build
if [ "$MODE" = registry ]; then
  : "${REGISTRY_BASE:?для --registry задайте REGISTRY_BASE}"
  : "${TAG:?для --registry задайте TAG (тег сборки в реестре)}"
  echo "[run] режим: реестр ($REGISTRY_BASE, тег $TAG)"
  export BUILT_PULL_POLICY=always
  export FRONTEND_IMAGE="$REGISTRY_BASE/frontend:$TAG"
  export BACKEND_IMAGE="$REGISTRY_BASE/backend:$TAG"
  export INFERENCE_IMAGE="$REGISTRY_BASE/inference:$TAG"
  export STORAGE_IMAGE="$REGISTRY_BASE/seaweedfs:$TAG"
  export POSTGRES_SERVER_IMAGE="$REGISTRY_BASE/postgres:$TAG"
  export POSTGRES_INIT_IMAGE="$REGISTRY_BASE/postgres-init:$TAG"
  export QDRANT_SERVER_IMAGE="$REGISTRY_BASE/qdrant:$TAG"
  export QDRANT_INIT_IMAGE="$REGISTRY_BASE/qdrant-init:$TAG"
  UP_BUILD=--no-build
elif [ "$MODE" = local ]; then
  echo "[run] режим: локальные образы (без сборки, без пулла)"
  export BUILT_PULL_POLICY=never
  UP_BUILD=--no-build
else
  echo "[run] режим: сборка образов (нужен интернет)"
fi

# Секреты: только латиница и цифры (они попадают в JSON и SQL без экранирования)
random_secret() { head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32; }
for dir in postgres vector storage; do
  if [ -f "$dir/.env.example" ] && [ ! -f "$dir/.env" ]; then
    echo "[run] создаю $dir/.env со случайными секретами"
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in
        *=change-me-*) printf '%s=%s\n' "${line%%=*}" "$(random_secret)" ;;
        *) printf '%s\n' "$line" ;;
      esac
    done < "$dir/.env.example" > "$dir/.env"
    chmod 600 "$dir/.env"
  fi
done

# backend читает секреты postgres/vector/storage, но compose подставляет ${VAR} только из .env рядом с compose-файлом сервиса.
# Поэтому backend/.env собираем заново из уже созданных .env (без него backend получил бы dev-заглушки и не вошёл бы в БД).
: > backend/.env
for var in POSTGRES_APP_USER POSTGRES_APP_PASSWORD S3_APP_ACCESS_KEY S3_APP_SECRET_KEY S3_BUCKET QDRANT_API_KEY QDRANT_COLLECTION; do
  grep -h -m1 "^${var}=" postgres/.env vector/.env storage/.env | head -n1 >> backend/.env || true
done
chmod 600 backend/.env

# Долгоживущие сервисы ждём по healthcheck, одноразовые init-сервисы (*-init) выполняем после них:
# `up --wait` считает завершившийся контейнер ошибкой, а init-сервис по замыслу завершается.
all_services=$(docker compose config --services)
# SKIP_SERVICES: сервисы, которые на этой машине поднимать не нужно
skip_services() {
  local service
  while IFS= read -r service; do
    case " ${SKIP_SERVICES:-} " in
      *" $service "*) ;;
      *) printf '%s\n' "$service" ;;
    esac
  done
}
long_running=$(printf '%s\n' "$all_services" | grep -v -- '-init$' | skip_services)
init_services=$(printf '%s\n' "$all_services" | grep -- '-init$' | skip_services || true)
[ -z "${SKIP_SERVICES:-}" ] || echo "[run] пропускаю: $SKIP_SERVICES"

# Порядок важен: роль backend-а в БД создаёт postgres-init, поэтому backend нельзя запускать до init-сервисов
# (на пустой БД он падает при подключении, и `up --wait` завершается ошибкой). Сначала хранилища, затем init, затем остальное.
data_services=""
app_services=""
for service in $long_running; do
  case "$service" in
    postgres|qdrant|seaweedfs) data_services="$data_services $service" ;;
    *) app_services="$app_services $service" ;;
  esac
done

if [ -n "$data_services" ]; then
  echo "[run] запуск хранилищ:$data_services"
  # shellcheck disable=SC2086
  docker compose up -d $UP_BUILD --wait --wait-timeout 300 $data_services
fi

for service in $init_services; do
  echo "[run] инициализация: $service"
  docker compose run --rm --no-deps "$service"
done

if [ -n "$app_services" ]; then
  echo "[run] запуск приложения:$app_services"
  # shellcheck disable=SC2086
  docker compose up -d $UP_BUILD --wait --wait-timeout 300 $app_services
fi

docker compose ps
echo "[run] готово. UI: http://localhost:${FRONTEND_PORT:-8080}"
