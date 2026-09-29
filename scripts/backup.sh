#!/usr/bin/env bash
# Резервная копия всех данных решения: PostgreSQL (каталог, история, порог), Qdrant (эмбеддинги), SeaweedFS (кадры)
# и секреты сервисов (.env), без которых восстановленные данные не открыть.
#   scripts/backup.sh [каталог]     по умолчанию $BACKUP_DIR/<дата-время> (BACKUP_DIR по умолчанию backups)
#
# Копия холодная: на время архивации сервисы останавливаются (секунды или минуты в зависимости от объёма),
# зато в архиве согласованное состояние всех трёх хранилищ. Дополнительно снимается логический дамп PostgreSQL
# (переносимый между версиями). Вспомогательный контейнер использует образ python:3.12-alpine (при отсутствии на хосте Docker скачает его).
#
# Копия собирается в <каталог>.partial и переименовывается только после проверки архивов, поэтому оборванная
# копия никогда не выглядит готовой. После успеха хранятся BACKUP_KEEP свежих копий (по умолчанию 7, 0 = не удалять).
# Одновременно работает один запуск (блокировка BACKUP_DIR/.lock), поэтому его можно вешать на cron: см. scripts/backup-schedule.sh.
# Копия содержит секреты (права 700/600): храните её так же бережно, как сами .env.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=lib.sh
. scripts/lib.sh

umask 077
mkdir -p "$BACKUP_ROOT"

custom=1
if [ -z "${1:-}" ]; then custom=0; fi
OUT="${1:-$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)}"
[ ! -e "$OUT" ] || { echo "[backup] $OUT уже существует" >&2; exit 1; }
TMP="$OUT.partial"

# Блокировка: PID в файле, чтобы после аварийного завершения (kill -9, перезагрузка) она не блокировала копии навсегда
lock="$BACKUP_ROOT/.lock"
if ! mkdir "$lock" 2>/dev/null; then
  if kill -0 "$(cat "$lock/pid" 2>/dev/null)" 2>/dev/null; then
    echo "[backup] уже выполняется другая копия (PID $(cat "$lock/pid"))" >&2
    exit 1
  fi
  echo "[backup] снимаю устаревшую блокировку"
  rm -rf "$lock"
  mkdir "$lock"
fi
echo $$ > "$lock/pid"

stopped=0
running=()
cleanup() {
  local rc=$?
  if [ "$stopped" = 1 ]; then
    echo "[backup] запускаю сервисы"
    docker compose start "${running[@]}" >/dev/null || rc=1
  fi
  rm -rf "$lock"
  if [ "$rc" -ne 0 ]; then
    rm -rf "$TMP"
    echo "[backup] ОШИБКА: копия не создана" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT

mkdir -p "$TMP"
TMP_ABS=$(cd "$TMP" && pwd)

# Секреты сервисов
mkdir -p "$TMP_ABS/env"
for f in $ENV_FILES; do
  if [ -f "$f" ]; then
    mkdir -p "$TMP_ABS/env/$(dirname "$f")"
    cp "$f" "$TMP_ABS/env/$f"
  fi
done

# Логический дамп: горячий, до остановки сервисов
if docker compose ps --status running --services | grep -qx postgres; then
  echo "[backup] pg_dump -> postgres.dump"
  docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB"' > "$TMP_ABS/postgres.dump"
  [ -s "$TMP_ABS/postgres.dump" ] || { echo "[backup] pg_dump вернул пустой файл" >&2; exit 1; }
fi

# Останавливаем и потом запускаем только то, что работало (после `down` копию можно снять и без сервисов;
# одноразовые init-контейнеры повторно не запускаются)
while IFS= read -r svc; do
  svc=${svc%$'\r'}
  if [ -n "$svc" ]; then running+=("$svc"); fi
done < <(docker compose ps --status running --services)
if [ "${#running[@]}" -gt 0 ]; then
  echo "[backup] останавливаю сервисы: ${running[*]}"
  docker compose stop "${running[@]}" >/dev/null
  stopped=1
fi

for key in $VOLUME_KEYS; do
  volume=$(volume_of "$key") || { echo "[backup] нет тома $key, пропускаю"; continue; }
  echo "[backup] $volume"
  docker run --rm --pull=never -v "$volume:/data:ro" -v "$TMP_ABS:/backup" "$HELPER_IMAGE" \
    tar czf "/backup/$key.tar.gz" -C /data .
done

# Сервисы можно запускать: дальше только проверка и контрольные суммы
if [ "$stopped" = 1 ]; then
  echo "[backup] запускаю сервисы"
  docker compose start "${running[@]}" >/dev/null
  stopped=0
fi

ls "$TMP_ABS"/*.tar.gz >/dev/null 2>&1 || { echo "[backup] ни одного тома не найдено, копия пуста" >&2; exit 1; }
for archive in "$TMP_ABS"/*.tar.gz; do
  gzip -t "$archive" || { echo "[backup] повреждён архив $archive" >&2; exit 1; }
done

( cd "$TMP_ABS"
  files=(./*.tar.gz)
  if [ -f postgres.dump ]; then files+=(postgres.dump); fi
  while IFS= read -r f; do files+=("$f"); done < <(find env -type f | sort)
  sha256sum "${files[@]}" > SHA256SUMS )

mv "$TMP" "$OUT"
echo "[backup] готово: $OUT ($(du -sh "$OUT" | cut -f1)); работали: ${running[*]:-ничего}"

if [ "$custom" = 0 ]; then prune_backups "$BACKUP_ROOT" "$BACKUP_KEEP"; fi
