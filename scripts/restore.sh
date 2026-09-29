#!/usr/bin/env bash
# Восстановление данных из резервной копии, сделанной scripts/backup.sh.
#   scripts/restore.sh <каталог копии>
#
# ВНИМАНИЕ: текущее содержимое томов PostgreSQL, Qdrant и SeaweedFS будет заменено.
# Сервисы на время восстановления останавливаются и затем запускаются снова.
set -euo pipefail

cd "$(dirname "$0")/.."
SRC="${1:?укажите каталог резервной копии: scripts/restore.sh backups/<дата-время>}"
[ -d "$SRC" ] || { echo "[restore] нет каталога $SRC" >&2; exit 1; }
SRC_ABS=$(cd "$SRC" && pwd)
# shellcheck source=lib.sh
. scripts/lib.sh

( cd "$SRC_ABS" && sha256sum --quiet -c SHA256SUMS ) || { echo "[restore] контрольные суммы не совпали" >&2; exit 1; }

# Останавливаем и потом запускаем только то, что работало (после `down` или на новом хосте контейнеров нет:
# тогда данные просто кладутся в тома, а стек поднимает `make up`)
running=()
while IFS= read -r svc; do
  svc=${svc%$'\r'}
  if [ -n "$svc" ]; then running+=("$svc"); fi
done < <(docker compose ps --status running --services)
if [ "${#running[@]}" -gt 0 ]; then
  echo "[restore] останавливаю сервисы: ${running[*]}"
  docker compose stop "${running[@]}" >/dev/null
fi

for key in $VOLUME_KEYS; do
  archive="$SRC_ABS/$key.tar.gz"
  [ -f "$archive" ] || { echo "[restore] в копии нет $key, пропускаю"; continue; }
  # Существующий том очищаем; если его нет (после `down -v`), создаём с метками compose, чтобы он подхватился
  volume=$(volume_of "$key" || true)
  if [ -z "$volume" ]; then
    volume="${COMPOSE_PROJECT}_$key"
    docker volume create --label "com.docker.compose.project=$COMPOSE_PROJECT" \
      --label "com.docker.compose.volume=$key" "$volume" >/dev/null
  fi
  echo "[restore] $key -> $volume"
  docker run --rm --pull=never -v "$volume:/data" -v "$SRC_ABS:/backup:ro" "$HELPER_IMAGE" \
    sh -c 'find /data -mindepth 1 -delete && tar xzf "/backup/$0.tar.gz" -C /data' "$key"
done

# Секреты: недостающие .env возвращаем из копии, существующие не перезаписываем (на новом хосте run.sh уже мог
# сгенерировать другие пароли: если сервисы не подключаются к БД или S3, возьмите нужный файл из <копия>/env/)
for f in $ENV_FILES; do
  [ -f "$SRC_ABS/env/$f" ] || continue
  if [ ! -f "$f" ]; then
    mkdir -p "$(dirname "$f")"
    cp "$SRC_ABS/env/$f" "$f"
    chmod 600 "$f"
    echo "[restore] вернул $f из копии"
  elif ! cmp -s "$SRC_ABS/env/$f" "$f"; then
    echo "[restore] ВНИМАНИЕ: $f отличается от копии, оставил текущий (копия: $SRC/env/$f)" >&2
  fi
done

if [ "${#running[@]}" -gt 0 ]; then
  echo "[restore] запускаю сервисы"
  docker compose start "${running[@]}" >/dev/null
else
  echo "[restore] сервисы не были запущены: поднимите стек командой make up"
fi
echo "[restore] готово"
