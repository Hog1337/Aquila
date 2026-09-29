# shellcheck shell=bash disable=SC2034  # переменные используются скриптами, которые подключают этот файл
# Общие функции резервного копирования (подключается через `. scripts/lib.sh`).
# Ключи томов в docker-compose: данные каждого хранилища лежат в отдельном именованном томе.
# Имя проекта у корня и всех частей одно (`falcon-reid`), поэтому тома называются falcon-reid_<ключ>
# при любом способе запуска.
VOLUME_KEYS="postgres-data qdrant-data seaweedfs-data"
COMPOSE_PROJECT=falcon-reid
HELPER_IMAGE=python:3.12-alpine   # нужен только tar; если образа нет на хосте, Docker скачает его при первом копировании

# Куда складывать копии (можно указать другой диск или смонтированный сетевой ресурс) и сколько свежих хранить (0 = все)
BACKUP_ROOT="${BACKUP_DIR:-backups}"
BACKUP_KEEP="${BACKUP_KEEP:-7}"
# Секреты сервисов: без них восстановленные данные не открыть (пароли БД, ключи S3 и Qdrant)
ENV_FILES=".env postgres/.env vector/.env storage/.env"

# Оставляет $2 самых свежих копий вида ГГГГММДД-ЧЧММСС в каталоге $1, остальные удаляет.
prune_backups() {
  local root="$1" keep="$2" name
  case "$keep" in ''|*[!0-9]*) echo "[backup] BACKUP_KEEP должен быть числом, получено: $keep" >&2; return 1 ;; esac
  [ "$keep" -gt 0 ] || return 0
  while read -r name; do
    echo "[backup] удаляю старую копию $name"
    rm -rf "${root:?}/$name"
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]{8}-[0-9]{6}$' | sort -r | tail -n +$((keep + 1)))
}

# Имя существующего тома по ключу; код возврата 1, если тома нет.
volume_of() {
  local name="${COMPOSE_PROJECT}_$1"
  if docker volume inspect "$name" >/dev/null 2>&1; then
    echo "$name"
    return 0
  fi
  return 1
}
