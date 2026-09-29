#!/usr/bin/env bash
# Регулярные резервные копии через cron текущего пользователя (на демо-хосте aquila, Linux).
#   scripts/backup-schedule.sh install ["30 3 * * *"]   включить (по умолчанию каждую ночь в 03:30)
#   scripts/backup-schedule.sh remove                   выключить
#   scripts/backup-schedule.sh status                   показать запись cron и последние копии
#
# Копия холодная: на время архивации стек недоступен, поэтому расписание лучше ставить на ночь.
# Каталог копий и число хранимых копий берутся из BACKUP_DIR и BACKUP_KEEP на момент install
# (BACKUP_DIR=/mnt/disk2/falcon-backups BACKUP_KEEP=14 scripts/backup-schedule.sh install).
# Журнал: <каталог копий>/backup.log.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=lib.sh
. scripts/lib.sh

MARK="# falcon-reid-backup"
ROOT=$(pwd)
cmd="${1:-status}"

command -v crontab >/dev/null || { echo "[schedule] нет crontab: установите cron или запускайте scripts/backup.sh вручную" >&2; exit 1; }

current() { crontab -l 2>/dev/null | grep -vF "$MARK" || true; }

case "$cmd" in
  install)
    when="${2:-30 3 * * *}"
    mkdir -p "$BACKUP_ROOT"
    log=$(cd "$BACKUP_ROOT" && pwd)/backup.log
    # Значения с пробелами и спецсимволами в cron-строке экранируем через printf %q
    line=$(printf '%s cd %q && BACKUP_DIR=%q BACKUP_KEEP=%q scripts/backup.sh >> %q 2>&1 %s' \
      "$when" "$ROOT" "$BACKUP_ROOT" "$BACKUP_KEEP" "$log" "$MARK")
    { current; echo "$line"; } | crontab -
    echo "[schedule] включено: $when, храню $BACKUP_KEEP копий в $BACKUP_ROOT, журнал $log"
    ;;
  remove)
    current | crontab -
    echo "[schedule] выключено"
    ;;
  status)
    if crontab -l 2>/dev/null | grep -F "$MARK"; then :; else echo "[schedule] расписание не включено"; fi
    if [ -d "$BACKUP_ROOT" ]; then
      echo "[schedule] копии в $BACKUP_ROOT:"
      find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]{8}-[0-9]{6}$' | sort -r | head -n 10 || true
    fi
    ;;
  *)
    echo "использование: $0 install [\"cron-выражение\"] | remove | status" >&2
    exit 2
    ;;
esac
