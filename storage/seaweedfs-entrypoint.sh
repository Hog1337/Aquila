#!/bin/sh
# Точка входа SeaweedFS для галереи ReID.
#   1. генерирует s3.json из переменных окружения (секреты не хранятся в образе и в git);
#   2. запускает `weed server` (master + volume + filer + S3-шлюз) в одном контейнере;
#   3. создаёт бакет галереи и только после этого отдаёт флаг готовности healthcheck-у.
# Ключи должны состоять из латиницы и цифр: они подставляются в JSON без экранирования.
set -eu

: "${S3_BUCKET:?не задан S3_BUCKET}"
: "${S3_APP_ACCESS_KEY:?не задан S3_APP_ACCESS_KEY}"
: "${S3_APP_SECRET_KEY:?не задан S3_APP_SECRET_KEY}"
: "${S3_ADMIN_ACCESS_KEY:?не задан S3_ADMIN_ACCESS_KEY}"
: "${S3_ADMIN_SECRET_KEY:?не задан S3_ADMIN_SECRET_KEY}"

CONF=/tmp/s3.json
READY=/tmp/bucket.ready
rm -f "$READY"

umask 077
cat > "$CONF" <<EOF
{
  "identities": [
    {
      "name": "reid-backend",
      "credentials": [{"accessKey": "${S3_APP_ACCESS_KEY}", "secretKey": "${S3_APP_SECRET_KEY}"}],
      "actions": ["Read:${S3_BUCKET}", "Write:${S3_BUCKET}", "List:${S3_BUCKET}", "Tagging:${S3_BUCKET}"]
    },
    {
      "name": "admin",
      "credentials": [{"accessKey": "${S3_ADMIN_ACCESS_KEY}", "secretKey": "${S3_ADMIN_SECRET_KEY}"}],
      "actions": ["Admin", "Read", "Write", "List", "Tagging"]
    }
  ]
}
EOF

# volume.max=0: число томов выбирается автоматически по свободному месту на диске
weed server \
  -dir=/data \
  -ip.bind=0.0.0.0 \
  -master.volumeSizeLimitMB="${VOLUME_SIZE_LIMIT_MB:-1024}" \
  -volume.max=0 \
  -filer \
  -s3 \
  -s3.config="$CONF" &
WEED_PID=$!

# Корректная остановка: docker stop шлёт SIGTERM PID 1 (этому скрипту), пробрасываем его в weed
trap 'kill -TERM "$WEED_PID" 2>/dev/null || true' TERM INT

create_bucket() {
  i=0
  while [ "$i" -lt 60 ]; do
    echo "s3.bucket.create -name ${S3_BUCKET}" \
      | weed shell -master=127.0.0.1:9333 -filer=127.0.0.1:8888 >/dev/null 2>&1 || true
    # Истина — список бакетов: повторный старт (бакет уже есть) тоже проходит эту проверку
    if echo "s3.bucket.list" \
         | weed shell -master=127.0.0.1:9333 -filer=127.0.0.1:8888 2>/dev/null \
         | grep -qw "${S3_BUCKET}"; then
      touch "$READY"
      echo "[init] бакет '${S3_BUCKET}' готов"
      return 0
    fi
    i=$((i + 1))
    sleep 2
  done
  echo "[init] не удалось создать бакет '${S3_BUCKET}' за 120 с" >&2
  kill -TERM "$WEED_PID" 2>/dev/null || true
  return 1
}

create_bucket &

# Первый wait прерывается сигналом, второй дожидается фактического выхода weed
wait "$WEED_PID" || true
wait "$WEED_PID" 2>/dev/null
