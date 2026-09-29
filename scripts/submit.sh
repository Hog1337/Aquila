#!/bin/sh
# Пакетный прогон решения на закрытом тесте (ТЗ п.8) одной командой:
#
#   scripts/submit.sh --input /path/to/test --output /path/to/out [options]
#
# --input должен содержать images/ (плоский каталог кадров) и CSV галереи/запросов
# (по умолчанию test_gallery.csv / test_query.csv).
#
# Поднимает весь стек (если он ещё не запущен), монтирует --input и --output в контейнер backend
# и выполняет там `python -m app.cli submit`, который импортирует данные и сразу формирует
# submission.csv, embeddings.npy, candidates.csv в --output.
#
# Если указан --gt-csv, после экспорта запускается оценка качества (mAP, Rank-1/5).
#
# Флаги:
#   --gt-csv PATH    — ground truth для валидации после экспорта
#   --clear          — очистить volumes перед запуском
#   --no-cleanup     — НЕ удалять данные после завершения (оставить стек запущенным)
#   --rerank         — использовать Query Expansion
#   --threshold N    — порог отказа (по умолчанию 0.87)
#   --gallery-csv NAME, --query-csv NAME — имена CSV внутри --input
set -eu

INPUT=""
OUTPUT=""
EXTRA_ARGS=""
GT_CSV=""
NO_CLEANUP=0
CLEAR=0

while [ $# -gt 0 ]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gt-csv) GT_CSV="$2"; shift 2 ;;
    --threshold) EXTRA_ARGS="$EXTRA_ARGS --threshold $2"; shift 2 ;;
    --gallery-csv) EXTRA_ARGS="$EXTRA_ARGS --gallery-csv $2"; shift 2 ;;
    --query-csv) EXTRA_ARGS="$EXTRA_ARGS --query-csv $2"; shift 2 ;;
    --rerank) EXTRA_ARGS="$EXTRA_ARGS --rerank"; shift ;;
    --no-cleanup) NO_CLEANUP=1; shift ;;
    --clear) CLEAR=1; shift ;;
    *) echo "Неизвестный аргумент: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Использование: $0 --input <каталог с images/+CSV> --output <каталог для результата>" >&2
  echo ""
  echo "Опции:"
  echo "  --gt-csv PATH     ground truth CSV для валидации (image_id,vehicle_id,camera_id,split)"
  echo "  --clear           очистить volumes перед запуском"
  echo "  --no-cleanup      не удалять данные после завершения"
  echo "  --rerank          использовать Query Expansion"
  echo "  --threshold N     порог отказа (по умолчанию 0.87)"
  exit 1
fi
if [ ! -d "$INPUT" ]; then
  echo "Не найден каталог --input: $INPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"

# Абсолютные пути
INPUT_ABS=$(cd "$INPUT" && pwd)
OUTPUT_ABS=$(cd "$OUTPUT" && pwd)
GT_ABS=""
if [ -n "$GT_CSV" ]; then
  GT_ABS=$(cd "$(dirname "$GT_CSV")" && pwd)/$(basename "$GT_CSV")
  if [ ! -f "$GT_ABS" ]; then
    echo "GT CSV не найден: $GT_ABS" >&2
    exit 1
  fi
fi

# MSYS на Windows
if command -v cygpath >/dev/null 2>&1; then
  INPUT_ABS=$(cygpath -w "$INPUT_ABS")
  OUTPUT_ABS=$(cygpath -w "$OUTPUT_ABS")
  [ -n "$GT_ABS" ] && GT_ABS=$(cygpath -w "$GT_ABS")
  export MSYS_NO_PATHCONV=1
fi

cd "$(dirname "$0")/.."   # корень решения (aquila/)

if [ -n "$(docker compose ps -q 2>/dev/null)" ] && [ "$NO_CLEANUP" = 0 ]; then
  echo "[submit] Стек уже запущен: после завершения docker compose down -v сотрёт данные." >&2
fi

# Очистка перед запуском
if [ "$CLEAR" = 1 ]; then
  echo "[submit] Очищаю volumes (docker compose down -v)..."
  docker compose down -v 2>/dev/null || true
fi

cleanup() {
  if [ "$NO_CLEANUP" = 1 ]; then
    echo "[submit] Пропускаю очистку (--no-cleanup). Стек и данные сохранены."
  else
    echo "[submit] Останавливаю стек и удаляю volumes (docker compose down -v)..."
    docker compose down -v 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[submit] Поднимаю стек (./run.sh)..."
./run.sh

echo "[submit] Запускаю python -m app.cli submit --input /input --output /output $EXTRA_ARGS"
# shellcheck disable=SC2086
docker compose run --rm \
  -v "$INPUT_ABS":/input:ro \
  -v "$OUTPUT_ABS":/output \
  backend python -m app.cli submit --input /input --output /output $EXTRA_ARGS

# Оценка качества, если указан GT
if [ -n "$GT_ABS" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  ОЦЕНКА КАЧЕСТВА (--gt-csv)"
  echo "═══════════════════════════════════════════════════════════════"
  # Определяем имена CSV для полной оценки (embeddings + candidates)
  GALLERY_CSV="${EXTRA_ARGS##*--gallery-csv }"  # хрупкий парсинг, но для простоты
  GALLERY_CSV="${GALLERY_CSV%% *}"
  [ -z "$GALLERY_CSV" ] && GALLERY_CSV="test_gallery.csv"
  QUERY_CSV="${EXTRA_ARGS##*--query-csv }"
  QUERY_CSV="${QUERY_CSV%% *}"
  [ -z "$QUERY_CSV" ] && QUERY_CSV="test_query.csv"

  docker compose run --rm \
    -v "$OUTPUT_ABS":/output \
    -v "$GT_ABS":/gt.csv:ro \
    -v "$INPUT_ABS":/input:ro \
    backend python -m app.cli evaluate \
      --gt /gt.csv \
      --submission /output/submission.csv \
      --candidates /output/candidates.csv \
      --embeddings /output/embeddings.npy \
      --query "/input/$QUERY_CSV" \
      --gallery "/input/$GALLERY_CSV" \
      --json /output/evaluation_report.json
  echo "═══════════════════════════════════════════════════════════════"
fi

echo "[submit] Готово. Результат в $OUTPUT_ABS"
